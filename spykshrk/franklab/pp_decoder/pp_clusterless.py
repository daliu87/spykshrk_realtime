import functools
import logging

import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import warnings

import torch

from spykshrk.franklab.data_containers import LinearPosition, FlatLinearPosition, SpikeObservation, EncodeSettings, DecodeSettings, Posteriors, pos_col_format
from spykshrk.franklab.pp_decoder.util import gaussian, normal2D, apply_no_anim_boundary, normal_pdf_int_lookup, normal_pdf_int_lookup_torch
from spykshrk.util import Groupby, AttrDict

#logger = logging.getLogger(__name__)

class OfflinePPEncoder(object):

    def __init__(self, linpos, enc_spk_amp, dec_spk_amp, encode_settings: EncodeSettings,
                 decode_settings: DecodeSettings, linpos_col_name='linpos_flat', chunk_size=1000,
                 cuda=False, norm=True):
        """
        Constructor for OfflinePPEncoder.
        
        Args:
            linflat (FlatLinearPosition): Observered 1D unique animal position
            enc_spk_amp (SpikeObservation): Observered spikes in encoding model
            dec_spk_amp (SpikeObservation): Observered spikes in decoding model
            encode_settings (EncodeSettings): Realtime encoder settings.
            decode_settings (DecodeSettings): Realtime decoder settings.

        """
        self.linpos = linpos
        self.linpos_col_name = linpos_col_name
        self.enc_spk_amp = enc_spk_amp
        self.dec_spk_amp = dec_spk_amp
        self.encode_settings = encode_settings
        self.decode_settings = decode_settings
        self.chunk_size = chunk_size
        if cuda:
            self.device_name = 'cuda'
        else:
            self.device_name = 'cpu'
        self.dtype = torch.float
        self.device = torch.device(self.device_name)
        self.norm = norm

        self.calc_occupancy()
        self.calc_firing_rate()
        self.calc_prob_no_spike()

        self.trans_mat = dict.fromkeys(['learned', 'simple', 'uniform'])
        self.trans_mat['learned'] = self.calc_learned_state_trans_mat(self.linpos, self.encode_settings,
                                                                      self.decode_settings,
                                                                      linflat_col_name=self.linpos_col_name)
        self.trans_mat['simple'] = self.calc_simple_trans_mat(self.encode_settings)
        self.trans_mat['uniform'] = self.calc_uniform_trans_mat(self.encode_settings)

    def run_encoder(self):

        self.results = self._run_loop()

        tet_ids = np.unique(self.enc_spk_amp.index.get_level_values('elec_grp_id'))
        observ_tet_list = []
        grp = self.dec_spk_amp.groupby('elec_grp_id')
        for tet_ii, (tet_id, grp_spk) in enumerate(grp):
            tet_result = self.results[tet_id]
            tet_result.set_index(grp_spk.index, inplace=True)
            observ_tet_list.append(tet_result)

        observ = pd.concat(observ_tet_list)
        self.observ_obj = SpikeObservation.create_default(observ.sort_index(level=['day', 'epoch',
                                                                            'timestamp', 'elec_grp_id']),
                                                          self.encode_settings)

        self.observ_obj['elec_grp_id'] = self.observ_obj.index.get_level_values('elec_grp_id')
        self.observ_obj.index = self.observ_obj.index.droplevel('elec_grp_id')

        self.observ_obj['position'] = (self.linpos.get_irregular_resampled(self.observ_obj).
            get_mapped_single_axis()[self.linpos_col_name])

        self.observ_obj.set_distribution(self.observ_obj.get_distribution_as_np() + np.finfo(float).eps)

        return self.observ_obj

    def _run_loop(self):
        # grp = self.spk_amp.groupby('elec_grp_id')
        dec_grp = self.dec_spk_amp.groupby('elec_grp_id')
        observations = {}
        task = []

        for dec_tet_id, dec_spk_tet in dec_grp:
            observations.setdefault(dec_tet_id, pd.DataFrame())
            enc_spk_tet = self.enc_spk_amp.query('elec_grp_id==@dec_tet_id')
            
            if len(enc_spk_tet) == 0 | len(dec_spk_tet) == 0:
                continue
            enc_tet_linpos_resamp = self.linpos.get_irregular_resampled(enc_spk_tet)
            if len(enc_tet_linpos_resamp) == 0:
                continue
            # maintain elec_grp_id info. get mark and index column names to reindex dask arrays
            mark_columns = dec_spk_tet.columns
            index_columns = dec_spk_tet.index.names
            #dask_dec_spk_tet = dd.from_pandas(dec_spk_tet.reset_index(), chunksize=self.dask_chunksize)

            chunk_start_ii = -self.chunk_size
            for chunk_start_ii in list(range(0, len(dec_spk_tet), self.chunk_size))[:-1]:
                dec_spk_tet_chunk = dec_spk_tet.iloc[chunk_start_ii:chunk_start_ii + self.chunk_size].reset_index()
                

            #df_meta = pd.DataFrame([], columns=[pos_col_format(ii, self.encode_settings.pos_num_bins)
            #                                    for ii in range(self.encode_settings.pos_num_bins)])

                observ = self.compute_observ_tet(dec_spk=dec_spk_tet_chunk, enc_spk=enc_spk_tet,
                                                 tet_linpos_resamp=enc_tet_linpos_resamp, occupancy=self.occupancy,
                                                 encode_settings=self.encode_settings,
                                                 mark_columns=mark_columns, index_columns=index_columns,
                                                 device=self.device, dtype=self.dtype, norm=self.norm)
                observations[dec_tet_id] = observations[dec_tet_id].append(observ)

            chunk_start_ii = chunk_start_ii + self.chunk_size
            dec_spk_tet_chunk = dec_spk_tet.iloc[chunk_start_ii:].reset_index()

            observ = self.compute_observ_tet(dec_spk=dec_spk_tet_chunk, enc_spk=enc_spk_tet,
                                             tet_linpos_resamp=enc_tet_linpos_resamp, occupancy=self.occupancy,
                                             encode_settings=self.encode_settings,
                                             mark_columns=mark_columns, index_columns=index_columns,
                                             device=self.device, dtype=self.dtype, norm=self.norm)
            observations[dec_tet_id] = observations[dec_tet_id].append(observ)
            # setup decode of decode spikes from encoding of encoding spikes
        return observations

    def compute_observ_tet(self, dec_spk, enc_spk, tet_linpos_resamp, occupancy, encode_settings,
                           mark_columns, index_columns, device, dtype, norm):
        pos_distrib_tet = sp.stats.norm.pdf(np.expand_dims(encode_settings.pos_bins, 0),
                                            np.expand_dims(tet_linpos_resamp[self.linpos_col_name], 1),
                                            encode_settings.pos_kernel_std)
        if device.type == 'cuda':
            pos_distrib_tet_torch = torch.from_numpy(pos_distrib_tet).to(device=device, dtype=dtype)
            mark_contrib = normal_pdf_int_lookup_torch(np.expand_dims(dec_spk[mark_columns], 1),
                                                       np.expand_dims(enc_spk, 0),
                                                       encode_settings.mark_kernel_std, 
                                                       device=device, dtype=dtype)
            all_contrib = torch.prod(mark_contrib, dim=2, dtype=torch.float)
            del mark_contrib
            observ_torch = torch.matmul(all_contrib, pos_distrib_tet_torch)
            del all_contrib
            observ = observ_torch.to(device='cpu').numpy()
            del observ_torch

        else:
            mark_contrib = normal_pdf_int_lookup(np.expand_dims(dec_spk[mark_columns], 1),
                                                 np.expand_dims(enc_spk, 0),
                                                 encode_settings.mark_kernel_std)
            all_contrib = np.prod(mark_contrib, axis=2)
            del mark_contrib
            observ = np.matmul(all_contrib, pos_distrib_tet)
            del all_contrib

        # occupancy normalize 
        observ = observ / (occupancy)

        if norm:
            # normalize factor for each row (#dec spks x #pos_bins)
            observ_sum = np.nansum(observ, axis=1)

            # replace all rows that are all zeros with uniform distribution
            observ_sum_zero = observ_sum == 0
            observ[observ_sum_zero, :] = 1/(self.encode_settings.pos_bins[-1] - self.encode_settings.pos_bins[0])
            observ_sum[observ_sum_zero] = 1

            # apply normalization factor
            observ = observ / observ_sum[:, np.newaxis]
        else:
            pass

        ret_df = pd.DataFrame(observ, index=dec_spk.set_index(index_columns).index,
                              columns=pos_col_format(range(observ.shape[1]), observ.shape[1]))
        return ret_df

    def calc_occupancy(self):
        self.occupancy = self._calc_occupancy(self.linpos, self.encode_settings,
                                              linflat_col_name=self.linpos_col_name)

    def calc_firing_rate(self):
        self.firing_rate = self._calc_firing_rate_tet(self.enc_spk_amp, self.linpos, self.encode_settings,
                                                      linflat_col_name=self.linpos_col_name)

    def calc_prob_no_spike(self):
        self.prob_no_spike = self._calc_prob_no_spike(self.firing_rate, self.occupancy,
                                                      self.encode_settings, self.decode_settings)

    @staticmethod
    def _calc_occupancy(lin_obj: FlatLinearPosition, enc_settings: EncodeSettings, linflat_col_name='linpos_flat'):
        """
        Args:
            lin_obj (LinearPositionContainer): Linear position of the animal.
            enc_settings (EncodeSettings): Realtime encoding settings.
        Returns (np.array): The occupancy of the animal
        """
        occupancy, occ_bin_edges = np.histogram(lin_obj[linflat_col_name], bins=enc_settings.pos_bin_edges)
        occupancy = np.convolve(occupancy, enc_settings.pos_kernel, mode='same')

        # occupancy
        occupancy = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates, occupancy, np.nan)
        occupancy += 0.1
        occupancy /= enc_settings.pos_sampling_rate
        return occupancy

    @staticmethod
    def _calc_firing_rate_tet(observ: SpikeObservation, lin_obj: FlatLinearPosition, enc_settings: EncodeSettings,
                              linflat_col_name='linpos_flat'):
        # initialize conditional intensity function
        firing_rate = {}
        enc_tet_lin_pos = (lin_obj.get_irregular_resampled(observ))
        #enc_tet_lin_pos['elec_grp_id'] = observ.index.get_level_values(level='elec_grp_id')
        tet_pos_groups = enc_tet_lin_pos.loc[:, linflat_col_name].groupby('elec_grp_id')
        for tet_id, tet_spikes in tet_pos_groups:
            tet_pos_hist, _ = np.histogram(tet_spikes, bins=enc_settings.pos_bin_edges)
            firing_rate[tet_id] = tet_pos_hist
        for fr_key in firing_rate.keys():
            firing_rate[fr_key] = np.convolve(firing_rate[fr_key], enc_settings.pos_kernel, mode='same')
            firing_rate[fr_key] = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates,
                                                         firing_rate[fr_key])
            #firing_rate[fr_key] = firing_rate[fr_key] / (firing_rate[fr_key].sum() * enc_settings.pos_bin_delta)
            #firing_rate[fr_key] += 1
        return firing_rate

    @staticmethod
    def _calc_prob_no_spike(firing_rate: dict, occupancy, enc_settings: EncodeSettings, dec_settings: DecodeSettings):
        """
        
        Args:
            firing_rate (pd.DataFrame): Occupancy firing rate, from calc_observation_intensity(...).
            occupancy (np.array): The occupancy of the animal.
            enc_settings (EncodeSettings): Realtime encode settings.
            dec_settings (DecodeSettings): Realtime decoding settings.

        Returns (dict[int, np.array]): Dictionary of probability that no spike occured per tetrode.

        """
        prob_no_spike = {}
        for tet_id, tet_fr in firing_rate.items():
            prob_no_spike[tet_id] = np.exp(-dec_settings.time_bin_size/enc_settings.sampling_rate * tet_fr / occupancy)
        return prob_no_spike

    
    @staticmethod
    def calc_learned_state_trans_mat(linpos_simple, enc_settings, dec_settings, linflat_col_name='linpos_flat'):
        """
        Calculate the point process transition matrix using the real behavior of the animal.
        This is the 2D matrix that defines the possible range of transitions in the position
        estimate.
        
        The learned values are smoothed with a gaussian kernel and a uniform offset is added,
        specified by the encoding config.  The matrix is column normalized.
        
        Args:
            linpos_simple (pd.DataFrame): Linear position pandas table with no MultiIndex.
            enc_settings (EncodeSettings): Encoder settings from a realtime config.
            dec_settings (DecodeSettings): Decoder settings from a realtime config.

        Returns: Learned transition matrix.

        """
        pos_num_bins = len(enc_settings.pos_bins)

        # Smoothing kernel for learned pos transition matrix
        xv, yv = np.meshgrid(np.arange(-20, 21), np.arange(-20, 21))
        kernel = normal2D(xv, yv, dec_settings.trans_smooth_std)
        kernel /= kernel.sum()

        linpos_state = linpos_simple[linflat_col_name]
        linpos_ind = np.searchsorted(enc_settings.pos_bins, linpos_state, side='right') - 1

        # Create learned pos transition matrix
        learned_trans_mat = np.zeros([pos_num_bins, pos_num_bins])
        for first_pos_ind, second_pos_ind in zip(linpos_ind[:-1], linpos_ind[1:]):
            learned_trans_mat[first_pos_ind, second_pos_ind] += 1

        # normalize
        learned_trans_mat = learned_trans_mat / (learned_trans_mat.sum(axis=0)[None, :])
        learned_trans_mat[np.isnan(learned_trans_mat)] = 0

        # smooth
        learned_trans_mat = sp.signal.fftconvolve(learned_trans_mat, kernel, mode='same')
        learned_trans_mat = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates,
                                                   learned_trans_mat)

        # uniform offset
        uniform_gain = dec_settings.trans_uniform_gain
        uniform_dist = np.ones(learned_trans_mat.shape)

        # no-animal boundary
        uniform_dist = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates, uniform_dist)

        # normalize uniform offset
        uniform_dist = uniform_dist / (uniform_dist.sum(axis=0)[None, :])
        uniform_dist[np.isnan(uniform_dist)] = 0

        # apply uniform offset
        learned_trans_mat = learned_trans_mat * (1 - uniform_gain) + uniform_dist * uniform_gain

        # renormalize
        #learned_trans_mat = learned_trans_mat / (learned_trans_mat.sum(axis=0)[None, :])
        #learned_trans_mat[np.isnan(learned_trans_mat)] = 0

        # 2D normalization at end
        learned_trans_mat = learned_trans_mat / learned_trans_mat.sum()
        learned_trans_mat[np.isnan(learned_trans_mat)] = 0

        return learned_trans_mat

    @staticmethod
    def calc_simple_trans_mat(enc_settings):
        """
        Calculate a simple point process transition matrix using a gaussian kernel.
        
        Args:
            enc_settings (EncodeSettings): Encoder setting from realtime config.

        Returns (np.array): Simple gaussian transition matrix.

        """
        pos_num_bins = len(enc_settings.pos_bins)

        # Setup transition matrix
        transition_mat = np.ones([pos_num_bins, pos_num_bins])
        for bin_ii in range(pos_num_bins):
            transition_mat[bin_ii, :] = gaussian(enc_settings.pos_bins, enc_settings.pos_bins[bin_ii], 3)

        transition_mat = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates,
                                                transition_mat)

        # uniform offset
        uniform_gain = 0.01
        uniform_dist = np.ones(transition_mat.shape)
        uniform_dist = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates,
                                              uniform_dist)

        # normalize transition matrix
        transition_mat = transition_mat/(transition_mat.sum(axis=0)[None, :])
        transition_mat[np.isnan(transition_mat)] = 0

        # normalize uniform offset
        uniform_dist = uniform_dist/(uniform_dist.sum(axis=0)[None, :])
        uniform_dist[np.isnan(uniform_dist)] = 0

        # apply uniform offset
        transition_mat = transition_mat * (1 - uniform_gain) + uniform_dist * uniform_gain

        return transition_mat

    @staticmethod
    def calc_uniform_trans_mat(enc_settings):
        """
        Calculate a simple uniform point process transition matrix.
        
        Args:
            enc_settings (EncodeSettings): Encoder setting from realtime config.

        Returns (np.array): Simple uniform transition matrix.

        """

        pos_num_bins = len(enc_settings.pos_bins)

        # Setup transition matrix
        transition_mat = np.ones([pos_num_bins, pos_num_bins])
        transition_mat = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates, transition_mat)

        # normalize transition matrix
        transition_mat = transition_mat/(transition_mat.sum(axis=0)[None, :])

        transition_mat = np.nan_to_num(transition_mat)

        return transition_mat
    

class OfflinePPLikelihood:
    def __init__(self, observ: SpikeObservation, encode_settings: EncodeSettings,
                 decode_settings: DecodeSettings, time_bin_size=30,
                 trans_mat=None, prob_no_spike=None,
                 cuda=False, dtype=np.float32):

        if prob_no_spike:
            self.prob_no_spike = prob_no_spike
        else:
            self.prob_no_spike = {tet_id: np.ones(encode_settings.pos_num_bins) for tet_id in encode_settings.tetrodes}
        self.trans_mat = trans_mat
        self.observ = observ
        self.encode_settings = encode_settings
        self.decode_settings = decode_settings
        # self.which_trans_mat = which_trans_mat
        self.time_bin_size = time_bin_size

        self.cuda = cuda
        if self.cuda:
            self.device_name = 'cuda'
        else:
            self.device_name = 'cpu'
        self.dtype = torch.float
        self.device = torch.device(self.device_name)

        self.dtype = dtype

        self.likelihoods = None

    def calc(self):
        print("Beginning likelihood calculation")
        self.recalc_likelihood()

        return self.likelihoods

    def recalc_likelihood(self):
        self.likelihoods = self.calc_observation_intensity(self.observ,
                                                           self.prob_no_spike,
                                                           self.encode_settings,
                                                           self.decode_settings,
                                                           time_bin_size=self.time_bin_size,
                                                           dtype=self.dtype)

    @staticmethod
    def calc_observation_intensity(observ: SpikeObservation,
                                   prob_no_spike,
                                   enc_settings: EncodeSettings,
                                   dec_settings: DecodeSettings,
                                   time_bin_size=None,
                                   dtype=np.float32):
        """

        Args:
            observ (SpikeObservation): Object containing observation data frame, one row per spike observed.
            enc_settings (EncodeSettings): Encoder settings from realtime config.
            dec_settings (DecodeSettings): Decoder settings from realtime config.
            time_bin_size (float): Delta time per bin.

        Returns: (pd.DataFrame, dict[int, np.array]) DataFrame of observation per time bin in each row.
            Dictionary of numpy arrays, one per tetrode, containing occupancy firing rate.

        """

        day = observ.index.get_level_values('day')[0]
        epoch = observ.index.get_level_values('epoch')[0]

        if time_bin_size is not None:
            observ.update_observations_bins(time_bin_size=time_bin_size, inplace=True)
        else:
            time_bin_size = dec_settings.time_bin_size
            observ.update_observations_bins(time_bin_size=time_bin_size, inplace=True)

        observ.update_parallel_bins(60000, inplace=True)

        observ.update_num_missing_future_bins(inplace=True)

        #observ_dask = dd.from_pandas(observ.get_no_multi_index(), chunksize=30000)
        observ_grp = observ.groupby('parallel_bin')

        observ_meta = [(key, 'f8') for key in [pos_col_format(ii, enc_settings.pos_num_bins)
                                               for ii in range(enc_settings.pos_num_bins)]]
        observ_meta.append(('timestamp', 'f8'))
        observ_meta.append(('num_spikes', 'f8'))
        observ_meta.append(('dec_bin', 'f8'))

        elec_grp_list = observ['elec_grp_id'].unique()

        dec_agg_results = pd.DataFrame()
        for parallel_bin_ind, observ_par in observ_grp:

            bin_observed_par = OfflinePPLikelihood._calc_observation_single_bin(observ_par,
                                                                                elec_grp_list=elec_grp_list,
                                                                                prob_no_spike=prob_no_spike,
                                                                                time_bin_size=time_bin_size,
                                                                                enc_settings=enc_settings,
                                                                                dtype=dtype),
            dec_agg_results = dec_agg_results.append(bin_observed_par[0])

        dec_agg_results.sort_values('timestamp', inplace=True)

        #encoding mask: convert timestamps to int in order to run get_irregular_resample
        dec_agg_results = dec_agg_results.astype({'timestamp': int})

        dec_new_ind = pd.MultiIndex.from_product([[day], [epoch], dec_agg_results['timestamp']])
        lev = list(dec_new_ind.levels)
        lab = list(dec_new_ind.codes)

        lev.append(dec_agg_results['timestamp']/float(enc_settings.sampling_rate))
        dec_agg_results.drop(columns='timestamp', inplace=True)

        lab.append(range(len(dec_agg_results)))

        dec_new_ind = pd.MultiIndex(levels=lev, codes=lab, names=['day', 'epoch', 'timestamp', 'time'])

        dec_agg_results.set_index(dec_new_ind, inplace=True)

        binned_observ = dec_agg_results

        binned_observ = Posteriors.from_dataframe(binned_observ, enc_settings=enc_settings,
                                                  dec_settings=dec_settings,
                                                  user_key={'encode_settings': enc_settings,
                                                            'decode_settings': dec_settings,
                                                            'multi_index_keys': binned_observ.index.names})

        return binned_observ

    @staticmethod
    def _calc_observation_single_bin(spikes_in_parallel, elec_grp_list,
                                     prob_no_spike, time_bin_size, enc_settings, dtype=np.float32):

        global_prob_no_spike = np.prod(list(prob_no_spike.values()), axis=0)

        results = []

        dec_grp = Groupby(spikes_in_parallel.values, spikes_in_parallel['dec_bin'].values)
        pos_col_ind = spikes_in_parallel.columns.slice_locs(enc_settings.pos_col_names[0],
                                                            enc_settings.pos_col_names[-1])
        elec_grp_ind = spikes_in_parallel.columns.get_loc('elec_grp_id')
        num_missing_ind = spikes_in_parallel.columns.get_loc('num_missing_bins')
        dec_bin_start_ind = spikes_in_parallel.columns.get_loc('dec_bin_start')

        for dec_bin_ii, spikes_in_bin in dec_grp:
            obv_in_bin = np.ones(enc_settings.pos_num_bins)

            num_spikes = len(spikes_in_bin)

            elec_set = set()

            spike_bin_raw = spikes_in_bin

            # Contribution of each spike
            missing_bins_list = []
            dec_bin_timestamp = spike_bin_raw[0, dec_bin_start_ind]
            for obv, elec_grp_id, num_missing_bins in zip(spike_bin_raw[:, slice(*pos_col_ind)],
                                                          spike_bin_raw[:, elec_grp_ind],
                                                          spike_bin_raw[:, num_missing_ind]):

                elec_set.add(elec_grp_id)
                missing_bins_list.append(num_missing_bins)

                obv_in_bin = obv_in_bin * (obv + np.finfo(dtype).eps)
                obv_in_bin = obv_in_bin * prob_no_spike[elec_grp_id]
                obv_in_bin += np.finfo(dtype).eps

                #obv_in_bin = obv_in_bin / (np.nansum(obv_in_bin) * enc_settings.pos_bin_delta)

            # Contribution for electrodes that no spikes in this bin
            for elec_grp_id in elec_set.symmetric_difference(elec_grp_list):
                obv_in_bin = obv_in_bin * prob_no_spike[elec_grp_id]

            # Checking if missing bin has more than 1 whole number (and the rest are zeros)
            missing_bins_list = np.array(missing_bins_list)
            if np.count_nonzero(missing_bins_list) > 1:
                warnings.warn('For decode bin (' + dec_bin_ii + ') bin time (' + dec_bin_timestamp +
                              ') there are multiple possible values for missing bins ' + missing_bins_list +
                              ', which is not allowed.')

            results.append(np.concatenate([obv_in_bin, [dec_bin_timestamp, num_spikes, dec_bin_ii]]))
            for missing_ii in range(int(max(missing_bins_list))):
                results.append(np.concatenate([global_prob_no_spike, [dec_bin_timestamp+((missing_ii+1)*time_bin_size),
                                                                      0, dec_bin_ii+missing_ii+1]]))

        likelihoods = pd.DataFrame(np.vstack(results),
                                   columns=enc_settings.pos_col_names+['timestamp', 'num_spikes', 'dec_bin'])
        return likelihoods


class OfflinePPPosterior:

    def __init__(self, likelihoods: Posteriors, encode_settings: EncodeSettings,
                 decode_settings: DecodeSettings, time_bin_size=30,
                 trans_mat=None, prob_no_spike=None,
                 cuda=False, dtype=np.float32):

        if prob_no_spike:
            self.prob_no_spike = prob_no_spike
        else:
            self.prob_no_spike = {tet_id: np.ones(encode_settings.pos_num_bins) for tet_id in encode_settings.tetrodes}
        self.trans_mat = trans_mat
        self.likelihoods = likelihoods
        self.encode_settings = encode_settings
        self.decode_settings = decode_settings
        # self.which_trans_mat = which_trans_mat
        self.time_bin_size = time_bin_size

        self.cuda = cuda
        if self.cuda:
            self.device_name = 'cuda'
        else:
            self.device_name = 'cpu'
        self.dtype = torch.float
        self.device = torch.device(self.device_name)

        self.dtype = dtype

        self.posteriors = None

    def calc(self):
        print("Beginning posterior calculation")
        self.recalc_posterior()

        return self.posteriors

    def recalc_posterior(self):
        post = self.calc_posterior(self.likelihoods, self.trans_mat, self.encode_settings)
        self.posteriors = Posteriors.from_dataframe(post, enc_settings=self.encode_settings,
                                                    dec_settings=self.decode_settings,
                                                    user_key={'encode_settings': self.encode_settings,
                                                              'decode_settings': self.decode_settings,
                                                              'multi_index_keys': post.index.names})

    @staticmethod
    def calc_posterior(likelihoods, transition_mat, enc_settings: EncodeSettings):
        """

        Args:
            likelihoods (pd.DataFrame): The evaluated likelihood function per time bin, from calc_likelihood(...).
            transition_mat (np.array): The point process state transition matrix.
            enc_settings (EncodeSettings): Realtime encoding settings.

        Returns (pd.DataFrame): The decoded posteriors per time bin estimating the animal's location.

        """
        likelihoods_pos = likelihoods.loc[:, pos_col_format(0, enc_settings.pos_num_bins):
                                             pos_col_format(enc_settings.pos_num_bins-1,
                                                            enc_settings.pos_num_bins)]

        posteriors = OfflinePPPosterior._posterior_from_numpy(likelihoods_pos.values,
                                                              transition_mat, enc_settings.pos_num_bins,
                                                              enc_settings.pos_bin_delta)

        # posteriors = pp_cy.calc_posterior_cy(likelihoods_pos.values.copy(order='C'),
        #                                      transition_mat.copy(order='C'), enc_settings.pos_num_bins,
        #                                      enc_settings.pos_bin_delta)

        # copy observ DataFrame and replace with likelihoods, preserving other columns

        posteriors_df = pd.DataFrame(posteriors, index=likelihoods.index,
                                     columns=enc_settings.pos_col_names)

        posteriors_df['num_spikes'] = likelihoods['num_spikes']
        posteriors_df['dec_bin'] = likelihoods['dec_bin']

        return posteriors_df

    @staticmethod
    def _posterior_from_numpy(likelihoods, transition_mat, pos_num_bins, pos_delta):

        last_posterior = np.ones(pos_num_bins)

        posteriors = np.zeros(likelihoods.shape)

        for like_ii, like in enumerate(likelihoods):
            posteriors[like_ii, :] = like * np.matmul(transition_mat, np.nan_to_num(last_posterior))
            posteriors[like_ii, :] = posteriors[like_ii, :] / (np.nansum(posteriors[like_ii, :]) *
                                                               pos_delta)
            #last_posterior = posteriors[like_ii, :]

            # velocity mask - reset posterior after masked encoding bins
            if np.isnan(like).all():
                last_posterior = np.ones(pos_num_bins)

            else:
                last_posterior = posteriors[like_ii, :]

        # copy observ DataFrame and replace with likelihoods, preserving other columns

        return posteriors


class OfflinePPSinglePosterior:
    def __init__(self, likelihoods: Posteriors, encode_settings: EncodeSettings,
                 decode_settings: DecodeSettings,
                 trans_mat=None, prob_no_spike=None,
                 cuda=False, dtype=np.float32, **kwargs):

        if prob_no_spike:
            self.prob_no_spike = prob_no_spike
        else:
            self.prob_no_spike = {tet_id: np.ones(encode_settings.pos_num_bins) for tet_id in encode_settings.tetrodes}
        self.trans_mat = trans_mat
        self.likelihoods = likelihoods
        self.encode_settings = encode_settings
        self.decode_settings = decode_settings
        # self.which_trans_mat = which_trans_mat

        self.cuda = cuda
        if self.cuda:
            self.device_name = 'cuda'
        else:
            self.device_name = 'cpu'
        self.dtype = torch.float
        self.device = torch.device(self.device_name)

        self.dtype = dtype

        self.posteriors = self.likelihoods.copy()

        self.like_np = self.likelihoods.get_distribution_view().values
        self.post_np = np.zeros(self.like_np.shape)

        self.bin_num = 0
        self.prior = np.ones(encode_settings.pos_num_bins)

    def calc_cur_raw_post(self):
        self.post_np[self.bin_num, :] = (OfflinePPSinglePosterior.
                                         _posterior_single_step(self.like_np[self.bin_num, :],
                                                                self.prior, self.trans_mat))

    def get_cur_raw_post_integral(self):
        return np.sum(self.post_np[self.bin_num, :])

    def norm_cur_indicators(self, sum_all_indicators):
        self.post_np[self.bin_num, :] /= sum_all_indicators

    def next_bin(self):
        self.prior = self.post_np[self.bin_num, :]
        self.bin_num += 1

    def get_posterior(self):
        self.posteriors.set_posterior(self.post_np)
        return self.posteriors

    @staticmethod
    def _posterior_single_step(like, last_post, transition_mat):
        post = like * np.matmul(transition_mat, np.nan_to_num(last_post))
        return post


class OfflinePPIndicatorPosterior:

    def __init__(self, indicator_states, encode_settings: EncodeSettings,
                 decode_settings: DecodeSettings, cuda=False, dtype=np.float32):

        self.indicator_states = indicator_states
        self.encode_settings = encode_settings
        self.decode_settings = decode_settings

        self.bin_num = 0

        self.indicator_num_bins = {}
        for indic_key, indic_state in indicator_states.items():
            self.indicator_num_bins[indic_key] = indic_state['likelihoods']['dec_bin'].max()
        self.num_bins = int(min(self.indicator_num_bins.values()))

        self.cuda = cuda
        if self.cuda:
            self.device_name = 'cuda'
        else:
            self.device_name = 'cpu'
        self.dtype = torch.float
        self.device = torch.device(self.device_name)

        self.dtype = dtype

        self.indicator_posts = {}
        for indic_key, indic_kwds in indicator_states.items():
            self.indicator_posts[indic_key] = \
                OfflinePPSinglePosterior(**indic_kwds,
                                         encode_settings=self.encode_settings,
                                         decode_settings=self.decode_settings,
                                         cuda=self.cuda,
                                         dtype=self.dtype)

    def calc(self):
        self.bin_num = 0
        for bin_ii in range(self.num_bins):
            self.calc_single_step()

        for indic_key, indic_post in self.indicator_posts.items():
            indic_state = self.indicator_states.setdefault(indic_key, AttrDict({}))
            indic_state['posteriors'] = indic_post.posteriors

    def calc_single_step(self):
        indic_sum = 0
        for indic_key, indic_post in self.indicator_posts.items():
            indic_post.calc_cur_raw_post()
            indic_sum += indic_post.get_cur_raw_post_integral()

        for indic_key, indic_post in self.indicator_posts.items():
            indic_post.norm_cur_indicators(indic_sum)
            indic_post.next_bin()

        self.bin_num += 1


class OfflinePPDecoder(object):
    """
    Implementation of Adaptive Marked Point Process Decoder [Deng, et. al. 2015].
    
    Requires spike observation containers (spykshrk.franklab.pp_decoder.SpikeObservation).
    along with encoding (spykshrk.franklab.pp_decoder.EncodeSettings) 
    and decoding settings (spykshrk.franklab.pp_decoder.DecodeSettings).
    
    """
    def __init__(self, observ_obj: SpikeObservation, encode_settings: EncodeSettings,
                 decode_settings: DecodeSettings, time_bin_size=30, 
                 velocity_filter=None, trans_mat=None, prob_no_spike=None,
                 cuda=False, dtype=np.float32):
        """
        Constructor for OfflinePPDecoder.
        
        Args:
            observ_obj (SpikeObservation): Observered position distribution for each spike.
            encode_settings (EncodeSettings): Realtime encoder settings.
            decode_settings (DecodeSettings): Realtime decoder settings.
            
            time_bin_size (float, optional): Delta time per bin to run decode, defaults to decoder_settings value.
        """
        if prob_no_spike:
            self.prob_no_spike = prob_no_spike
        else:
            self.prob_no_spike = {tet_id: np.ones(encode_settings.pos_num_bins) for tet_id in encode_settings.tetrodes}
        self.trans_mat = trans_mat
        self.observ_obj = observ_obj
        self.encode_settings = encode_settings
        self.decode_settings = decode_settings
        # self.which_trans_mat = which_trans_mat
        self.time_bin_size = time_bin_size

        self.cuda = cuda
        if self.cuda:
            self.device_name = 'cuda'
        else:
            self.device_name = 'cpu'
        self.dtype = torch.float
        self.device = torch.device(self.device_name)

        self.dtype = dtype

        self.likelihoods = None
        self.posteriors = None
        self.posteriors_obj = None

    def __del__(self):
        print('decoder deleting')

    def run_decoder(self):
        """
        Run the decoder at a given time bin size.  Intermediate results are saved as
        attributes to the class.

        Returns (pd.DataFrame): Final decoded posteriors that estimate position.

        """

        print("Beginning likelihood calculation")
        self.recalc_likelihood()

        print("Beginning posterior calculation")
        self.recalc_posterior()

        return self.posteriors_obj

    def recalc_likelihood(self):
        self.likelihoods = self.calc_observation_intensity(self.observ_obj,
                                                           self.prob_no_spike,
                                                           self.encode_settings,
                                                           self.decode_settings,
                                                           time_bin_size=self.time_bin_size,
                                                           dtype=self.dtype)

    def recalc_posterior(self):
        self.posteriors = self.calc_posterior(self.likelihoods, self.trans_mat, self.encode_settings)
        self.posteriors_obj = Posteriors.from_dataframe(self.posteriors, enc_settings=self.encode_settings,
                                                        dec_settings=self.decode_settings,
                                                        user_key={'encode_settings': self.encode_settings,
                                                                  'decode_settings': self.decode_settings,
                                                                  'multi_index_keys': self.posteriors.index.names})

    @staticmethod
    def calc_observation_intensity(observ: SpikeObservation,
                                   prob_no_spike,
                                   enc_settings: EncodeSettings,
                                   dec_settings: DecodeSettings,
                                   time_bin_size=None,
                                   dtype=np.float32):
        """
        
        Args:
            observ (SpikeObservation): Object containing observation data frame, one row per spike observed.
            enc_settings (EncodeSettings): Encoder settings from realtime config.
            dec_settings (DecodeSettings): Decoder settings from realtime config.
            time_bin_size (float): Delta time per bin.

        Returns: (pd.DataFrame, dict[int, np.array]) DataFrame of observation per time bin in each row.
            Dictionary of numpy arrays, one per tetrode, containing occupancy firing rate.

        """

        day = observ.index.get_level_values('day')[0]
        epoch = observ.index.get_level_values('epoch')[0]

        if time_bin_size is not None:
            observ.update_observations_bins(time_bin_size=time_bin_size, inplace=True)
        else:
            time_bin_size = dec_settings.time_bin_size
            observ.update_observations_bins(time_bin_size=time_bin_size, inplace=True)

        observ.update_parallel_bins(60000, inplace=True)

        observ.update_num_missing_future_bins(inplace=True)

        #observ_dask = dd.from_pandas(observ.get_no_multi_index(), chunksize=30000)
        observ_grp = observ.groupby('parallel_bin')

        observ_meta = [(key, 'f8') for key in [pos_col_format(ii, enc_settings.pos_num_bins)
                                               for ii in range(enc_settings.pos_num_bins)]]
        observ_meta.append(('timestamp', 'f8'))
        observ_meta.append(('num_spikes', 'f8'))
        observ_meta.append(('dec_bin', 'f8'))

        elec_grp_list = observ['elec_grp_id'].unique()

        dec_agg_results = pd.DataFrame()
        for parallel_bin_ind, observ_par in observ_grp:

            bin_observed_par = OfflinePPDecoder._calc_observation_single_bin(observ_par,
                                                                             elec_grp_list=elec_grp_list,
                                                                             prob_no_spike=prob_no_spike,
                                                                             time_bin_size=time_bin_size,
                                                                             enc_settings=enc_settings,
                                                                             dtype=dtype),
            dec_agg_results = dec_agg_results.append(bin_observed_par[0])

        '''    
        observ_task = observ_grp.apply(functools.partial(OfflinePPDecoder._calc_observation_single_bin,
                                                         elec_grp_list=elec_grp_list,
                                                         prob_no_spike=prob_no_spike,
                                                         time_bin_size=time_bin_size,
                                                         enc_settings=enc_settings),
                                       meta=observ_meta)

        dec_agg_results = observ_task.compute()
        '''

        dec_agg_results.sort_values('timestamp', inplace=True)

        #encoding mask: convert timestamps to int in order to run get_irregular_resample
        dec_agg_results = dec_agg_results.astype({'timestamp': int})

        dec_new_ind = pd.MultiIndex.from_product([[day], [epoch], dec_agg_results['timestamp']])
        lev = list(dec_new_ind.levels)
        lab = list(dec_new_ind.codes)

        lev.append(dec_agg_results['timestamp']/float(enc_settings.sampling_rate))
        dec_agg_results.drop(columns='timestamp', inplace=True)

        lab.append(range(len(dec_agg_results)))

        dec_new_ind = pd.MultiIndex(levels=lev, codes=lab, names=['day', 'epoch', 'timestamp', 'time'])

        dec_agg_results.set_index(dec_new_ind, inplace=True)

        binned_observ = dec_agg_results

        #dec_agg_results['day'] = day
        #dec_agg_results['epoch'] = epoch
        #dec_agg_results['time'] = dec_agg_results['timestamp']/float(enc_settings.sampling_rate)
        #binned_observ = dec_agg_results.set_index(['day', 'epoch', 'timestamp', 'time'])

        # Smooth and normalize firing rate (conditional intensity function)

        binned_observ = Posteriors.from_dataframe(binned_observ, enc_settings=enc_settings,
                                                  dec_settings=dec_settings,
                                                  user_key={'encode_settings': enc_settings,
                                                            'decode_settings': dec_settings,
                                                            'multi_index_keys': binned_observ.index.names})

        return binned_observ

    @staticmethod
    def _calc_observation_single_bin(spikes_in_parallel, elec_grp_list,
                                     prob_no_spike, time_bin_size, enc_settings, dtype=np.float32):

        global_prob_no_spike = np.prod(list(prob_no_spike.values()), axis=0)

        results = []
        #parallel_id = spikes_in_parallel['parallel_bin'].iloc[0]
        #dec_grp = spikes_in_parallel.groupby('dec_bin')

        dec_grp = Groupby(spikes_in_parallel.values, spikes_in_parallel['dec_bin'].values)
        pos_col_ind = spikes_in_parallel.columns.slice_locs(enc_settings.pos_col_names[0],
                                                            enc_settings.pos_col_names[-1])
        elec_grp_ind = spikes_in_parallel.columns.get_loc('elec_grp_id')
        num_missing_ind = spikes_in_parallel.columns.get_loc('num_missing_bins')
        dec_bin_start_ind = spikes_in_parallel.columns.get_loc('dec_bin_start')

        for dec_bin_ii, spikes_in_bin in dec_grp:
            #print('parallel #{} dec #{}'.format(parallel_id, dec_bin_ii))
            obv_in_bin = np.ones(enc_settings.pos_num_bins)

            num_spikes = len(spikes_in_bin)

            elec_set = set()

            spike_bin_raw = spikes_in_bin

            """obv_in_bin = spike_bin_raw[:, slice(*pos_col_ind)].prod(axis=0)

            for elec_grp_id in spike_bin_raw[:, elec_grp_ind]:
                elec_set.add(elec_grp_id)
                obv_in_bin *= prob_no_spike[elec_grp_id]

            obv_in_bin = obv_in_bin / (np.sum(obv_in_bin) * enc_settings.pos_bin_delta)

            num_missing_bins = spike_bin_raw[-1,num_missing_ind]
            """
            # Contribution of each spike
            missing_bins_list = []
            dec_bin_timestamp = spike_bin_raw[0, dec_bin_start_ind]
            for obv, elec_grp_id, num_missing_bins in zip(spike_bin_raw[:, slice(*pos_col_ind)],
                                                          spike_bin_raw[:, elec_grp_ind],
                                                          spike_bin_raw[:, num_missing_ind]):

                elec_set.add(elec_grp_id)
                missing_bins_list.append(num_missing_bins)

                obv_in_bin = obv_in_bin * (obv + np.finfo(dtype).eps)
                obv_in_bin = obv_in_bin * prob_no_spike[elec_grp_id]
                obv_in_bin += np.finfo(dtype).eps

                #obv_in_bin = obv_in_bin / (np.nansum(obv_in_bin) * enc_settings.pos_bin_delta)

            # Contribution for electrodes that no spikes in this bin
            for elec_grp_id in elec_set.symmetric_difference(elec_grp_list):
                obv_in_bin = obv_in_bin * prob_no_spike[elec_grp_id]

            # Checking if missing bin has more than 1 whole number (and the rest are zeros)
            missing_bins_list = np.array(missing_bins_list)
            if np.count_nonzero(missing_bins_list) > 1:
                warnings.warn('For decode bin (' + dec_bin_ii + ') bin time (' + dec_bin_timestamp +
                              ') there are multiple possible values for missing bins ' + missing_bins_list +
                              ', which is not allowed.')

            results.append(np.concatenate([obv_in_bin, [dec_bin_timestamp, num_spikes, dec_bin_ii]]))
            for missing_ii in range(int(max(missing_bins_list))):
                results.append(np.concatenate([global_prob_no_spike, [dec_bin_timestamp+((missing_ii+1)*time_bin_size),
                                                                      0, dec_bin_ii+missing_ii+1]]))

        likelihoods = pd.DataFrame(np.vstack(results),
                                   columns=enc_settings.pos_col_names+['timestamp', 'num_spikes', 'dec_bin'])
        return likelihoods

    @staticmethod
    def calc_posterior(likelihoods, transition_mat, enc_settings: EncodeSettings):
        """
        
        Args:
            likelihoods (pd.DataFrame): The evaluated likelihood function per time bin, from calc_likelihood(...).
            transition_mat (np.array): The point process state transition matrix.
            enc_settings (EncodeSettings): Realtime encoding settings.

        Returns (pd.DataFrame): The decoded posteriors per time bin estimating the animal's location.

        """
        likelihoods_pos = likelihoods.loc[:, pos_col_format(0, enc_settings.pos_num_bins):
                                          pos_col_format(enc_settings.pos_num_bins-1,
                                                         enc_settings.pos_num_bins)]

        posteriors = OfflinePPDecoder._posterior_from_numpy(likelihoods_pos.values,
                                                            transition_mat, enc_settings.pos_num_bins,
                                                            enc_settings.pos_bin_delta)

        # posteriors = pp_cy.calc_posterior_cy(likelihoods_pos.values.copy(order='C'),
        #                                      transition_mat.copy(order='C'), enc_settings.pos_num_bins,
        #                                      enc_settings.pos_bin_delta)

        # copy observ DataFrame and replace with likelihoods, preserving other columns

        posteriors_df = pd.DataFrame(posteriors, index=likelihoods.index,
                                     columns=enc_settings.pos_col_names)

        posteriors_df['num_spikes'] = likelihoods['num_spikes']
        posteriors_df['dec_bin'] = likelihoods['dec_bin']

        return posteriors_df

    @staticmethod
    def _posterior_from_numpy(likelihoods, transition_mat, pos_num_bins, pos_delta):

        last_posterior = np.ones(pos_num_bins)

        posteriors = np.zeros(likelihoods.shape)

        for like_ii, like in enumerate(likelihoods):
            posteriors[like_ii, :] = like * np.matmul(transition_mat, np.nan_to_num(last_posterior))
            posteriors[like_ii, :] = posteriors[like_ii, :] / (np.nansum(posteriors[like_ii, :]) *
                                                               pos_delta)
            #last_posterior = posteriors[like_ii, :]

            # velocity mask - reset posterior after masked encoding bins 
            if np.isnan(like).all():
                last_posterior = np.ones(pos_num_bins)

            else:
                last_posterior = posteriors[like_ii, :]

        # copy observ DataFrame and replace with likelihoods, preserving other columns

        return posteriors
