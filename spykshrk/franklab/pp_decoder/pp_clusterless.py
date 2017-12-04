import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from spykshrk.franklab.pp_decoder.util import gaussian, normal2D, apply_no_anim_boundary
from spykshrk.franklab.pp_decoder.data_containers import LinearPositionContainer, SpikeObservation, EncodeSettings, \
    DecodeSettings


class OfflinePPDecoder:
    """
    Implementation of Adaptive Marked Point Process Decoder [Deng, et. al. 2015].
    
    Requires linearized position (spykshrk.franklab.pp_decoder.LinearPositionContainer)
    and spike observation containers (spykshrk.franklab.pp_decoder.SpikeObservation).
    along with encoding (spykshrk.franklab.pp_decoder.EncodeSettings) 
    and decoding settings (spykshrk.franklab.pp_decoder.DecodeSettings).
    
    """
    def __init__(self, lin_obj: LinearPositionContainer, observ_obj: SpikeObservation, encode_settings: EncodeSettings,
                 decode_settings: DecodeSettings, which_trans_mat='learned', time_bin_size=None):
        """
        Constructor for OfflinePPDecoder.
        
        Args:
            lin_obj (LinearPositionContainer): Linear position of animal.
            observ_obj (SpikeObservation): Observered position distribution for each spike.
            encode_settings (EncodeSettings): Realtime encoder settings.
            decode_settings (DecodeSettings): Realtime decoder settings.
            which_trans_mat (str): Which point process transition matrix to use (learned, simple, uniform).
            time_bin_size (float, optional): Delta time per bin to run decode, defaults to decoder_settings value.
        """
        self.lin_obj = lin_obj
        self.observ_obj = observ_obj
        self.encode_settings = encode_settings
        self.decode_settings = decode_settings
        self.which_trans_mat = which_trans_mat
        self.time_bin_size = time_bin_size

        if self.which_trans_mat == 'learned':
            self.trans_mat = self.calc_learned_state_trans_mat(self.lin_obj.get_mapped_single_axis(),
                                                               self.encode_settings, self.decode_settings)
        elif self.which_trans_mat == 'simple':
            self.trans_mat = self.calc_simple_trans_mat(self.encode_settings)
        elif self.which_trans_mat == 'uniform':
            self.trans_mat = self.calc_uniform_trans_mat(self.encode_settings)

    def run_decoder(self, time_bin_size=None):
        """
        Run the decoder at a given time bin size.  Intermediate results are saved as
        attributes to the class.
        
        Args:
            time_bin_size (float, optional): Delta time per bin.

        Returns (pd.DataFrame): Final decoded posteriors that estimate position.

        """

        self.binned_observ, self.firing_rate = self.calc_observation_intensity(self.observ_obj,
                                                                               self.encode_settings,
                                                                               self.decode_settings,
                                                                               time_bin_size)

        self.occupancy = self.calc_occupancy(self.lin_obj, self.encode_settings)
        self.prob_no_spike = self.calc_prob_no_spike(self.firing_rate, self.occupancy, self.decode_settings)
        self.likelihoods = self.calc_likelihood(self.binned_observ, self.prob_no_spike, self.encode_settings)
        self.posteriors = self.calc_posterior(self.likelihoods, self.trans_mat, self.encode_settings)

        return self.posteriors

    @staticmethod
    def calc_learned_state_trans_mat(linpos_simple, enc_settings, dec_settings):
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

        linpos_state = linpos_simple
        linpos_ind = np.searchsorted(enc_settings.pos_bins, linpos_state, side='right') - 1

        # Create learned pos transition matrix
        learned_trans_mat = np.zeros([pos_num_bins, pos_num_bins])
        for first_pos_ind, second_pos_ind in zip(linpos_ind[:-1], linpos_ind[1:]):
            learned_trans_mat[first_pos_ind, second_pos_ind] += 1

        # normalize
        learned_trans_mat = learned_trans_mat / (learned_trans_mat.sum(axis=0)[None, :])
        learned_trans_mat[np.isnan(learned_trans_mat)] = 0

        # smooth
        learned_trans_mat = sp.signal.convolve2d(learned_trans_mat, kernel, mode='same')
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
        learned_trans_mat = learned_trans_mat / (learned_trans_mat.sum(axis=0)[None, :])
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

        # uniform offset
        uniform_gain = 0.01
        uniform_dist = np.ones(transition_mat.shape)

        # normalize transition matrix
        transition_mat = transition_mat/(transition_mat.sum(axis=0)[None, :])

        # normalize uniform offset
        uniform_dist = uniform_dist/(uniform_dist.sum(axis=0)[None, :])

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

        # normalize transition matrix
        transition_mat = transition_mat/( transition_mat.sum(axis=0)[None, :])

        return transition_mat

    @staticmethod
    def calc_observation_intensity(observ: SpikeObservation,
                                   enc_settings: EncodeSettings,
                                   dec_settings: DecodeSettings,
                                   time_bin_size=None):
        """
        
        Args:
            observ (SpikeObservation): Object containing observation data frame, one row per spike observed.
            enc_settings (EncodeSettings): Encoder settings from realtime config.
            dec_settings (DecodeSettings): Decoder settings from realtime config.
            time_bin_size (float): Delta time per bin.

        Returns: (pd.DataFrame, dict[int, np.array]) DataFrame of observation per time bin in each row.
            Dictionary of numpy arrays, one per tetrode, containing occupancy firing rate.

        """
        pos_num_bins = len(enc_settings.pos_bins)
        pos_bin_delta = enc_settings.pos_bins[1] - enc_settings.pos_bins[0]

        if time_bin_size is not None:
            spike_decode = observ.get_observations_bin_assigned(time_bin_size=time_bin_size)
        else:
            time_bin_size = dec_settings.time_bin_size
            spike_decode = observ.get_observations_bin_assigned(time_bin_size=time_bin_size)

        dec_est = np.zeros([int(spike_decode.index.get_level_values('dec_bin')[-1]) + 1, pos_num_bins])

        # initialize conditional intensity function
        firing_rate = {ntrode_id: np.zeros(pos_num_bins) for ntrode_id in spike_decode['ntrode_id'].unique()}

        groups = spike_decode.groupby('dec_bin')
        bin_num_spikes = [0] * len(dec_est)

        for bin_id, spikes_in_bin in groups:
            dec_in_bin = np.ones(pos_num_bins)

            bin_num_spikes[bin_id] = len(spikes_in_bin)

            # Count spikes for occupancy firing rate (conditional intensity function)
            for ntrode_id, pos in spikes_in_bin.loc[:, ('ntrode_id', 'position')].values:
                firing_rate[ntrode_id][np.searchsorted(enc_settings.pos_bins, pos, side='right') - 1] += 1

            for dec_ii, dec in enumerate(spikes_in_bin.loc[:, ['x{:0{dig}d}'.
                                                               format(bin_id, dig=len(str(enc_settings.pos_num_bins)))
                                                               for bin_id in range(enc_settings.pos_num_bins)]].values):
                dec_in_bin = dec_in_bin * dec
                dec_in_bin = dec_in_bin / (np.sum(dec_in_bin) * pos_bin_delta)

            dec_est[bin_id, :] = dec_in_bin

        # Smooth and normalize firing rate (conditional intensity function)
        for fr_key in firing_rate.keys():
            firing_rate[fr_key] = np.convolve(firing_rate[fr_key], enc_settings.pos_kernel, mode='same')

            firing_rate[fr_key] = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates,
                                                         firing_rate[fr_key])

            firing_rate[fr_key] = firing_rate[fr_key] / (firing_rate[fr_key].sum() * pos_bin_delta)

        start_bin_time = np.floor(spike_decode.index.get_level_values('timestamp')[0] / time_bin_size) * time_bin_size
        dec_bin_times = np.arange(start_bin_time, start_bin_time + time_bin_size * len(dec_est), time_bin_size)

        binned_observ = pd.DataFrame(data=dec_est, index=dec_bin_times,
                                     columns=['x{:0{dig}d}'.
                                              format(bin_id, dig=len(str(enc_settings.pos_num_bins)))
                                              for bin_id in range(enc_settings.pos_num_bins)])
        binned_observ['num_spikes'] = bin_num_spikes

        return binned_observ, firing_rate

    @staticmethod
    def calc_occupancy(lin_obj: LinearPositionContainer, enc_settings: EncodeSettings):
        """
        
        Args:
            lin_obj (LinearPositionContainer): Linear position of the animal.
            enc_settings (EncodeSettings): Realtime encoding settings.

        Returns (np.array): The occupancy of the animal

        """
        occupancy, occ_bin_edges = np.histogram(lin_obj.get_mapped_single_axis(), bins=enc_settings.pos_bin_edges,
                                                normed=True)

        occupancy = np.convolve(occupancy, enc_settings.pos_kernel, mode='same')

        return occupancy

    @staticmethod
    def calc_prob_no_spike(firing_rate, occupancy, dec_settings: DecodeSettings):
        """
        
        Args:
            firing_rate (pd.DataFrame): Occupancy firing rate, from calc_observation_intensity(...).
            occupancy (np.array): The occupancy of the animal.
            dec_settings (DecodeSettings): Realtime decoding settings.

        Returns (dict[int, np.array]): Dictionary of probability that no spike occured per tetrode.

        """
        prob_no_spike = {}
        for tet_id, tet_fr in firing_rate.items():
            prob_no_spike[tet_id] = np.exp(-dec_settings.time_bin_size/30000 * tet_fr / occupancy)

        return prob_no_spike

    @staticmethod
    def calc_likelihood(binned_observ: pd.DataFrame, prob_no_spike, enc_settings: EncodeSettings):
        """
        
        Args:
            binned_observ (pd.DataFrame): Observation distribution per time bin, from calc_observation_intensity(...).
            prob_no_spike (dict[int, np.array]): Dictionary of probability that no spike occured per tetrode, from
                calc_prob_no_spike(...).
            enc_settings (EncodeSettings): Realtime encoding settings.

        Returns (pd.DataFrame): The evaluated likelihood function per time bin.

        """
        num_spikes = binned_observ['num_spikes'].values
        dec_est = binned_observ.loc[:, ['x{:0{dig}d}'.format(bin_id, dig=len(str(enc_settings.pos_num_bins)))
                                        for bin_id in range(enc_settings.pos_num_bins)]].values
        likelihoods = np.ones(dec_est.shape)

        for num_spikes, (dec_ind, dec_est_bin) in zip(num_spikes, enumerate(dec_est)):
            if num_spikes > 0:
                likelihoods[dec_ind, :] = dec_est_bin

                for prob_no in prob_no_spike.values():
                    likelihoods[dec_ind, :] *= prob_no
            else:

                for prob_no in prob_no_spike.values():
                    likelihoods[dec_ind, :] *= prob_no

            # Normalize
            likelihoods[dec_ind, :] = likelihoods[dec_ind, :] / (likelihoods[dec_ind, :].sum() *
                                                                 enc_settings.pos_bin_delta)

        return pd.DataFrame(data=likelihoods, index=binned_observ.index,
                            columns=['x{:0{dig}d}'.format(bin_id, dig=len(str(enc_settings.pos_num_bins)))
                                     for bin_id in range(enc_settings.pos_num_bins)])

    @staticmethod
    def calc_posterior(likelihoods, transition_mat, enc_settings: EncodeSettings):
        """
        
        Args:
            likelihoods (pd.DataFrame): The evaluated likelihood function per time bin, from calc_likelihood(...).
            transition_mat (np.array): The point process state transition matrix.
            enc_settings (EncodeSettings): Realtime encoding settings.

        Returns (pd.DataFrame): The decoded posteriors per time bin estimating the animal's location.

        """
        last_posterior = np.ones(enc_settings.pos_num_bins)

        posteriors = np.zeros(likelihoods.shape)

        for like_ii, like in enumerate(likelihoods.values):
            posteriors[like_ii, :] = like * (transition_mat * last_posterior).sum(axis=1)
            posteriors[like_ii, :] = posteriors[like_ii, :] / (posteriors[like_ii, :].sum() *
                                                               enc_settings.pos_bin_delta)
            last_posterior = posteriors[like_ii, :]

        return pd.DataFrame(data=posteriors, index=likelihoods.index,
                            columns=['x{:0{dig}d}'.format(bin_id, dig=len(str(enc_settings.pos_num_bins)))
                                     for bin_id in range(enc_settings.pos_num_bins)])


class DecodeVisualizer:
    def plot_decode_2d(dec_bin_times, dec_est, stim_lockout_ranges, linpos_flat, plt_range, x_tick=1.0):
        stim_lockout_ranges_sec = stim_lockout_ranges/30000
        stim_lockout_range_sec_sub = stim_lockout_ranges_sec[(stim_lockout_ranges_sec[1] > plt_range[0]) &
                                                             (stim_lockout_ranges_sec[0] < plt_range[1])]

        plt.imshow(dec_est[(dec_bin_times > plt_range[0]*30000) & (dec_bin_times < plt_range[1]*30000)].transpose(),
                   extent=[plt_range[0], plt_range[1], 0, 450], origin='lower', aspect='auto', cmap='hot', zorder=0)

        plt.colorbar()

        # Plot linear position
        if isinstance(linpos_flat.index, pd.MultiIndex):
            linpos_index_s = linpos_flat.index.get_level_values('timestamp') / 30000
        else:
            linpos_index_s = linpos_flat.index / 30000
        index_mask = (linpos_index_s > plt_range[0]) & (linpos_index_s < plt_range[1])

        plt.plot(linpos_index_s[index_mask],
                 linpos_flat.values[index_mask], 'c.', zorder=1, markersize=5)

        plt.plot(stim_lockout_range_sec_sub.values.transpose(), np.tile([[440], [440]], [1, len(stim_lockout_range_sec_sub)]), 'c-*' )

        for stim_lockout in stim_lockout_range_sec_sub.values:
            plt.axvspan(stim_lockout[0], stim_lockout[1], facecolor='#AAAAAA', alpha=0.3)

        plt.plot(plt_range, [74, 74], '--', color='gray')
        plt.plot(plt_range, [148, 148], '--', color='gray')
        plt.plot(plt_range, [256, 256], '--', color='gray')
        plt.plot(plt_range, [298, 298], '--', color='gray')
        plt.plot(plt_range, [407, 407], '--', color='gray')

        plt.xticks(np.arange(plt_range[0], plt_range[1], x_tick))

