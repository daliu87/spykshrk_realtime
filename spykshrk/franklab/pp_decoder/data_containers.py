import numpy as np
import pandas as pd
from itertools import product

from spykshrk.franklab.pp_decoder.util import gaussian


def pos_col_format(ind, num_bins):
    return 'x{:0{dig}d}'.format(ind, dig=len(str(num_bins)))


class PandaContainer(object):

    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    @property
    def loc(self):
        return self.data.loc

    @property
    def iloc(self):
        return self.data.iloc

    @property
    def xs(self):
        return self.data.xs

    def _repr_html_(self):
        return self.data._repr_html_()

    def _repr_latex_(self):
        return self.data._repr_latex_()

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()


class EncodeSettings:
    """
    Mapping of encoding parameters from realtime configuration into class attributes for easy access.
    """
    def __init__(self, realtime_config):
        """
        
        Args:
            realtime_config (dict[str, *]): JSON realtime configuration imported as a dict
        """

        self.arm_coordinates = realtime_config['encoder']['position']['arm_pos']

        self.pos_upper = realtime_config['encoder']['position']['upper']
        self.pos_lower = realtime_config['encoder']['position']['lower']
        self.pos_num_bins = realtime_config['encoder']['position']['bins']
        self.pos_bin_delta = ((self.pos_upper - self.pos_lower) / self.pos_num_bins)

        self.pos_bins = np.linspace(0, self.pos_bin_delta * (self.pos_num_bins - 1), self.pos_num_bins)
        self.pos_bin_edges = np.linspace(0, self.pos_bin_delta * self.pos_num_bins, self.pos_num_bins+1)

        self.pos_kernel_std = realtime_config['encoder']['position_kernel']['std']

        self.pos_kernel = gaussian(self.pos_bins,
                                   self.pos_bins[int(len(self.pos_bins)/2)],
                                   self.pos_kernel_std)


class DecodeSettings:
    """
    Mapping of decoding parameters from realtime configuration into class attributes for easy access.
    """
    def __init__(self, realtime_config):
        """
        
        Args:
            realtime_config (dict[str, *]): JSON realtime configuration imported as a dict
        """
        self.time_bin_size = realtime_config['pp_decoder']['bin_size']     # Decode bin size in samples (usually 30kHz)
        self.trans_smooth_std = realtime_config['pp_decoder']['trans_mat_smoother_std']
        self.trans_uniform_gain = realtime_config['pp_decoder']['trans_mat_uniform_gain']


class LinearPositionContainer:
    """
    Container for Linearized position read from an AnimalInfo.  
    
    The linearized position can come from Frank Lab's Matlab data read by the NSpike data parser
    (spykshrk.realtime.simulator.nspike_data), using the AnimalInfo class to parse the directory
    structure and PosMatDataStream to parse the linearized position files.
    """

    def __init__(self, nspike_pos_data, enc_settings: EncodeSettings):
        """
        
        Args:
            nspike_pos_data: The position panda table from an animal info.  Expects a specific multi-index format.
            enc_settings: Encoder settings, used to get the endpoints of the W track
        """
        self.data = nspike_pos_data
        self.enc_settings = enc_settings
        self.arm_coord = enc_settings.arm_coordinates

        # make sure there's a time field


    def get_pd_no_multiindex(self):
        """
        Removes the MultiIndexes, for a simplier panda table. Maintains well distances, 
        reduces velocity to just center velocity, and removes day and epoch info from index.
        
        This should not be used for multi-day datasets where the timestamp resets.
        
        Returns: Copy of linearized panda table with MultiIndexes removed

        """

        pos_data_time = self.data.loc[:, 'time']

        pos_data_simple = self.data.loc[:, 'lin_dist_well'].copy()
        pos_data_simple.loc[:, 'lin_vel_center'] = self.data.loc[:, ('lin_vel', 'well_center')]
        pos_data_simple.loc[:, 'seg_idx'] = self.data.loc[:, ('seg_idx', 'seg_idx')]
        pos_data_simple.loc[:, 'timestamps'] = pos_data_time*30000
        pos_data_simple = pos_data_simple.set_index('timestamps')

        return pos_data_simple

    def get_resampled(self, bin_size):
        """
        
        Args:
            bin_size: size of time 

        Returns (LinearPositionContainer): copy of self with times resampled using backfill.

        """

        def epoch_rebin_func(df):

            new_timestamps = np.arange(df.time.timestamp[0] +
                                       (bin_size - df.time.timestamp[0] % bin_size),
                                       df.time.timestamp[-1]+1, bin_size)

            df.set_index(df.index.get_level_values('timestamp'), inplace=True)
            pos_data_bin_ids = np.arange(0, len(new_timestamps), 1)

            pos_data_binned = df.reindex(new_timestamps, method='bfill')
            pos_data_binned.set_index(new_timestamps)
            pos_data_binned['bin'] = pos_data_bin_ids

            return pos_data_binned

        grp = self.data.groupby(level=['day', 'epoch'])

        pos_data_rebinned = grp.apply(epoch_rebin_func)
        return type(self)(pos_data_rebinned.copy(), self.enc_settings)

    def get_mapped_single_axis(self):
        """
        Returns linearized position converted into a segmented 1-D representation.
        
        Returns (pd.DataFrame): Segmented 1-D linear position.

        """

        center_pos_flat = (self.data.query('@self.data.seg_idx.seg_idx == 1').
                           loc[:, [('lin_dist_well', 'well_center')]]) + self.arm_coord[0][0]
        left_pos_flat = (self.data.query('@self.data.seg_idx.seg_idx == 2 | '
                                         '@self.data.seg_idx.seg_idx == 3').
                         loc[:, [('lin_dist_well', 'well_left')]]) + self.arm_coord[1][0]
        right_pos_flat = (self.data.query('@self.data.seg_idx.seg_idx == 4 | '
                                          '@self.data.seg_idx.seg_idx == 5').
                          loc[:, [('lin_dist_well', 'well_right')]]) + self.arm_coord[2][0]

        center_pos_flat.columns = ['linpos_flat']
        left_pos_flat.columns = ['linpos_flat']
        right_pos_flat.columns = ['linpos_flat']

        linpos_flat = pd.concat([center_pos_flat, left_pos_flat, right_pos_flat])
        linpos_flat = linpos_flat.sort_index()

        return linpos_flat


class SpikeObservation(PandaContainer):
    """
    The observations can be generated by the realtime system or from an offline encoding model.
    The observations consist of a panda table with one row for each spike being decoded.  Each
    spike (and observation) is identified by a unique timestamp and electrode id. The observation
    content is the estimated probability that the spike and its marks will be observed in the
    encoding model.
    """
    def __init__(self, spike_dec):
        self.data = spike_dec
        self.start_timestamp = self.data['timestamp'][0]
        self.data['time'] = self.data['timestamp'] / 30000.
        super().__init__(self.data.pivot_table(index=['timestamp', 'time']))

    def get_observations_bin_assigned(self, time_bin_size):
        dec_bins = np.floor((self.data.index.get_level_values('timestamp') -
                             self.data.index.get_level_values('timestamp')[0]) / time_bin_size).astype('int')
        dec_bins_start = (int(self.data.index.get_level_values('timestamp')[0] / time_bin_size) *
                          time_bin_size + dec_bins * time_bin_size)
        self.data['dec_bin'] = dec_bins
        self.data['dec_bin_start'] = dec_bins_start

        self.data.set_index([self.data.index.get_level_values('timestamp'), 'dec_bin'], inplace=True)

        return self


class Posteriors:
    def __init__(self, posts):
        self.data = posts


class StimLockout(PandaContainer):

    def __init__(self, data: pd.DataFrame):
        self.data_raw = data
        stim_lockout_ranges = data.pivot(index='lockout_num', columns='lockout_state', values='timestamp')
        stim_lockout_ranges = stim_lockout_ranges.reindex(columns=[1, 0])
        stim_lockout_ranges.columns = pd.MultiIndex.from_product([['timestamp'], ['on', 'off']])
        stim_lockout_ranges_sec = stim_lockout_ranges / 30000.
        stim_lockout_ranges_sec.columns = pd.MultiIndex.from_product([['time'], ['on', 'off']])
        df = pd.concat([stim_lockout_ranges, stim_lockout_ranges_sec], axis=1)      # type: pd.DataFrame

        super().__init__(df)

    def get_range_sec(self, low, high):
        return self.data[(self.data.time.off > low) & (self.data.time.on < high)]


