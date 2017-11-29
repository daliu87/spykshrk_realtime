import numpy as np
import pandas as pd
from itertools import product

from spykshrk.franklab.pp_decoder.util import gaussian


class EncodeSettings:
    def __init__(self, realtime_config):

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
    def __init__(self, realtime_config):
        self.time_bin_size = realtime_config['pp_decoder']['bin_size']     # Decode bin size in samples (usually 30kHz)
        self.trans_smooth_std = realtime_config['pp_decoder']['trans_mat_smoother_std']
        self.trans_uniform_gain = realtime_config['pp_decoder']['trans_mat_uniform_gain']


class LinearPositionContainer:

    def __init__(self, nspike_pos_data, enc_settings: EncodeSettings):
        """
        Container for Linearized position read from an AnimalInfo.  
        Currently only supports nspike linearized position data.
        
        Args:
            nspike_pos_data: The position panda table from an animal info.  Expects a specific multi-index format.
            arm_coord: nested list that contains the end point of the three segments
        """
        self.pos_data = nspike_pos_data
        self.arm_coord = enc_settings.arm_coordinates

    def get_pd_no_multiindex(self):

        pos_data_time = self.pos_data.loc[:, 'time']

        pos_data_simple = self.pos_data.loc[:, 'lin_dist_well'].copy()
        pos_data_simple.loc[:, 'lin_vel_center'] = self.pos_data.loc[:, ('lin_vel', 'well_center')]
        pos_data_simple.loc[:, 'seg_idx'] = self.pos_data.loc[:, ('seg_idx', 'seg_idx')]
        pos_data_simple.loc[:, 'timestamps'] = pos_data_time*30000
        pos_data_simple = pos_data_simple.set_index('timestamps')

        return pos_data_simple

    def get_resampled(self, bin_size):

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

        grp = self.pos_data.groupby(level=['day', 'epoch'])

        pos_data_rebinned = grp.apply(epoch_rebin_func)
        return pos_data_rebinned

    def get_mapped_single_axis(self):

        center_pos_flat = (self.pos_data.query('@self.pos_data.seg_idx.seg_idx == 1').
                           loc[:, ('lin_dist_well', 'well_center')]) + self.arm_coord[0][0]
        left_pos_flat = (self.pos_data.query('@self.pos_data.seg_idx.seg_idx == 2 | '
                                             '@self.pos_data.seg_idx.seg_idx == 3').
                         loc[:, ('lin_dist_well', 'well_left')]) + self.arm_coord[1][0]
        right_pos_flat = (self.pos_data.query('@self.pos_data.seg_idx.seg_idx == 4 | '
                                              '@self.pos_data.seg_idx.seg_idx == 5').
                          loc[:, ('lin_dist_well', 'well_right')]) + self.arm_coord[2][0]

        center_pos_flat.name = 'linpos_flat'
        left_pos_flat.name = 'linpos_flat'
        right_pos_flat.name = 'linpos_flat'

        linpos_flat = pd.concat([center_pos_flat, left_pos_flat, right_pos_flat])
        linpos_flat = linpos_flat.sort_index()

        return linpos_flat


class SpikeObservation:

    def __init__(self, spike_dec):
        self.spike_dec = spike_dec
        self.start_timestamp = self.spike_dec['timestamp'][0]

    def get_observations_bin_assigned(self, time_bin_size):
        dec_bins = np.floor((self.spike_dec['timestamp'] -
                             self.spike_dec['timestamp'][0]) / time_bin_size).astype('int')
        dec_bins_start = int(self.spike_dec['timestamp'][0] / time_bin_size) * time_bin_size + dec_bins * time_bin_size
        self.spike_dec['dec_bin'] = dec_bins
        self.spike_dec['dec_bin_start'] = dec_bins_start

        return self.spike_dec


