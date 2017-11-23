import numpy as np
import pandas as pd
from itertools import product


class LinearPositionContainer:

    def __init__(self, nspike_pos_data):
        """
        Container for Linearized position read from an AnimalInfo.  
        Currently only supports nspike linearized position data.
        
        Args:
            nspike_pos_data: The position panda table from an animal info.  Expects a specific multi-index format.
        """
        self.pos_data = nspike_pos_data

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

        center_pos_flat = self.pos_data.query('@self.pos_data.seg_idx.seg_idx == 1').loc[:, ('lin_dist_well',
                                                                                             'well_center')]
        left_pos_flat = self.pos_data.query('@self.pos_data.seg_idx.seg_idx == 2 | '
                                            '@self.pos_data.seg_idx.seg_idx == 3').loc[:, ('lin_dist_well',
                                                                                           'well_left')]
        right_pos_flat = self.pos_data.query('@self.pos_data.seg_idx.seg_idx == 4 | '
                                             '@self.pos_data.seg_idx.seg_idx == 5').loc[:, ('lin_dist_well',
                                                                                            'well_right')]

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

    def get_observations_binned(self, dec_bin_size):
        dec_bins = np.floor((self.spike_dec['timestamp'] -
                             self.spike_dec['timestamp'][0])/dec_bin_size).astype('int')
        dec_bins_start = int(self.spike_dec['timestamp'][0] / dec_bin_size) * dec_bin_size + dec_bins * dec_bin_size
        self.spike_dec['dec_bin'] = dec_bins
        self.spike_dec['dec_bin_start'] = dec_bins_start