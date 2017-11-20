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

        pos_data_simple = self.pos_data.loc[:,'lin_dist_well']
        pos_data_simple.loc[:, 'lin_vel_center'] = self.pos_data.loc[:, ('lin_vel', 'well_center')]
        pos_data_simple.loc[:, 'seg_idx'] = self.pos_data.loc[:, ('seg_idx', 0)]
        pos_data_simple.loc[:, 'timestamps'] = pos_data_time*30000
        pos_data_simple = pos_data_simple.set_index('timestamps')

        return pos_data_simple

    def get_rebinned(self, bin_size):

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

        stuff = grp.apply(epoch_rebin_func)
        return stuff
