

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
        pos_data_simple.loc[:, 'lin_vel_center'] = self.pos_data.loc[:,('lin_vel', 'well_center')]
        pos_data_simple.loc[:, 'seg_idx'] = self.pos_data.loc[:,('seg_idx', 0)]
        pos_data_simple.loc[:,'timestamps'] = pos_data_time*30000
        pos_data_simple = pos_data_simple.set_index('timestamps')

        return pos_data_simple


    def get_rebinned(self, bin_size):

        pos_data_new_times = np.arange(self.pos_data.index[0] + (bin_size - self.pos_data.index[0] % bin_size),
                                       pos_data.index[-1]+1, bin_size)
        pos_data_bin_ids = np.arange(0, len(pos_data_new_times), 1)

        pos_data_binned = pos_data.reindex(pos_data_new_times, method='nearest')

        pos_data_binned['bin'] = pos_data_bin_ids

        return pos_data_binned
