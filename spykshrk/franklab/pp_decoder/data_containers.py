import numpy as np
import pandas as pd
import functools
from abc import ABC, ABCMeta, abstractclassmethod, abstractmethod
from itertools import product
import uuid

from spykshrk.franklab.pp_decoder.util import gaussian, AttrDict


class DataFormatError(RuntimeError):
    pass


def pos_col_format(ind, num_bins):
    return 'x{:0{dig}d}'.format(ind, dig=len(str(num_bins)))


class SeriesClass(pd.Series):

    @property
    def _constructor(self):
        return self.__class__

    @property
    def _constructor_expanddim(self):
        return DataFrameClass


class DataFrameClass(pd.DataFrame, metaclass=ABCMeta):

    _metadata = ['kwds', 'history']
    _internal_names = pd.DataFrame._internal_names + ['uuid']
    _internal_names_set = set(_internal_names)

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None, **kwds):
        """
        
        Args:
            data: 
            index: 
            columns: 
            dtype: 
            copy: 
            parent: Uses parent history if avaliable.
            history: List that is set as the history of this object.  
                Overrides both the history of the parent and data source.
            **kwds: 
        """
        # print('called for {} with shape {}'.format(type(data), data.shape))
        self.uuid = uuid.uuid4()
        self.kwds = kwds

        if history is not None:
            self.history = list(history)
        elif parent is not None:
            if hasattr(parent, 'history'):
                self.history = list(parent.history)
                self.history.append(parent)
            else:
                self.history = [parent]
        else:
            if hasattr(data, 'history'):
                self.history = list(data.history)
                self.history.append(data)
            else:
                self.history = [data]
        #self.history.append(self)

        super().__init__(data, index, columns, dtype, copy)

    def __setstate__(self, state):
        # Insert new uuid (internal value)
        self.uuid = uuid.uuid4()
        super().__setstate__(state)

    #def __setstate__(self, state):
    #    #self.__init__(data=state['_data'], history=state['history'], kwds=state['kwds'])
    #    self.__dict__.update(state.copy())

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     return state

    @property
    def _constructor(self):
        if hasattr(self, 'history'):
            return functools.partial(type(self), history=self.history, **self.kwds)
        else:
            return type(self)

    @property
    def _constructor_sliced(self):
        return SeriesClass

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError

    @classmethod
    @abstractclassmethod
    def create_default(cls, df, parent=None, **kwd):
        pass

    def __repr__(self):
        return "blar"
        #return '<{}: {}, shape: ({})>'.format(self.__class__.__name__, self.uuid, self.shape)


class DayEpochTimeSeries(DataFrameClass):

    def __init__(self, **kwds):
        data = kwds['data']
        index = kwds['index']

        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.MultiIndex):
                raise DataFormatError("DataFrame index must use MultiIndex as index.")

            if not all([col in data.index.names for col in ['day', 'epoch', 'timestamp', 'time']]):
                raise DataFormatError("DayEpochTimeSeries must have index with 4 levels named: "
                                      "day, epoch, timestamp, time.")

            epoch_grps = data.groupby(['day', 'epoch'])
            try:
                kwds['kwds']['start_times'] = []
                kwds['kwds']['start_timestamps'] = []
            except KeyError:
                kwds['kwds'] = {}
                kwds['kwds']['start_times'] = []
                kwds['kwds']['start_timestamps'] = []

            for key, epoch_data in epoch_grps:
                kwds['kwds']['start_times'].append(epoch_data.index.get_level_values('time')[0])
                kwds['kwds']['start_timestamps'].append(epoch_data.index.get_level_values('timestamp')[0])

        if index is not None and not isinstance(index, pd.MultiIndex):
            raise DataFormatError("Index to be set must be MultiIndex.")

        super().__init__(**kwds)

    def get_time(self):
        return self.index.get_level_values('time')

    def get_timestamp(self):
        return self.index.get_level_values('timestamp')

    def get_time_range(self):
        return [self.index.get_level_values('time')[0], self.index.get_level_values('time')[-1]]

    def get_relative_index(self, inplace=False):
        if inplace:
            ret_data = self
            ind = self.index
        else:
            ret_data = self.copy()
            ind = ret_data.index
        ind.set_levels(level='time', levels=(self.index.get_level_values('time') -
                                             self.index.get_level_values('time')[0]), inplace=True)
        ind.set_levels(level='timestamp', levels=(self.index.get_level_values('timestamp') -
                                                  self.index.get_level_values('timestamp')[0]), inplace=True)

        return ret_data


class DayEpochElecTimeChannelSeries(DayEpochTimeSeries):

    def __init__(self, **kwds):
        data = kwds['data']
        index = kwds['index']

        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.MultiIndex):
                raise DataFormatError("DataFrame index must use MultiIndex as index.")

            if not all([col in data.index.names for col in ['day', 'epoch', 'elec_grp',
                                                            'timestamp', 'time', 'channel']]):
                raise DataFormatError("DayEpochTimeSeries must have index with 6 levels named: "
                                      "day, epoch, elec_grp, timestamp, time.")

        if index is not None and not isinstance(index, pd.MultiIndex):
            raise DataFormatError("Index to be set must be MultiIndex.")

        super().__init__(**kwds)


class DayEpochElecTimeSeries(DayEpochTimeSeries):

    def __init__(self, **kwds):
        data = kwds['data']
        index = kwds['index']

        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.MultiIndex):
                raise DataFormatError("DataFrame index must use MultiIndex as index.")

            if not all([col in data.index.names for col in ['day', 'epoch', 'elec_grp', 'timestamp', 'time']]):
                raise DataFormatError("DayEpochTimeSeries must have index with 6 levels named: "
                                      "day, epoch, elec_grp, timestamp, time.")

        if index is not None and not isinstance(index, pd.MultiIndex):
            raise DataFormatError("Index to be set must be MultiIndex.")

        super().__init__(**kwds)


class EncodeSettings:
    """
    Mapping of encoding parameters from realtime configuration into class attributes for easy access.
    """
    def __init__(self, realtime_config):
        """
        
        Args:
            realtime_config (dict[str, *]): JSON realtime configuration imported as a dict
        """
        encoder_config = realtime_config['encoder']

        self.sampling_rate = encoder_config['sampling_rate']
        self.arm_coordinates = encoder_config['position']['arm_pos']

        self.pos_upper = encoder_config['position']['upper']
        self.pos_lower = encoder_config['position']['lower']
        self.pos_num_bins = encoder_config['position']['bins']
        self.pos_bin_delta = ((self.pos_upper - self.pos_lower) / self.pos_num_bins)

        self.pos_bins = np.linspace(0, self.pos_bin_delta * (self.pos_num_bins - 1), self.pos_num_bins)
        self.pos_bin_edges = np.linspace(0, self.pos_bin_delta * self.pos_num_bins, self.pos_num_bins+1)

        self.pos_kernel_std = encoder_config['position_kernel']['std']

        self.pos_kernel = gaussian(self.pos_bins,
                                   self.pos_bins[int(len(self.pos_bins)/2)],
                                   self.pos_kernel_std)

        self.mark_kernel_mean = encoder_config['mark_kernel']['mean']
        self.mark_kernel_std = encoder_config['mark_kernel']['std']


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


class SpikeWaves(DayEpochElecTimeChannelSeries):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None, **kwds):

        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy, parent=parent,
                         history=history, **kwds)

    @classmethod
    def create_default(cls, df, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls(df, parent=parent, **kwds)

    @classmethod
    def from_df(cls, df, enc_settings, parent=None, **kwds):
        df['time'] = df.index.get_level_values('timestamp') / float(enc_settings.sampling_rate)
        df.set_index('time', append=True, inplace=True)
        #df.index = df.index.swaplevel(4, 5)
        return cls(data=df, enc_settings=enc_settings, parent=parent, **kwds)


class SpikeFeatures(DayEpochElecTimeSeries):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None, **kwds):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy, parent=parent,
                         history=history, **kwds)

    @classmethod
    def create_default(cls, df, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls(df, parent=parent, **kwds)

    @classmethod
    def from_numpy_single_epoch_elec(cls, day, epoch, elec_grp, timestamp, amps, sampling_rate):
        ind = pd.MultiIndex.from_arrays([[day]*len(timestamp), [epoch]*len(timestamp), [elec_grp]*len(timestamp),
                                         timestamp, timestamp/float(sampling_rate)],
                                        names=['day', 'epoch', 'elec_grp', 'timestamp', 'time'])

        return cls(data=amps, index=ind)

    def get_above_threshold(self, threshold):
        ind = np.nonzero(np.any(self.values > threshold, axis=1))
        return self.iloc[ind]

    def get_simple_index(self):
        """
        Only use if MultiIndex has been selected for day, epoch, and tetrode.
        Returns:

        """

        return self.set_index(self.index.get_level_values('timestamp'))


class LinearPosition(DayEpochTimeSeries):
    """
    Container for Linearized position read from an AnimalInfo.  
    
    The linearized position can come from Frank Lab's Matlab data read by the NSpike data parser
    (spykshrk.realtime.simulator.nspike_data), using the AnimalInfo class to parse the directory
    structure and PosMatDataStream to parse the linearized position files.
    """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None, **kwds):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy, parent=parent,
                         history=history, **kwds)

    @classmethod
    def create_default(cls, df, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls.create_from_nspike_posmat(df, parent=parent, **kwds)

    @classmethod
    def from_nspike_posmat(cls, nspike_pos_data, enc_settings: EncodeSettings, parent=None):
        """
        
        Args:
            parent: 
            nspike_pos_data: The position panda table from an animal info.  Expects a specific multi-index format.
            enc_settings: Encoder settings, used to get the endpoints of the W track
        """
        if parent is None:
            parent = nspike_pos_data

        return cls(data=nspike_pos_data, parent=parent, enc_settings=enc_settings)

        # make sure there's a time field

    def get_pd_no_multiindex(self):
        """
        Removes the MultiIndexes, for a simplier panda table. Maintains well distances, 
        reduces velocity to just center velocity, and removes day and epoch info from index.
        
        This should not be used for multi-day datasets where the timestamp resets.
        
        Returns: Copy of linearized panda table with MultiIndexes removed

        """

        pos_data_time = self.index.get_level_values('time')
        pos_data_timestamp = self.index.get_level_values('timestamp')

        pos_data_simple = self.loc[:, 'lin_dist_well'].copy()
        pos_data_simple.loc[:, 'lin_vel_center'] = self.loc[:, ('lin_vel', 'well_center')]
        pos_data_simple.loc[:, 'seg_idx'] = self.loc[:, ('seg_idx', 'seg_idx')]
        pos_data_simple.loc[:, 'time'] = pos_data_time
        pos_data_simple.loc[:, 'timestamp'] = pos_data_timestamp
        pos_data_simple = pos_data_simple.set_index('timestamp')

        return pos_data_simple

    def get_resampled(self, bin_size):
        """
        
        Args:
            bin_size: size of time 
7
        Returns (LinearPositionContainer): copy of self with times resampled using backfill.

        """

        def epoch_rebin_func(df):
            day = df.index.get_level_values('day')[0]
            epoch = df.index.get_level_values('epoch')[0]

            new_timestamps = np.arange(df.index.get_level_values('timestamp')[0] +
                                       (bin_size - df.index.get_level_values('timestamp')[0] % bin_size),
                                       df.index.get_level_values('timestamp')[-1]+1, bin_size)
            new_times = new_timestamps / float(self.kwds['enc_settings'].sampling_rate)

            new_indices = pd.MultiIndex.from_tuples(list(zip(
                                                        new_timestamps, new_times)), names=['timestamp', 'time'])

            #df.set_index(df.index.get_level_values('timestamp'), inplace=True)
            pos_data_bin_ids = np.arange(0, len(new_timestamps), 1)

            pos_data_binned = df.loc[day, epoch].reindex(new_indices, method='ffill')
            #pos_data_binned.set_index(new_timestamps)
            pos_data_binned['bin'] = pos_data_bin_ids

            return pos_data_binned

        grp = self.groupby(level=['day', 'epoch'])

        pos_data_rebinned = grp.apply(epoch_rebin_func)
        return type(self)(pos_data_rebinned, history=self.history, **self.kwds)

    def get_irregular_resampled(self, timestamps):

        grp = self.groupby(level=['day', 'epoch'])
        for (day, epoch), grp_df in grp:
            ind = pd.MultiIndex.from_arrays([[day]*len(timestamps), [epoch]*len(timestamps),
                                             timestamps, np.array(timestamps)/
                                             float(self.kwds['enc_settings'].sampling_rate)],
                                            names=['day', 'epoch', 'timestamp', 'time'])

            return grp_df.reindex(ind, method='ffill', fill_value=0)

    def get_mapped_single_axis(self):
        """
        Returns linearized position converted into a segmented 1-D representation.
        
        Returns (pd.DataFrame): Segmented 1-D linear position.

        """

        invalid = self.query('@self.seg_idx.seg_idx == 0')
        invalid_flat = pd.DataFrame([[0, 0] for _ in range(len(invalid))],
                                    columns=['linpos_flat', 'linvel_flat'], index=invalid.index)

        center_flat = (self.query('@self.seg_idx.seg_idx == 1').
                       loc[:, [('lin_dist_well', 'well_center'),
                               ('lin_vel', 'well_center')]]) + self.kwds['enc_settings'].arm_coordinates[0][0]
        left_flat = (self.query('@self.seg_idx.seg_idx == 2 | '
                                '@self.seg_idx.seg_idx == 3').
                     loc[:, [('lin_dist_well', 'well_left'),
                             ('lin_vel', 'well_left')]]) + self.kwds['enc_settings'].arm_coordinates[1][0]
        right_flat = (self.query('@self.seg_idx.seg_idx == 4 | '
                                 '@self.seg_idx.seg_idx == 5').
                      loc[:, [('lin_dist_well', 'well_right'),
                              ('lin_vel', 'well_right')]]) + (self.kwds['enc_settings'].
                                                              arm_coordinates[2][0])
        center_flat.columns = ['linpos_flat', 'linvel_flat']
        left_flat.columns = ['linpos_flat', 'linvel_flat']
        right_flat.columns = ['linpos_flat', 'linvel_flat']

        linpos_flat = pd.concat([invalid_flat, center_flat, left_flat, right_flat]) # type: pd.DataFrame
        linpos_flat = linpos_flat.sort_index()

        linpos_flat['seg_idx'] = self.seg_idx.seg_idx


        # reset history to remove intermediate query steps
        return FlatLinearPosition.create_default(linpos_flat, parent=self)

    def get_time_only_index(self):
        return self.reset_index(level=['day', 'epoch'])


class SpikeObservation(DayEpochTimeSeries):
    """
    The observations can be generated by the realtime system or from an offline encoding model.
    The observations consist of a panda table with one row for each spike being decoded.  Each
    spike (and observation) is identified by a unique timestamp and electrode id. The observation
    content is the estimated probability that the spike and its marks will be observed in the
    encoding model.
    """
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None, **kwds):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy, parent=parent,
                         history=history, **kwds)

    @classmethod
    def create_default(cls, df, sampling_rate, parent=None, **kwds):
        if parent is None:
            parent = df

        enc_settings = AttrDict({'sampling_rate': sampling_rate})

        return cls.from_realtime(df, enc_setting=enc_settings, parent=parent, **kwds)

    @classmethod
    def from_realtime(cls, spike_dec, day, epoch, enc_settings, parent=None, **kwds):
        if parent is None:
            parent = spike_dec

        start_timestamp = spike_dec['timestamp'][0]
        spike_dec['time'] = spike_dec['timestamp'] / float(enc_settings.sampling_rate)
        return cls(spike_dec.set_index(pd.MultiIndex.from_arrays([[day]*len(spike_dec), [epoch]*len(spike_dec),
                                                                  spike_dec['timestamp'], spike_dec['time']],
                                                                 names=['day', 'epoch', 'timestamp', 'time'])),
                   parent=parent, **kwds)

    @classmethod
    def from_df(cls, df, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls(data=df, parent=parent, **kwds)

    def update_observations_bins(self, time_bin_size):
        dec_bins = np.floor((self.index.get_level_values('timestamp') -
                             self.index.get_level_values('timestamp')[0]) / time_bin_size).astype('int')
        dec_bins_start = (int(self.index.get_level_values('timestamp')[0] / time_bin_size) *
                          time_bin_size + dec_bins * time_bin_size)
        self['dec_bin'] = dec_bins
        self['dec_bin_start'] = dec_bins_start

        return self

    def update_parallel_bins(self, time_bin_size):
        parallel_bins = np.floor((self.index.get_level_values('timestamp') -
                                  self.index.get_level_values('timestamp')[0]) / time_bin_size).astype('int')
        self['parallel_bin'] = parallel_bins

        return self

    def get_no_multi_index(self):
        return pd.DataFrame(self.set_index(self.index.get_level_values('timestamp')))


class Posteriors(DayEpochTimeSeries):

    _metadata = DataFrameClass._metadata + ['enc_settings']

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None, **kwds):
        if 'enc_settings' in kwds:
            self.enc_settings = kwds['enc_settings']
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy, parent=parent,
                         history=history, **kwds)

    @classmethod
    def create_default(cls, df, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls.from_dataframe(df, parent=parent, **kwds)

    @classmethod
    def from_dataframe(cls, posterior: pd.DataFrame, index=None, columns=None, parent=None,
                       encode_settings=None, **kwds):
        if parent is None:
            parent = posterior

        if index is not None:
            posterior.set_index(index)
        if columns is not None:
            posterior.columns = columns
        return cls(data=posterior, parent=parent, enc_settings=encode_settings, **kwds)

    @classmethod
    def from_numpy(cls, posterior, day, epoch, timestamps, times, columns=None, parent=None, encode_settings=None):
        if parent is None:
            parent = posterior

        return cls(data=posterior, index=pd.MultiIndex.from_arrays([[day]*len(posterior), [epoch]*len(posterior),
                                                                    timestamps, times],
                                                                   names=['day', 'epoch', 'timestamp', 'time']),
                   columns=columns, parent=parent, enc_settings=encode_settings)

    @classmethod
    def from_realtime(cls, posterior: pd.DataFrame, day, epoch, columns=None, copy=False, parent=None,
                      enc_settings=None):
        if parent is None:
            parent = posterior

        if copy:
            posterior = posterior.copy()    # type: pd.DataFrame
        posterior.set_index(pd.MultiIndex.from_arrays([[day]*len(posterior), [epoch]*len(posterior),
                                                       posterior['timestamp'], posterior['timestamp']/
                                                       enc_settings.sampling_rate],
                                                      names=['day', 'epoch', 'timestamp', 'time']), inplace=True)

        if columns is not None:
            posterior.columns = columns

        return cls(data=posterior, parent=parent, enc_settings=enc_settings)

    def get_posteriors_as_np(self):
        return self[pos_col_format(0, self.kwds['enc_settings'].pos_num_bins):
                    pos_col_format(self.kwds['enc_settings'].pos_num_bins-1,
                                   self.kwds['enc_settings'].pos_num_bins)].values


    def get_distribution_view(self):
        return self.loc[:, pos_col_format(0, self.enc_settings.pos_num_bins):
                        pos_col_format(self.enc_settings.pos_num_bins-1, self.enc_settings.pos_num_bins)]


class StimLockout(DataFrameClass):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None, **kwds):
        super().__init__(data, index, columns, dtype, copy, parent, history, **kwds)

    @classmethod
    def create_default(cls, df, sampling_rate, parent=None, **kwds):
        if parent is None:
            parent = df

        enc_settings = AttrDict({'sampling_rate': sampling_rate})
        return cls.from_realtime(df, enc_settings=enc_settings, parent=parent, **kwds)

    @classmethod
    def from_realtime(cls, stim_lockout, enc_settings, parent=None, **kwds):
        """
        Class factory to create stimulation lockout from realtime system.  
        Reshapes the structure to a more useful format (stim lockout intervals)
        Args:
            parent: 
            stim_lockout: Stim lockout pandas table from realtime records

        Returns: StimLockout

        """
        if parent is None:
            parent = stim_lockout

        stim_lockout_ranges = stim_lockout.pivot(index='lockout_num', columns='lockout_state', values='timestamp')
        stim_lockout_ranges = stim_lockout_ranges.reindex(columns=[1, 0])
        stim_lockout_ranges.columns = pd.MultiIndex.from_product([['timestamp'], ['on', 'off']])
        stim_lockout_ranges_sec = stim_lockout_ranges / float(enc_settings.sampling_rate)
        stim_lockout_ranges_sec.columns = pd.MultiIndex.from_product([['time'], ['on', 'off']])
        df = pd.concat([stim_lockout_ranges, stim_lockout_ranges_sec], axis=1)      # type: pd.DataFrame

        return cls(df, parent=parent, enc_settings=enc_settings, **kwds)

    def get_range_sec(self, low, high):
        sel = self.query('@self.time.off > @low and @self.time.on < @high')
        return type(self)(sel)


class FlatLinearPosition(DayEpochTimeSeries):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None, **kwds):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy, parent=parent,
                         history=history, **kwds)

        if isinstance(data, pd.DataFrame) and 'linvel_flat' not in data.columns:
            raise DataFormatError("Missing 'linvel_flat' column.")

    @classmethod
    def create_default(cls, df, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls(df, parent=parent, **kwds)

    @classmethod
    def from_numpy_single_epoch(cls, day, epoch, timestamp, lin_pos, lin_vel, sampling_rate):
        time = timestamp/float(sampling_rate)
        return cls(pd.DataFrame(list(zip(lin_pos, lin_vel)), columns=['linpos_flat', 'linvel_flat'],
                                index=pd.MultiIndex.from_arrays([[day]*len(timestamp), [epoch]*len(timestamp),
                                                                 timestamp, time],
                                                                names=['day', 'epoch', 'timestamp', 'time'])))

    def get_above_velocity(self, threshold):

        # explicitly return copy convert weakref, for pickling
        return pd.DataFrame(self.query('linvel_flat > @threshold'))

    def get_mapped_single_axis(self):
        return self