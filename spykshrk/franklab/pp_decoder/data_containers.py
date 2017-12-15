import numpy as np
import pandas as pd
import functools
from abc import ABC, ABCMeta, abstractclassmethod, abstractmethod
from itertools import product
import uuid

from spykshrk.franklab.pp_decoder.util import gaussian


class DataFormatError(RuntimeError):
    pass


def pos_col_format(ind, num_bins):
    return 'x{:0{dig}d}'.format(ind, dig=len(str(num_bins)))


class LocIndexerContainer(object):
    def __init__(self, container_cls, container, loc):
        self.container_cls = container_cls
        self.container = container
        self.loc = loc

    def __getitem__(self, item):
        return self.container_cls(data=self.loc.__getitem__(item), history=self.container.history)

    def __setitem__(self, key, value):
        return self.loc.__setitem__(key, value)


class PandaContainer(ABC):

    def __init__(self, dataframe: pd.DataFrame, history=None):
        self.data = dataframe

        if history is None:
            self.history = [self]
        else:
            self.history = history + [self]

        self.loc_cont = LocIndexerContainer(self.__class__, self, self.data.loc)
        self.iloc_cont = LocIndexerContainer(self.__class__, self, self.data.iloc)
        self.xs_cont = LocIndexerContainer(self.__class__, self, self.data.xs)

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    @property
    def loc(self):
        return self.loc_cont

    @property
    def iloc(self):
        return self.iloc_cont

    @property
    def xs(self):
        return self.xs_cont

    def _repr_html_(self):
        return self.data._repr_html_()

    def _repr_latex_(self):
        return self.data._repr_latex_()

    def __str__(self):
        return self.data.__str__()


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

        # print('called for {} with shape {}'.format(type(data), data.shape))
        self.uuid = uuid.uuid4()
        self.kwds = kwds
        if history is not None:
            self.history = list(history)
            self.history.append(self)
        if parent is not None:
            if hasattr(parent, 'history'):
                # Assumes history contains everything including self
                self.history = list(parent.history)
            else:
                self.history = [parent]
            self.history.append(self)

        super().__init__(data, index, columns, dtype, copy)

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
    def create_default(cls, df, **kwd):
        pass

    def __repr__(self):
        return '<{}: {}, shape: ({})>'.format(self.__class__.__name__, self.uuid, self.shape)


class DayEpochTimeSeries:

    def __init__(self, **kwds):
        data = kwds['data']
        index = kwds['index']

        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.MultiIndex):
                raise DataFormatError("DataFrame index must use MultiIndex as index.")

            if not (data.index.names == ['day', 'epoch', 'timestamp', 'time']):
                raise DataFormatError("DayEpochTimeSeries must have index with 4 levels named: "
                                      "day, epoch, timestamp, time.")

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


class LinearPosition(DayEpochTimeSeries, DataFrameClass):
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
    def create_default(cls, df, **kwds):
        return cls.create_from_nspike_posmat(df, **kwds)

    @classmethod
    def from_nspike_posmat(cls, nspike_pos_data, enc_settings: EncodeSettings):
        """
        
        Args:
            nspike_pos_data: The position panda table from an animal info.  Expects a specific multi-index format.
            enc_settings: Encoder settings, used to get the endpoints of the W track
        """

        return cls(data=nspike_pos_data, parent=nspike_pos_data, enc_settings=enc_settings)

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
            new_times = new_timestamps / 30000.

            new_indices = pd.MultiIndex.from_tuples(list(zip(
                                                        new_timestamps, new_times)), names=['timestamp', 'time'])

            #df.set_index(df.index.get_level_values('timestamp'), inplace=True)
            pos_data_bin_ids = np.arange(0, len(new_timestamps), 1)

            pos_data_binned = df.loc[day, epoch].reindex(new_indices, method='bfill')
            #pos_data_binned.set_index(new_timestamps)
            pos_data_binned['bin'] = pos_data_bin_ids

            return pos_data_binned

        grp = self.groupby(level=['day', 'epoch'])

        pos_data_rebinned = grp.apply(epoch_rebin_func)
        return type(self)(pos_data_rebinned, history=self.history, **self.kwds)

    def get_mapped_single_axis(self):
        """
        Returns linearized position converted into a segmented 1-D representation.
        
        Returns (pd.DataFrame): Segmented 1-D linear position.

        """

        center_pos_flat = (self.query('@self.seg_idx.seg_idx == 1').
                           loc[:, [('lin_dist_well', 'well_center')]]) + self.kwds['enc_settings'].arm_coordinates[0][0]
        left_pos_flat = (self.query('@self.seg_idx.seg_idx == 2 | '
                                    '@self.seg_idx.seg_idx == 3').
                         loc[:, [('lin_dist_well', 'well_left')]]) + self.kwds['enc_settings'].arm_coordinates[1][0]
        right_pos_flat = (self.query('@self.seg_idx.seg_idx == 4 | '
                                     '@self.seg_idx.seg_idx == 5').
                          loc[:, [('lin_dist_well', 'well_right')]]) + (self.kwds['enc_settings'].
                                                                        arm_coordinates[2][0])
        center_pos_flat.columns = ['linpos_flat']
        left_pos_flat.columns = ['linpos_flat']
        right_pos_flat.columns = ['linpos_flat']

        linpos_flat = pd.concat([center_pos_flat, left_pos_flat, right_pos_flat])   # type: pd.DataFrame
        linpos_flat = linpos_flat.sort_index()

        # reset history to remove intermediate query steps
        return type(self)(linpos_flat, history=self.history)

    def get_time_only_index(self):
        return self.reset_index(level=['day', 'epoch'])


class SpikeObservation(DayEpochTimeSeries, DataFrameClass):
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
    def create_default(cls, df, **kwds):
        return cls.from_realtime(df, **kwds)

    @classmethod
    def from_realtime(cls, spike_dec, day, epoch, **kwds):
        start_timestamp = spike_dec['timestamp'][0]
        spike_dec['time'] = spike_dec['timestamp'] / 30000.
        return cls(spike_dec.set_index(pd.MultiIndex.from_arrays([[day]*len(spike_dec), [epoch]*len(spike_dec),
                                                                  spike_dec['timestamp'], spike_dec['time']],
                                                                 names=['day', 'epoch', 'timestamp', 'time'])),
                   parent=spike_dec, **kwds)

    def update_observations_bins(self, time_bin_size):
        dec_bins = np.floor((self.index.get_level_values('timestamp') -
                             self.index.get_level_values('timestamp')[0]) / time_bin_size).astype('int')
        dec_bins_start = (int(self.index.get_level_values('timestamp')[0] / time_bin_size) *
                          time_bin_size + dec_bins * time_bin_size)
        self['dec_bin'] = dec_bins
        self['dec_bin_start'] = dec_bins_start

        return self


class Posteriors(DayEpochTimeSeries, DataFrameClass):

    _metadata = DataFrameClass._metadata + ['enc_settings']

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None, **kwds):
        if 'enc_settings' in kwds:
            self.enc_settings = kwds['enc_settings']
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy, parent=parent,
                         history=history, **kwds)

    @classmethod
    def create_default(cls, df, **kwds):
        return cls.from_dataframe(df, **kwds)

    @classmethod
    def from_dataframe(cls, posterior: pd.DataFrame, index=None, columns=None, encode_settings=None):
        if index is not None:
            posterior.set_index(index)
        if columns is not None:
            posterior.columns = columns
        return cls(data=posterior, enc_settings=encode_settings)

    @classmethod
    def from_numpy(cls, posterior, day, epoch, timestamps, times, columns=None, encode_settings=None):
        return cls(data=posterior, index=pd.MultiIndex.from_arrays([[day]*len(posterior), [epoch]*len(posterior),
                                                                    timestamps, times],
                                                                   names=['day', 'epoch', 'timestamp', 'time']),
                   columns=columns, enc_settings=encode_settings)

    @classmethod
    def from_realtime(cls, posterior: pd.DataFrame, day, epoch, columns=None, copy=False, encode_settings=None):
        if copy:
            posterior = posterior.copy()    # type: pd.DataFrame
        posterior.set_index(pd.MultiIndex.from_arrays([[day]*len(posterior), [epoch]*len(posterior),
                                                       posterior['timestamp'], posterior['timestamp']/30000],
                                                      names=['day', 'epoch', 'timestamp', 'time']), inplace=True)

        if columns is not None:
            posterior.columns = columns

        return cls(data=posterior, enc_settings=encode_settings)

    def get_posteriors_as_np(self):
        return self[pos_col_format(0, self.kwds['enc_settings'].pos_num_bins):
                    pos_col_format(self.kwds['enc_settings'].pos_num_bins-1,
                                   self.kwds['enc_settings'].pos_num_bins)].values

    def get_time(self):
        return self.index.get_level_values('time')

    def get_timestamp(self):
        return self.index.get_level_values('timestamp')

    def get_distribution_only(self):
        return self.loc[:, pos_col_format(0, self.enc_settings.pos_num_bins):
                        pos_col_format(self.enc_settings.pos_num_bins-1, self.enc_settings.pos_num_bins)]


class StimLockoutContainer(PandaContainer):

    def __init__(self, data_raw: pd.DataFrame=None, data: pd.DataFrame=None, history=None):

        self.data_raw = data_raw

        if data is None:
            stim_lockout_ranges = self.data_raw.pivot(index='lockout_num', columns='lockout_state', values='timestamp')
            stim_lockout_ranges = stim_lockout_ranges.reindex(columns=[1, 0])
            stim_lockout_ranges.columns = pd.MultiIndex.from_product([['timestamp'], ['on', 'off']])
            stim_lockout_ranges_sec = stim_lockout_ranges / 30000.
            stim_lockout_ranges_sec.columns = pd.MultiIndex.from_product([['time'], ['on', 'off']])
            df = pd.concat([stim_lockout_ranges, stim_lockout_ranges_sec], axis=1)      # type: pd.DataFrame

            super().__init__(df, history)

        else:
            super().__init__(data, history)

    def get_range_sec(self, low, high):
        return self.data[(self.data.time.off > low) & (self.data.time.on < high)]


class StimLockout(DataFrameClass):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None, **kwds):
        super().__init__(data, index, columns, dtype, copy, parent, history, **kwds)

    @classmethod
    def create_default(cls, df, **kwds):
        return cls.from_realtime(df, **kwds)

    @classmethod
    def from_realtime(cls, stim_lockout, **kwds):
        """
        Class factory to create stimulation lockout from realtime system.  
        Reshapes the structure to a more useful format (stim lockout intervals)
        Args:
            stim_lockout: Stim lockout pandas table from realtime records

        Returns: StimLockout

        """
        stim_lockout_ranges = stim_lockout.pivot(index='lockout_num', columns='lockout_state', values='timestamp')
        stim_lockout_ranges = stim_lockout_ranges.reindex(columns=[1, 0])
        stim_lockout_ranges.columns = pd.MultiIndex.from_product([['timestamp'], ['on', 'off']])
        stim_lockout_ranges_sec = stim_lockout_ranges / 30000.
        stim_lockout_ranges_sec.columns = pd.MultiIndex.from_product([['time'], ['on', 'off']])
        df = pd.concat([stim_lockout_ranges, stim_lockout_ranges_sec], axis=1)      # type: pd.DataFrame

        return cls(df, parent=stim_lockout, **kwds)

    def get_range_sec(self, low, high):
        return self.query('@self.time.off > @low and @self.time.on < @high')

