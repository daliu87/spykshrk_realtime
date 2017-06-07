"""
Defines classes necessary to access and stream NSpike animal's data.

"""
import numpy as np
import pandas as pd

from struct import unpack
from array import array

from scipy.io import loadmat

import os.path
from glob import glob
import math

from spykshrk.realtime.datatypes import LFPPoint, LinPosPoint, SpikePoint

try:
    from IPython.terminal.debugger import TerminalPdb
    bp = TerminalPdb(color_scheme='linux').set_trace
except AttributeError as err:
    print('Warning: Attribute Error ({}), disabling IPython TerminalPdb.'.format(err))
    bp = lambda: None

# setup pandas string/print format for debugging/interactive
pd.set_option('float_format', '{:,.1f}'.format)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 120)

# Constant variable definitions
TIMESTAMP_SCALE = 10000  # factor to convert timestamp to s


class DataReadError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ConfigurationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class AnimalInfo:
    """ animalinfo - stores details of the single animal to process and generates
    file lists that points to the data to process.
    
    """

    def __init__(self, base_dir, name, days, epochs, tetrodes, timescale=10000,
                 new_data=True):
        """ init function
        Args:
            base_dir: root directory of animal data
            name: name of animal
            days: array of which days of data to process
            tetrodes: array of which tetrodes to process
            epochs: list of epochs for encoding
        
        """
        self.base_dir = base_dir
        self.name = name
        self.days = days
        self.epochs = epochs
        self.tetrodes = tetrodes
        self.timescale = timescale
        self.times = self.parse_times()
        self.new_data = new_data

    def parse_times(self):
        """ parse_time - stores time data (times.mat) for each day to make sure
        each day is synchronized.  Should only be called from __init__

        Stores only the times for each epoch, 0 indexed.
        
        """
        times = {}
        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name,
                                    '%s%02d/times.mat' % (self.name, day))
            mat = loadmat(day_path)
            time_ranges = mat['ranges']
            times.setdefault(day, time_ranges[1::, :].astype('int32').tolist())
        return times

    def get_spike_paths(self):
        """ get_spike_paths - returns a list of paths that points to the spike
        data ([animalname][day]-[tetrode].mat) of each tetrode and day.

        This function uses the matlab data structure because the reference dataset
        being used is Frank, whose raw data was lost.
        
        """
        path_list = []
        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name,
                                    '%s%02d' % (self.name, day))
            for tet in self.tetrodes:
                tet_path_glob = os.path.join(day_path, '%02d*' % tet)
                tet_paths = glob(tet_path_glob)

                # only keep matching directories
                for tet_path in tet_paths:
                    if not os.path.isdir(tet_path):
                        del tet_paths[tet_paths.index(tet_path)]

                # Directory sanity checks
                if len(tet_paths) < 1:
                    print(('WARNING: %s day %02d does not have file for tetrode %02d,' +
                          ' skipping tetrode (%s)') %
                          (self.name, day, tet, tet_path_glob))
                    continue
                elif len(tet_paths) > 1:
                    print(('WARNING: %s day %02d has multiple directories for tetrode %02d,' +
                          'by default using first entry\n(%s)') %
                          (self.name, day, tet, tet_paths))

                spike_data_path = os.path.join(tet_paths[0], '%s%02d-%02d.mat'
                                               % (self.name, day, tet))

                path_list.append((day, tet, spike_data_path))

        return path_list

    def get_eeg_paths(self):
        """ returns a list of paths that points to the eeg data ([tetrode]-###.eeg) of each tetrode and day.

        This function uses the older \*.eeg because the reference dataset is frank.
        
        """
        path_list = []

        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name, '%s%02d'
                                    % (self.name, day))
            for tet in self.tetrodes:
                tet_path_glob = os.path.join(day_path, '%02d*.eeg' % tet)
                tet_paths = glob(tet_path_glob)
                if len(tet_paths) < 1:
                    print(('WARNING: %s day %02d does not have eeg file for tetrode %02d,' +
                           ' skipping tetrode (%s)')
                          % (self.name, day, tet, tet_path_glob))
                    continue
                elif len(tet_paths) > 1:
                    print(('WARNING: %s day %02d has multiple eeg files for tetrode %02d,' +
                           'by default using first entry\n(%s)')
                          % (self.name, day, tet, tet_paths))
                path_list.append((day, tet, tet_paths[0]))
        return path_list

    def get_pos_paths(self):
        """ get_pos_paths - returns a list of paths that points to the pos
        data (matclust_[animalname][day]-[tetrode].mat) of each tetrode and day.

        This function uses the already processed matclust data to extract position
        because the raw data of the reference dataset used (frank) was lost.
        
        """
        path_list = []
        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name, '%s%02d'
                                    % (self.name, day))
            for tet in self.tetrodes:
                tet_path_glob = os.path.join(day_path, '%02d*' % tet, '%s%02d-%02d_params.mat'
                                             % (self.name, day, tet))
                tet_paths = glob(tet_path_glob)

                # directory sanity check
                if len(tet_paths) < 1:
                    print(('WARNING: %s day %02d does not have file for tetrode %02d,' +
                           ' skipping tetrode (%s)')
                          % (self.name, day, tet, tet_path_glob))
                    continue
                elif len(tet_paths) > 1:
                    print(('WARNING: %s day %02d has multiple directories for tetrode %02d,' +
                           'by default using first entry\n(%s)')
                          % (self.name, day, tet, tet_paths))

                path_list.append((day, tet, tet_paths[0]))
        return path_list

    def get_posmat_paths(self):
        """ get_pathmat_paths - returns a list of paths that points to the post processed pos
        data ([post dir]/[3 char name prefix]pos[day].mat) of each day.

        This function uses the already processed video position data to extract position
        because the raw data of the reference dataset used (frank) was lost.
        
        """
        path_list = []
        for day in self.days:
            anim_prefix = self.name[0:3].title()
            day_glob = os.path.join(self.base_dir, anim_prefix.title()[0:3],
                                    '%slinpos%02d.mat' % (self.name[0:3].lower(), day))
            day_path = glob(day_glob)
            print(self.base_dir + anim_prefix.title()[0:3])
            print(day_path)
            # directory sanity check
            if len(day_path) < 1:
                print('WARNING: %s day %02d does not have file %slinpos%02d.mat'
                      % (self.name, day, self.name[0:3].lower(), day))
                continue
            elif len(day_path) > 1:
                print(('WARNING: %s day %02d has multiple directories %slinpos%02d.mat' +
                       'which should not happen...')
                      % (self.name, day, self.name[0:3].lower(), day))

            path_list.append((day, day_path[0]))
        return path_list

    def num_epochs(self, day):
        time_day = self.times[day]
        return np.size(time_day, 0)

    def calc_epoch_state(self, day, cur_time):
        """ Returns the encoded epoch that the time belongs to.  If the time
        is not in an epoch or if the epoch is not being encoded, return -1
        
        """
        time = self.times[day]
        for ii in self.epochs:
            if time[ii][0] <= cur_time <= time[ii][1]:
                return ii
        return -1

    def get_epoch_time_range(self, day, epoch):
        time = self.times[day]
        if epoch not in self.epochs:
            raise ConfigurationError('Epoch requested not an '
                                     'epoch being processed')

        return time[epoch][0], time[epoch][1]


class SpikeData:
    """	spike_data - Opens all files specified by animalinfo and creates a generator
    that can be called/polled to return spike data in bundles of the set timestep
    
    Args:
        anim (AnimalInfo): Defines the animal data and locations
        timestep: The time interval for the generator to return on each call
    """

    DAY_IND = 0
    TET_IND = 1
    DATA_IND = 2
    TIME_IND = 2
    WAVE_IND = 3

    def __init__(self, anim, timestep):
        # self.timestamps = []
        print('SpikeData: INIT start')
        self.data = pd.DataFrame()
        self.timestep = timestep
        self.anim = anim
        self.timebin_uniq = {}
        path_list = anim.get_spike_paths()
        self.days = set([])

        for path in path_list:
            day = path[self.DAY_IND]
            tet = path[self.TET_IND]
            mat = loadmat(path[self.DATA_IND])
            self.days.add(day)

            day_timestamps = mat['timestamps']
            day_waves = mat['waves']
            # Make sure not empty
            if day_timestamps.any() and day_waves.any():
                for epoch in self.anim.epochs:
                    epoch_start, epoch_end = self.anim.get_epoch_time_range(day, epoch)
                    epoch_mask = ((day_timestamps >= epoch_start) &
                                  (day_timestamps <= epoch_end)).ravel()
                    epoch_numspks = np.nonzero(epoch_mask)[0].size

                    epoch_spk_time = day_timestamps[epoch_mask]
                    epoch_spk_wave = day_waves[:, :, epoch_mask]
                    # convert nparray to a list so each waveform set can be
                    # added to the right index
                    epoch_spk_wave_list = epoch_spk_wave.swapaxes(0, 2).tolist()

                    spk_pd_idx = pd.MultiIndex.from_product(
                        ([day], [tet], [epoch], range(epoch_numspks)),
                        names=['day', 'tet', 'epoch', 'idx'])
                    spk_pd_col = pd.Series(['time', 'waveform', 'time_step_bin'])

                    spkdata_df = pd.DataFrame(index=spk_pd_idx, columns=spk_pd_col)
                    spkdata_df['time'] = epoch_spk_time
                    spkdata_df['waveform'] = epoch_spk_wave_list
                    timebin_assignment = (spkdata_df['time'] -
                                          epoch_start).floordiv(self.timestep)

                    spkdata_df['time_step_bin'] = timebin_assignment

                    self.data = self.data.append(spkdata_df)

                    # store the list of valid timebins for each day and epoch.  Timebins are
                    # determined across tetrodes
                    timebin_set = set(timebin_assignment[timebin_assignment >= 0].tolist())
                    timebin_global = self.timebin_uniq.setdefault((day, epoch), set())
                    self.timebin_uniq.update({(day, epoch): timebin_global.union(timebin_set)})

        print('SpikeData: INIT done')

    def __call__(self):
        for day in self.days:
            for epoch in self.anim.epochs:

                timebin_itr = sorted(list(self.timebin_uniq[(day, epoch)]))
                for timebin_num in timebin_itr:
                    data_rows = self.data[self.data['time_step_bin'] == timebin_num]

                    yield data_rows


class SpkDataStream:
    def __init__(self, anim, timestep):
        self.anim = anim
        self.block_data = SpikeData(anim, timestep)
        self.timestep = timestep

    def __iter__(self):
        return self

    def __call__(self):
        data_itr = self.block_data()

        for spk_block in data_itr:
            spk_sorted = spk_block.sort_values('time')
            for spk_row in spk_sorted.iterrows():
                tet_ind = spk_sorted.index.names.index('tet')
                tetnum = spk_row[0][tet_ind]
                spk_packet = SpikePoint(timestamp=spk_row[1]['time'],
                                        ntrode_index=self.anim.tetrodes.index(tetnum),
                                        data=spk_row[1]['waveform'])
                yield spk_packet


class EEGDataStream:
    """	Streams NSpike EEG Data specified by AnimalInfo.
    
    Opens all files specified by anim (AnimalInfo) and creates a generator
    that can be called/polled to return eeg data points.  EEG points are returned
    one at a time for each tetrode and in chronological order.
    
    Args:
        anim (AnimalInfo): Defines the animal data and locations
    """
    DAY_IND = 0
    TET_IND = 1
    TIME_IND = 2
    DATA_IND = 2
    EEG_REC_TIME_IND = 0
    EEG_REC_NUMSAMP_IND = 1
    EEG_REC_SAMPRATE_IND = 2
    EEG_REC_DATA_IND = 3

    def __init__(self, anim):

        self.data = {}
        self.anim = anim
        path_list = anim.get_eeg_paths()
        days = list(set(map(lambda x: x[self.DAY_IND], path_list)))
        open_files = {day: [] for day in days}
        # Dict of DataFrames for each day
        self.data = {}
        for path in path_list:
            day = path[self.DAY_IND]
            tet = path[self.TET_IND]
            # record reader iterator
            get_eeg_rec = self.read_eeg_rec(path[self.DATA_IND])

            try:
                rec = get_eeg_rec.__next__()
                cur_epoch = self.anim.calc_epoch_state(path[0], rec[0])
                # Assume first record's sample frequency is correct
                # and all files have the same sample frequency
                self.sampfreq = rec[2]

                self.dt = 1/self.sampfreq*anim.timescale

                # empty dataframe per day so each tetrode's dataframe
                # can be joined
                day_data = self.data.setdefault(day, pd.DataFrame())
                print('preallocating')

                tet_eeg_data_list = []
                tet_eeg_time_list = []
                tet_eeg_rowcount = 0
                # keep reading records until iterator ends
                while True:
                    rec_start_time = rec[0]
                    cur_epoch = self.anim.calc_epoch_state(path[0], rec_start_time)
                    # if rec in valid epoch range
                    if cur_epoch != -1:
                        rec_num_samp = rec[1]
                        rec_data = rec[3]
                        # interpolate start time to the entire record
                        rec_times = np.arange(0,rec_num_samp)*self.dt+rec_start_time
                        tet_eeg_data_list.extend(rec_data)
                        tet_eeg_time_list.extend(rec_times)

                        tet_eeg_rowcount += rec_num_samp

                    rec = get_eeg_rec.__next__()

            except StopIteration as ex:
                pass

            # processing file done
            day_data = day_data.merge(pd.DataFrame(index=tet_eeg_time_list,
                                                   data=tet_eeg_data_list, columns=[tet]),
                                      how='outer', left_index=True, right_index=True, copy=False)
            # re add merged data to dict
            self.data[day] = day_data

    def __call__(self):
        # extract number of days into a unique day list
        day_list = self.data.keys()
        days = list(set(day_list))
        for day in days:

            day_data = self.data[day]   # type: pd.DataFrame

            for epoch in self.anim.epochs:

                epoch_start_time = self.anim.times[day][epoch][0]
                epoch_end_time = self.anim.times[day][epoch][1]
                # Initialize starting row to the start of the epoch
                row_cursor = np.abs(day_data.index - epoch_start_time).argmin()

                timestamp = day_data.index[row_cursor]
                while timestamp < epoch_end_time:
                    # Gets the voltages values of all tetrodes at specific timepoint (cursor)
                    # bypassing pandas by using values to avoid creating Series
                    raw_timepoint = day_data.values[row_cursor]

                    for col_ind, tet_val in enumerate(raw_timepoint):
                        # test to make sure value is not NaN, this is a computationally efficient shortcut way.
                        if tet_val == tet_val:
                            tet_id = day_data.columns[col_ind]
                            yield LFPPoint(timestamp=int(timestamp * 3), ntrode_index=int(col_ind),
                                           ntrode_id=int(tet_id), data=int(tet_val))

                    # Increment to next row
                    row_cursor += 1
                    # retrieve new row's timestamp
                    timestamp = day_data.index[row_cursor]

    def stream_eeg_rec(self, rec):
        """ A generator function that takes a single record and returns
        each data point in the record with an interpolated timestamp
        based on the sample frequency.  The record must conform with \*.eeg
        format (timestamp, numsamples, sample rate, [data array])
        Args:
            rec: eeg record (timestamp, numsamples, sample rate, [data array])
        
        Returns:
            interp_time: interpolated timestamp of data point being returned
            eeg_pt: eeg sample data
        """
        rec_data = rec[self.EEG_REC_DATA_IND]
        rec_timestamp = rec[self.EEG_REC_TIME_IND]
        rec_numsamp = rec[self.EEG_REC_NUMSAMP_IND]
        rec_sampfreq = rec[self.EEG_REC_SAMPRATE_IND]

        samp_count = 0

        for point in rec_data:
            point_timestamp = rec_timestamp + 1 / float(rec_sampfreq) * samp_count * 10000
            yield point_timestamp, point
            samp_count += 1
            if samp_count > rec_numsamp:
                raise DataReadError('Record being streamed has more samples than \
                        set in the number of samples field.')
        if samp_count < rec_numsamp:
            raise DataReadError('Record being streamed has fewer samples than \
                    set in the number of samples field.')

    @staticmethod
    def read_eeg_header(path):
        f = open(path, 'r')
        header_text = []
        try:
            # reading header, limit to looking at top few lines
            # before throwing exception
            for linenum in range(0, 10):
                line = f.readline()

                header_text.append(line)
                if line == '%%ENDHEADER\n':
                    break

        except EOFError as e:
            print('EOFError reading EEG file (%s) - (%s, %s)' % (f.name, repr(e), e.args))
            f.close()
            return

        return header_text

    @staticmethod
    def read_eeg_rec(path):
        """ A generator function that for the file string specified in path,
        will return single eeg record (\*.eeg format) in sequential order. The
        generator returns (StopIterator) at the EOF.
        
        Args:
            path: eeg file path string
        
        Returns:
            timestamp: timestamp for begining of record
            numsamples: number of samples in record
            sampfreq: frequency of sample
            data: raw data (short) of eeg record
        """
        with open(path, 'rb') as f:
            error_code = 0
            try:
                # Skip header
                while f.readline() != b'%%ENDHEADER\n':
                    pass

                while True:
                    timestamp_bytes = f.read(4)
                    if not timestamp_bytes:
                        f.close()
                        return
                    timestamp = unpack('I', timestamp_bytes)[0]
                    error_code = 1
                    numsamples_bytes = f.read(4)
                    if not numsamples_bytes:
                        f.close()
                        return
                    numsamples = unpack('i', numsamples_bytes)[0]
                    error_code = 2
                    sampfreq_bytes = f.read(8)
                    if not sampfreq_bytes:
                        f.close()
                        return
                    sampfreq = unpack('d', sampfreq_bytes)[0]

                    error_code = 3
                    data = array('h')
                    data.fromfile(f, numsamples)

                    yield (timestamp, numsamples, sampfreq, data)

            except EOFError as e:
                print('EOFError reading EEG file (%s) - (%s, %s): code %d' % (f.name, repr(e), e.args, error_code))
                f.close()
                return

class EEGDataTimeBlock:
    """ Returns blocks of EEG Data specified by AnimalInfo.
    
    Opens all files specified by animalinfo and creates a generator
    that can be called/polled to return eeg data in bundles of the set timestep.
    Each returned set is just the closest number of points within that time, the
    actual size is not guarenteed and it is not aligned to anything in particular.
    
    Args:
        anim (AnimalInfo): Defines animal data and location
        timestep: The time interval for the generator to return on each call

    """
    DAY_IND = 0
    TET_IND = 1
    TIME_IND = 2
    DATA_IND = 2
    EEG_REC_TIME_IND = 0
    EEG_REC_NUMSAMP_IND = 1
    EEG_REC_SAMPRATE_IND = 2
    EEG_REC_DATA_IND = 3

    def __init__(self, anim, timestep):

        print('EEGData: INIT start')
        self.data = {}
        self.timestep = timestep
        self.anim = anim
        path_list = anim.get_eeg_paths()
        days = list(set(map(lambda x: x[self.DAY_IND], path_list)))
        open_files = {day: [] for day in days}
        # Dict of DataFrames for each day
        self.data = {}
        for path in path_list:
            day = path[self.DAY_IND]
            tet = path[self.TET_IND]
            # record reader iterator
            get_eeg_rec = self.read_eeg_rec(path[self.DATA_IND])

            try:
                rec = get_eeg_rec.__next__()
                cur_epoch = self.anim.calc_epoch_state(path[0], rec[0])
                # Assume first record's sample frequency is correct
                # and all files have the same sample frequency
                self.sampfreq = rec[2]

                self.dt = 1/self.sampfreq*anim.timescale

                # Get or prealloc DataFrame for the day.
                # Each row a timestamp, each column one tetrode.
                # Prealloc timestamps based on times.mat and sampfreq
                #day_data = self.data.setdefault( day,
                #		pd.DataFrame(index = np.arange(num_samp_day),
                #		columns = anim.tetrode_list))

                # empty dataframe per day so each tetrode's dataframe
                # can be joined
                day_data = self.data.setdefault(day, pd.DataFrame())
                print('preallocating')

                # prealloc dataframe for this tetrode/file
                #tet_eeg = pd.DataFrame(index = np.arange(math.ceil(num_samp_day)),columns=['time', tet])
                tet_eeg_data_list = []
                tet_eeg_time_list = []
                tet_eeg_rowcount = 0
                # keep reading records until iterator ends
                while True:
                    rec_start_time = rec[0]
                    cur_epoch = self.anim.calc_epoch_state(path[0], rec_start_time)
                    # if rec in valid epoch range
                    if cur_epoch != -1:
                        rec_num_samp = rec[1]
                        rec_data = rec[3]
                        # interpolate start time to the entire record
                        rec_times = np.arange(0,rec_num_samp)*self.dt+rec_start_time
                        tet_eeg_data_list.extend(rec_data)
                        tet_eeg_time_list.extend(rec_times)
                        # Take slice of table for current record and set it
                        #tet_eeg.loc[tet_eeg_rowcount:
                        #		tet_eeg_rowcount+rec_num_samp-1] = zip(rec_times,rec_data)
                        #temp = tet_eeg.loc[tet_eeg_rowcount: tet_eeg_rowcount+rec_num_samp-1]
                        #temp = zip(rec_times,rec_data)
                        tet_eeg_rowcount += rec_num_samp

                    rec = get_eeg_rec.__next__()

            except StopIteration as ex:
                pass

            # processing file done
            print('merging')
            day_data = day_data.merge(pd.DataFrame(index=tet_eeg_time_list,
                    data=tet_eeg_data_list, columns=[tet]),
                    how='outer', left_index=True, right_index=True, copy=False)
            # re add merged data to dict
            self.data[day] = day_data

            #day_data = self.data.setdefault(path[self.DAY_IND], {})
            #tet_data = day_data.setdefault(path[self.TET_IND], {})
            #tet_data.update(eeg)
        # self.data.append((path[self.DAY_IND],path[self.TET_IND]) + eeg_epoch)
        print('EEGData: INIT done')

    def __call__(self):
        # extract number of days into a unique day list
        day_list = self.data.keys()
        days = list(set(day_list))

        for day in days:

            day_data = self.data[day]

            for epoch in self.anim.epochs:

                epoch_end_time = self.anim.times[day][epoch][1]
                dt_idx = math.ceil(self.timestep / self.dt)
                # set time to smallest
                # index cursor for start of each timestep.  More efficient
                # than timestamp search?
                df_cursor = 0
                last_epoch_data = False
                while not last_epoch_data:

                    df_cursor_next = df_cursor + dt_idx

                    slice_data = day_data.iloc[df_cursor:df_cursor_next]
                    if df_cursor_next >= day_data.shape[0]:
                        last_epoch_data = True
                    elif day_data.iloc[df_cursor_next].name > epoch_end_time:
                        # only return slice within epoch
                        slice_data = slice_data.iloc[slice_data.index < epoch_end_time]
                        last_epoch_data = True

                    df_cursor = df_cursor_next
                    # for now yield raw data frame slice, no reason to
                    # package it seperately
                    yield slice_data

    def stream_eeg_rec(self, rec):
        """ A generator function that takes a single record and returns
        each data point in the record with an interpolated timestamp
        based on the sample frequency.  The record must conform with \*.eeg
        format (timestamp, numsamples, sample rate, [data array])
        
        Args:
        rec: eeg record (timestamp, numsamples, sample rate, [data array])
        
        Returns:
        interp_time: interpolated timestamp of data point being returned
        eeg_pt: eeg sample data
        """
        rec_data = rec[self.EEG_REC_DATA_IND]
        rec_timestamp = rec[self.EEG_REC_TIME_IND]
        rec_numsamp = rec[self.EEG_REC_NUMSAMP_IND]
        rec_sampfreq = rec[self.EEG_REC_SAMPRATE_IND]

        samp_count = 0

        for point in rec_data:
            point_timestamp = rec_timestamp + 1 / float(rec_sampfreq) * samp_count * 10000
            yield point_timestamp, point
            samp_count += 1
            if samp_count > rec_numsamp:
                raise DataReadError('Record being streamed has more samples than \
                        set in the number of samples field.')
        if samp_count < rec_numsamp:
            raise DataReadError('Record being streamed has fewer samples than \
                    set in the number of samples field.')

    @staticmethod
    def read_eeg_header(path):
        f = open(path, 'r')
        header_text = []
        try:
            # reading header, limit to looking at top few lines
            # before throwing exception
            for linenum in range(0, 10):
                line = f.readline()

                header_text.append(line)
                if line == '%%ENDHEADER\n':
                    break

        except EOFError as e:
            print('EOFError reading EEG file (%s) - (%s, %s)' % (f.name, repr(e), e.args))
            f.close()
            return

        return header_text

    @staticmethod
    def read_eeg_rec(path):
        """ A generator function that for the file string specified in path,
        will return single eeg record (\*.eeg format) in sequential order. The
        generator returns (StopIterator) at the EOF.
        Args:
            path: eeg file path string
        Returns:
            timestamp: timestamp for begining of record
            numsamples: number of samples in record
            sampfreq: frequency of sample
            data: raw data (short) of eeg record
        """
        with open(path, 'rb') as f:
            error_code = 0
            try:
                # Skip header
                while f.readline() != b'%%ENDHEADER\n':
                    pass

                while True:
                    timestamp_bytes = f.read(4)
                    if not timestamp_bytes:
                        f.close()
                        return
                    timestamp = unpack('I', timestamp_bytes)[0]
                    error_code = 1
                    numsamples_bytes = f.read(4)
                    if not numsamples_bytes:
                        f.close()
                        return
                    numsamples = unpack('i', numsamples_bytes)[0]
                    error_code = 2
                    sampfreq_bytes = f.read(8)
                    if not sampfreq_bytes:
                        f.close()
                        return
                    sampfreq = unpack('d', sampfreq_bytes)[0]

                    error_code = 3
                    data = array('h')
                    data.fromfile(f, numsamples)

                    yield (timestamp, numsamples, sampfreq, data)

            except EOFError as e:
                print('EOFError reading EEG file (%s) - (%s, %s): code %d' % (f.name, repr(e), e.args, error_code))
                f.close()
                return


class EEGDataTimeBlockStream:
    def __init__(self, anim, timestep):
        self.anim = anim
        self.block_data = EEGDataTimeBlock(anim, timestep)
        self.timestep = timestep

    def __iter__(self):
        return self

    def __call__(self):
        # initialize iterator for this instance.

        data_itr = self.block_data()

        for eeg_block in data_itr:
            tetnums = eeg_block.columns
            times_list = eeg_block.index
            data_mat = eeg_block.values
            # looping through each block of the dataframe
            for row_ii, timestamp in enumerate(times_list):
                for col_ii, tetnum in enumerate(tetnums):
                    tet_data = data_mat[row_ii,col_ii]
                    if not np.isnan(tet_data):
                        yield LFPPoint(timestamp=int(timestamp * 3),
                                       ntrode_index=self.anim.tetrodes.index(tetnum),
                                       ntrode_id=tetnum,
                                       data=tet_data)


class PosMatData:
    DAY_IND = 0
    TIME_IND = 0
    LINPOS_IND = 1
    PATH_IND = 1
    SEGINDEX_IND = 2
    VELLIN_IND = 3

    def __init__(self, anim, timestep):
        print('PosMatData: INIT start')
        self.timestep = timestep
        self.anim = anim
        path_list = anim.get_posmat_paths()
        # Initialized to empty dataframe to be appended/concated to
        self.data = pd.DataFrame([])
        print(path_list)

        self.days = set([])
        for path in path_list:
            day = path[self.DAY_IND]
            self.days.add(day)
            mat = loadmat(path[self.PATH_IND])
            posdata = mat['linpos'][0, day - 1]

            self.timebin_uniq = {}
            for epoch in anim.epochs:
                postime_epoch = posdata[0, epoch]['statematrix'][0, 0]['time'][0, 0]
                poslindist_epoch = posdata[0, epoch]['statematrix'][0, 0]['linearDistanceToWells'][0, 0]
                possegind_epoch = posdata[0, epoch]['statematrix'][0, 0]['segmentIndex'][0, 0]
                poslinvel_epoch = posdata[0, epoch]['statematrix'][0, 0]['linearVelocity'][0, 0]
                posdata_all = np.hstack((postime_epoch, poslindist_epoch, possegind_epoch, poslinvel_epoch))

                pos_pd_idx = pd.MultiIndex.from_product(
                    ([day], [epoch], range(postime_epoch.size)), names=['day', 'epoch', 'idx'])

                pos_pd_col = pd.MultiIndex.from_tuples(
                    [('time', 0), ('lin_dist_well', 'well_center'),
                     ('lin_dist_well', 'well_left'), ('lin_dist_well', 'well_right'),
                     ('seg_idx', 0), ('lin_vel', 'well_center'),
                     ('lin_vel', 'well_left'), ('lin_vel', 'well_right')])

                posdata_all_df = pd.DataFrame(posdata_all, index=pos_pd_idx, columns=pos_pd_col)

                # number 'mask' that assigns each row to its appropriate
                # timestep timebin based on its timestamp.  -1 for 0 offset
                timestep_mask = ((posdata_all_df['time'] -
                                  self.anim.times[day][epoch][0] / 10000.).
                                 floordiv(timestep / 10000.)) - 1

                timestep_ind = pd.MultiIndex.from_tuples([('time_step_bin', 0)])
                timestep_mask.columns = timestep_ind
                posdata_all_df[timestep_ind] = timestep_mask

                # Create a timebin mask set (unique) to use as iterator.
                # Remove all negative timebins because those are outside of range
                timebin_set = np.unique(timestep_mask.values.ravel()).astype('int', copy=False)
                self.timebin_uniq[(day, epoch)] = timebin_set[np.nonzero(timebin_set >= 0)]

                self.data = self.data.append(posdata_all_df)

        self.data.sortlevel(0)
        #print self.data
        print('PosMatData: INIT done')

    def __call__(self):
        for day in self.days:
            for epoch in self.anim.epochs:
                # get the list of unique timebins
                timebin_itr = self.timebin_uniq[(day, epoch)]
                # align start to epoch start time
                epoch_start_time = self.anim.times[day][epoch][0]

                cur_time = epoch_start_time

                next_time = cur_time

                # For every timebin, return the list of positions
                for timebin_num in timebin_itr:

                    #print 'PosMatData: new timebin {} requested'.format(timebin_num)
                    # isolate rows that are in the current time bin
                    #timebin_mask = self.data.loc[day,epoch]['time_step_bin'] == timebin_num
                    #timebin_idx = timebin_mask[timebin_mask].dropna().index.values.ravel()
                    #data_rows = self.data.iloc[timebin_idx]
                    data_rows = self.data[self.data['time_step_bin'][0] == timebin_num]

                    poslist = []
                    for row_series in data_rows.iterrows():
                        row = row_series[1]
                        segind = row['seg_idx', 0]
                        postime = int(row['time', 0] * 10000)

                        if segind == 1:
                            pos_conv = row['lin_dist_well', 'well_center']
                            veldata = row['lin_vel', 'well_center']
                        elif segind == 2 or segind == 3:
                            pos_conv = row['lin_dist_well', 'well_left'] + 150
                            veldata = row['lin_vel', 'well_left']
                        elif segind == 4 or segind == 5:
                            pos_conv = row['lin_dist_well', 'well_right'] + 300
                            veldata = row['lin_vel', 'well_right']

                        # Position data expected to be [offset position, list of dist to wells, segment id, velocity]
                        poslist.append((postime, (pos_conv,
                                                  row['lin_dist_well'].tolist(),
                                                  row['seg_idx'][0].tolist(), veldata)))

                    yield PosTimepoint(day, epoch, (), poslist)


class PosMatDataStream:
    def __init__(self, anim, timestep):
        self.anim = anim
        self.block_data = PosMatData(anim, timestep)
        self.timestep = timestep

    def __iter__(self):
        return self

    def __call__(self):
        # initialize iterator for this instance.

        data_itr = self.block_data()

        for pos_block in data_itr:
            if not pos_block.data:
                continue

            sorted_block = sorted(pos_block.data, key=(lambda x: x[0]))
            for pos_pt in sorted_block:
                pos_packet = LinPosPoint(timestamp=int(pos_pt[0]*30000), data=pos_pt[1])

                yield pos_packet


class PosData:
    """	Returns blocks of 2D raw pos data specified by AnimalInfo
    
    Opens all files specified by animalinfo and creates a generator
    that can be called/polled to return pos data in bundles of the set timestep
    
    Args:
        anim (AnimalInfo): Defines the animal data and locations
        timestep	time interval for generator to return on each call
    """
    DAY_IND = 0
    DATA_IND = 1
    PATH_IND = 2

    def __init__(self, anim, timestep):
        self.data = []
        self.timestep = timestep
        self.anim = anim
        path_list = anim.get_pos_paths()

        posdata_days = {}
        for path in path_list:
            mat = loadmat(path[self.PATH_IND])
            tmp_posdata = posdata_days.setdefault(path[self.DAY_IND], [])

            filedata = mat['filedata']
            tmp_posdata.append(filedata['params'][0, 0][:, [0, 7, 8]])

        for key in posdata_days.keys():
            tmp_posdata = posdata_days[key]

            total_num = reduce(lambda x, y: x + y.shape[0], tmp_posdata, 0)

            posdata = np.empty([total_num, 3], dtype=np.dtype(np.float32))

            insert_cur = 0
            for pos_sub in tmp_posdata:
                posdata[insert_cur:insert_cur + pos_sub.shape[0], :] = pos_sub
                insert_cur += pos_sub.shape[0]

            # Sort the timestamps within each day
            print('sorting day %d' % key)
            posdata = posdata[np.argsort(posdata[:, 0]), :]
            self.data.append([key, posdata])

    def __call__(self):
        # extract number of days into a unique day list
        day_list = map(lambda x: x[self.DAY_IND], self.data)
        days = list(set(day_list))
        for day in days:
            day_data = filter(lambda x: x[self.DAY_IND] == day, self.data)
            for day_i in range(len(day_data)):
                posdata = day_data[day_i][self.DATA_IND]
                day_start_time = self.anim.times[day][0][0]
                day_end_time = self.anim.times[day][-1][-1]

                cur_time = day_start_time

                next_time = cur_time

                pos_cursor = 0

                while cur_time <= day_end_time:
                    next_time = cur_time + self.timestep
                    poslist = []
                    while pos_cursor < len(posdata) and posdata[pos_cursor, 0] <= next_time:
                        poslist.append((posdata[pos_cursor, 0], (posdata[pos_cursor, 1], posdata[pos_cursor, 2])))
                        pos_cursor += 1
                    epoch = self.anim.calc_epoch_state(day, cur_time)
                    new_timepoint = PosTimepoint(day, epoch, (cur_time, next_time), poslist)
                    cur_time = next_time
                    yield new_timepoint

