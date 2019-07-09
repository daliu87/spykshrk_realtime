import datajoint as dj
import os
import re
import warnings
import glob
from scipy.io import loadmat

from spykshrk.franklab.pipeline.exceptions import CorruptData, InconsistentData

'''
schema = dj.schema('franklab_nspike', locals())
schema.drop()
schema = dj.schema('franklab_nspike', locals())
'''

schema = dj.schema('franklab_nspike', locals())


@schema
class WarningTable(dj.Manual):
    definition = """
    warning_id: int unsigned auto_increment
    ---
    warning_name: varchar(50)
    table_name: varchar(50)
    field_name: varchar(200)
    key_dict: varchar(200)
    warning_info: varchar(50)
    """

    suppress_warning = False

    def __init__(self, suppress_print=False, arg=None):
        self.suppress_print = suppress_print
        super().__init__(arg)
        
    @classmethod
    def missing_field(cls, table, key, field):
        if not cls.suppress_warning:
             warnings.warn('Missing entry for table {:s} {:s}: {:s}'.format(table, str(key), field), InconsistentData)
        cls.insert([{'warning_name': 'missing_field',
                     'table_name': table,
                     'field_name': field,
                     'key_dict': str(key),
                     'warning_info': ''}])

    @classmethod
    def duplicate(cls, table, key, field, using_first=True):
        if using_first:
            warning_info = 'using first'
        else:
            warning_info = 'not using first'
        if not cls.suppress_warning:
            warnings.warn('Duplicate entry for table {:s} {:s}: {:s} "{:s}".'.
                          format(table, str(key), field, warning_info), InconsistentData)
        cls.insert([{'warning_name': 'duplicate',
                     'table_name': table,
                     'field_name': field,
                     'key_dict': str(key),
                     'warning_info': warning_info}])

    @classmethod
    def duplicate_nonmatch(cls, table, key, field, value1, value2):
        if not cls.suppress_warning:
            warnings.warn('Duplicate entry for table {:s}, {:s}: {:s}, values are ({:s}, {:s})'.
                          format(table, str(key), field, value1, value2), InconsistentData)
        cls.insert([{'warning_name': 'duplicate_nonmatch',
                     'table_name': table,
                     'field_name': field,
                     'key_dict': str(key),
                     'warning_info': '('+str(value1)+'), ('+str(value2)+')'}])


@schema
class Animal(dj.Manual):
    definition = """
    anim_name: varchar(20)  #Name of animal
    ---
    anim_name_short: varchar(10)
    anim_path_raw: varchar(200)
    anim_path_mat: varchar(200)
    """

@schema
class Day(dj.Imported):
    definition = """
    -> Animal
    day: int
    ---
    day_path_raw: varchar(200)
    day_path_mat: varchar(200)
    day_start_time_sec: float
    day_end_time_sec: float
    day_start_time_nspike: int
    day_end_time_nspike: int
    """

    def make(self, key):

        anim_name, anim_path_raw, anim_path_mat = (Animal() & key).fetch1('anim_name', 'anim_path_raw', 'anim_path_mat')
        dir_names = os.listdir(anim_path_raw)
        for dir_name in dir_names:
            m = re.search('^{:s}(\d*)$'.format(anim_name.lower()), dir_name)
            if m:
                day = int(m.groups()[0])
                day_path_raw = os.path.join(anim_path_raw, dir_name)
                times_path = os.path.join(day_path_raw, 'times.mat')
                if os.path.isfile(times_path):
                    times_mat = loadmat(times_path)
                    time_ranges = times_mat['ranges']
                    day_start_time_nspike = time_ranges[0][0]
                    day_end_time_nspike = time_ranges[0][1]
                    day_start_time_sec = day_start_time_nspike/10000
                    day_end_time_sec = day_end_time_nspike/10000
                    self.insert1({'anim_name': anim_name,
                                  'day': day,
                                  'day_path_raw': day_path_raw,
                                  'day_path_mat': anim_path_mat,
                                  'day_start_time_sec': day_start_time_nspike,
                                  'day_end_time_sec': day_end_time_nspike,
                                  'day_start_time_nspike': day_start_time_sec,
                                  'day_end_time_nspike': day_end_time_sec})
                else:
                    # Missing times.mat means data folder was not processed for spike sorting (matclust)
                    WarningTable.missing_field(type(self).__name__, key, times_path)
                    pass


@schema
class Epoch(dj.Imported):
    definition = """
    -> Day
    epoch_id: tinyint
    ---
    epoch_name: varchar(50)
    epoch_time_str: varchar(50)
    epoch_start_time_sec: float
    epoch_end_time_sec: float
    epoch_start_time_nspike: int
    epoch_end_time_nspike: int
    """

    def make(self, key):
        anim_name, day, day_path_raw = (Animal() * (Day() & key)).fetch1('anim_name', 'day', 'day_path_raw')
        try:
            times_path = os.path.join(day_path_raw, 'times.mat')
            times_mat = loadmat(times_path)
            time_ranges = times_mat['ranges']
            names = times_mat['names']
            for epoch_id, epoch_time_range in enumerate(time_ranges[1:]):
                epoch_start_time_nspike = epoch_time_range[0]
                epoch_end_time_nspike = epoch_time_range[1]
                epoch_start_time_sec = epoch_start_time_nspike/10000
                epoch_end_time_sec = epoch_end_time_nspike/10000
                name_entry = names[epoch_id + 1][0][0]
                name_re = re.search('^\d*\s*(\w*)\s*([0-9:\-_]*)$', name_entry)
                if name_re:
                    epoch_name = name_re.groups()[0]
                    epoch_time_str = name_re.groups()[1]
                    self.insert1({'anim_name': anim_name,
                                  'day': day,
                                  'epoch_id': epoch_id,
                                  'epoch_name': epoch_name,
                                  'epoch_time_str': epoch_time_str,
                                  'epoch_start_time_sec': epoch_start_time_sec,
                                  'epoch_end_time_sec':epoch_start_time_sec,
                                  'epoch_start_time_nspike': epoch_start_time_nspike,
                                  'epoch_end_time_nspike': epoch_end_time_nspike
                                 })

        except FileNotFoundError:
            WarningTable.missing_field(type(self).__name__, key, times_path)


@schema
class Environment(dj.Lookup):
    definition = """
    env_name: varchar(20)
    ---
    evn_type: varchar(20)
    env_desc: varchar(20)

    """
    contents = [['TrackA', 'run', 'TrackA'],
                ['TrackB', 'run', 'TrackB'],
                ['Sleep', 'sleep', 'Sleep']
               ]

@schema
class Session(dj.Imported):
    definition = """
    -> Epoch
    ses_id: smallint unsigned
    ---
    -> Environment
    ses_prev: smallint unsigned
    ses_next: smallint unsigned
    exposure: smallint unsigned
    experimentday: smallint unsigned
    exposureday: smallint unsigned
    dailyexposure: smallint unsigned
    ses_numca1tets: smallint unsigned
    ses_tetmostcells: blob
    ses_tetmostcells_regions: blob 
    ses_tet2ndmostcells: blob
    ses_num_gammaftets: smallint unsigned
    ses_sta_reftet=NULL: blob
    ses_sta_reftet_desc=NULL: blob
    ses_sta_reftet_left=NULL: blob
    ses_sta_reftet_left_desc=NULL: blob
    """

    class RunTask(dj.Part):
        definition = """
        -> Session
        ---
        linearcoord: blob
        """

    class SleepTask(dj.Part):
        definition = """
        -> Session
        ---
        sleepnum: smallint unsigned
        """
    def make(self, key):
        anim_name, anim_path_mat, anim_name_short = (Animal() & key).fetch1('anim_name', 'anim_path_mat', 'anim_name_short')
        ses_task_fps = glob.glob(os.path.join(anim_path_mat, '{:s}task{:02d}.mat'.
                                              format(anim_name_short, key['day'])))
        if len(ses_task_fps) > 1:
            self.warn.duplicate(key, ses_task_fps, using_first=True)
        ses_task_mat = loadmat(ses_task_fps[0])[0]
        ses_task_ep = ses_task_mat[key['day']][0][key['epoch']][0]

        ses_type = ses_task_ep['task'][0][0][0]
        ses_desc = ses_task_ep['description'][0][0][0]
        ses_env = ses_task_ep['environment'][0][0][0]

        env_row = Environment() & 'env_name = "{:s}"'.format(ses_env)
        # if(len(env_row)):



@schema
class Tetrode(dj.Imported):
    definition = """
    -> Animal
    tet_id: tinyint
    ---
    tet_hemisphere: varchar(50)
    """

    def make(self, key):
        anim_name, anim_path_mat = (Animal() & key).fetch1('anim_name', 'anim_path_mat')
        mat = loadmat(os.path.join(anim_path_mat, 'bontetinfo.mat'))
        tet = {}
        for tet_days in mat['tetinfo'][0]:
            if len(tet_days[0]) > 0:
                for tet_epochs in tet_days[0]:
                    for tet_id, tet_epoch in enumerate(tet_epochs[0]):
                        if len(tet_epoch[0]) > 0:
                            tet_entry = tet.setdefault(tet_id, {})
                            tet_entry_hemi_list = tet_entry.setdefault('hemisphere', [])
                            try:
                                tet_hemi = tet_epoch[0][0]['hemisphere'][0]
                            except ValueError:
                                tet_hemi = None
                            tet_entry_hemi_list.append(tet_hemi)

        for tet_id, tet_entries in tet.items():
            tet_hemi = tet_entries['hemisphere']
            tet_hemi_set = set(tet_hemi)
            if len(tet_hemi_set) == 1:
                tet_hemisphere = list(tet_hemi_set)[0]
                if tet_hemisphere is None:
                    tet_hemisphere = ''
                self.insert1({'anim_name': anim_name,
                              'tet_id': tet_id,
                              'tet_hemisphere': tet_hemisphere
                             })
            else:
                WarningTable.duplicate_nonmatch(type(self).__name__, key, 'hemisphere',
                                                list(tet_hemi_set)[0], list(tet_hemi_set)[1:])


@schema
class TetrodeEpoch(dj.Imported):
    definition = """
    -> Tetrode
    -> Epoch
    ---
    tet_depth = NULL: int
    tet_num_cells = NULL: int
    tet_area = NULL: varchar(50)
    tet_subarea = NULL: varchar(50)
    tet_near_ca2 = NULL: tinyint       # boolean
    """

    def make(self, key):
        anim_name, anim_path_mat = (Animal() & key).fetch1('anim_name', 'anim_path_mat')
        try:
            mat = self.anim_tet_infos[anim_name]
        except AttributeError:
            self.anim_tet_infos = {}
            mat = loadmat(os.path.join(anim_path_mat, 'bontetinfo.mat'))
            self.anim_tet_infos[anim_name] = mat
        except KeyError:
            mat = loadmat(os.path.join(anim_path_mat, 'bontetinfo.mat'))
            self.anim_tet_infos[anim_name] = mat

        try:
            tet_epoch_data = mat['tetinfo'][0][key['day']-1][0][key['epoch_id']][0][key['tet_id']][0]

            # if mat cell is empty, skip insert
            if tet_epoch_data.size > 0:
                try:
                    key['tet_depth'] = tet_epoch_data['depth'][0][0][0][0][0]
                except (ValueError, IndexError):
                    WarningTable.missing_field(type(self).__name__, key, 'tet_depth')
                    # print(key)
                    # leave entry out
                    pass
                    # key['tet_depth'] = None
                try:
                    key['tet_num_cells'] = tet_epoch_data['numcells'][0][0][0]
                except (ValueError, IndexError):
                    WarningTable.missing_field(type(self).__name__, key, 'tet_num_cells')
                    # print(key)
                    # leave entry out
                    pass
                    # key['tet_num_cells'] = None
                try:
                    key['tet_area'] = tet_epoch_data['area'][0][0]
                except (ValueError, IndexError):
                    WarningTable.missing_field(type(self).__name__, key, 'tet_area')
                    # print(key)
                    # leave entry out
                    pass
                    # key['tet_area'] = None
                try:
                    key['tet_subarea'] = tet_epoch_data['subarea'][0][0]
                except (ValueError, IndexError):
                    WarningTable.missing_field(type(self).__name__, key, 'tet_subarea')
                    # print(key)
                    # leave entry out
                    pass
                    # key['tet_subarea'] = None
                try:
                    key['tet_near_ca2'] = tet_epoch_data['nearCA2'][0][0][0]
                except (ValueError, IndexError):
                    WarningTable.missing_field(type(self).__name__, key, 'tet_near_ca2')
                    # print(key)
                    # leave entry out
                    pass
                    # key['tet_near_ca2'] = None
                self.insert1(key)
        except (ValueError, IndexError):
            WarningTable.missing_field(type(self).__name__, key, 'entire tetrode')
            # print(key)
            pass

@schema
class LFP(dj.Imported):
    definition = """
    -> TetrodeEpoch
    ---
    lfp_fp = NULL: varchar(200)
    """

    def make(self, key):
        day_path_mat = (Day() & key).fetch1('day_path_mat')
        try:
            lfp_filepath_eeg_mats = glob.glob(os.path.join(day_path_mat, 'EEG/boneeg{:02d}-{:d}-{:02d}.mat'.
                                                           format(key['day'], key['epoch_id']+1, key['tet_id']+1)))

            if len(lfp_filepath_eeg_mats) > 1:
                WarningTable.duplicate(type(self).__name__, key, lfp_filepath_eeg_mats, using_first=True)
            lfp_filepath_eeg_mat = lfp_filepath_eeg_mats[0]

            key['lfp_fp'] = lfp_filepath_eeg_mat
        except IndexError:
            WarningTable.missing_field(type(self).__name__, key, 'eeg mat')

        self.insert1(key)


@schema
class LFPRaw(dj.Imported):
    definition = """
    -> TetrodeEpoch
    ---
    lfp_raw_fp = NULL: varchar(200)
    lfp_tet_depth = NULL: int
    """

    def make(self, key):
        day_path_raw, day_path_mat = (Day() & key).fetch1('day_path_raw', 'day_path_mat')
        tet_id = key['tet_id']
        lfp_filepath_raw = glob.glob(os.path.join(day_path_raw, '{:02d}-*.eeg').format(tet_id+1))[0]
        lfp_filename_raw = os.path.basename(lfp_filepath_raw)
        re_match = re.search('\d*-(\d*).eeg$', lfp_filename_raw)
        lfp_tet_depth_str = re_match.groups()[0]
        lfp_tet_depth = int(lfp_tet_depth_str)

        key['lfp_raw_fp'] = lfp_filepath_raw
        key['lfp_tet_depth'] = lfp_tet_depth

        self.insert1(key)


@schema
class LFPGnd(dj.Imported):
    definition = """
    -> TetrodeEpoch
    ---
    lfp_filepath_eeggnd_mat = NULL: varchar(200)
    """

    def make(self, key):
        day_path_mat = (Day() & key).fetch1('day_path_mat')

        try:
            lfp_filepath_eeggnd_mats = glob.glob(os.path.join(day_path_mat, 'EEG/boneeggnd{:02d}-{:d}-{:02d}.mat'.
                                                              format(key['day'], key['epoch_id']+1, key['tet_id']+1)))
            if len(lfp_filepath_eeggnd_mats) > 1:
                WarningTable.duplicate(type(self).__name__, key, lfp_filepath_eeggnd_mats)

            lfp_filepath_eeggnd_mat = lfp_filepath_eeggnd_mats[0]

            key['lfp_filepath_eeggnd_mat'] = lfp_filepath_eeggnd_mat
        except IndexError:
            WarningTable.missing_field(type(self).__name__, key, 'eeggnd')

        self.insert1(key)


@schema
class RippleDetectionConfig(dj.Lookup):
    definition = """
    rip_alg : varchar(20)
    rip_detect_thresh : decimal(5,2)
    rip_min_thresh_dur : decimal(6,4)
    rip_tet_filter : varchar(200)
    ---

    """

    contents = [['cons', 2.0, 0.0300, "'(isequal($validripple, 1))'"]]


@schema
class RippleConsInterval(dj.Imported):
    definition = """
    -> Epoch
    -> RippleDetectionConfig
    ---
    rip_cons_fp = NULL: varchar(200)
    eventname = NULL: varchar(40)
    nstd = NULL: decimal(4,2)
    min_suprathresh_dur = NULL: decimal(6,4)
    tetfilter = NULL: varchar(200)
    tetlist = NULL: blob
    starttime = NULL: longblob
    endtime = NULL: longblob 
    maxthresh = NULL: longblob
    smoothwin = NULL: decimal(6,4)
    timerange = NULL: tinyblob
    samprate = NULL: decimal(6,1)
    baseline = NULL: float
    std = NULL: float
    """

    class LFPSource(dj.Part):
        definition = """
        -> LFP
        -> RippleConsInterval
        ---
        
        """

    def make(self, key):
        anim_name_short, day_path_mat = (Animal() * (Day() & key)).fetch1('anim_name_short', 'day_path_mat')
        anim_name = key['anim_name']
        day = key['day']
        epoch_id = key['epoch_id']
        rip_alg = key['rip_alg']
        rip_mat_fp = glob.glob(os.path.join(day_path_mat, '{:s}ripples{:s}{:02d}.mat'.
                                            format(anim_name_short, rip_alg, key['day'])))

        if len(rip_mat_fp) > 1:
            WarningTable.duplicate(type(self).__name__, key, rip_mat_fp, using_first=True)
        elif len(rip_mat_fp) == 0:
            WarningTable.missing_field(type(self).__name__, key,
                                       os.path.join(day_path_mat, '{:s}ripples{:s}{:02d}.mat'.
                                                    format(anim_name_short, rip_alg, key['day'])))

        try:
            print(rip_mat_fp[0])
            rip_mat = self.day_rip_mats[key['day']-1]
            rip_epoch = rip_mat[key['day']-1][0][key['epoch_id']][0]
            rip_default = rip_epoch[0][0]
            key['rip_cons_fp'] = rip_mat_fp[0]
            key['eventname'] = rip_default['eventname'][0][0]
            key['nstd'] = rip_default['nstd'][0][0][0]
            key['min_suprathresh_dur'] = rip_default['min_suprathresh_duration'][0][0][0]
            key['tetfilter'] = rip_default['tetfilter'][0][0]
            key['tetlist'] = rip_default['tetlist'][0][0]
            key['starttime'] = rip_default['starttime'][0][0]
            key['endtime'] = rip_default['endtime'][0][0]
            key['maxthresh'] = rip_default['maxthresh'][0][0]
            key['smoothwin'] = rip_default['smoothwin'][0][0][0]
            key['timerange'] = rip_default['timerange'][0][0]
            key['samprate'] = rip_default['samprate'][0][0][0]
            key['baseline'] = rip_default['baseline'][0][0][0]
            key['std'] = rip_default['std'][0][0][0]
        except AttributeError:
            self.day_rip_mats = {}
            rip_mat = loadmat(rip_mat_fp[0])[str('ripples{:s}'.format(rip_alg))][0]
            self.day_rip_mats[key['day']-1] = rip_mat
        except KeyError:
            rip_mat = loadmat(rip_mat_fp[0])['ripples{:s}'.format(rip_alg)][0]
            self.day_rip_mats[key['day']-1] = rip_mat
        except IndexError:
            WarningTable.missing_field(type(self).__name__, key, '')

        self.insert1(key)



@schema
class RawSpikes(dj.Imported):
    definition = """
    -> TetrodeEpoch
    ---
    raw_spike_path: varchar(200)
    """

@schema
class Position(dj.Imported):
    definition = """
    -> Epoch
    ---
    pos_path: varchar(200)
    """

@schema
class LinearPosition(dj.Imported):
    definition = """
    -> Position
    ---
    lin_pos_path: varchar(200)
    """
