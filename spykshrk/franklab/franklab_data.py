import os
import time
import re
import pandas as pd
import warnings
import numpy as np

from abc import ABCMeta, abstractmethod

try:
    from IPython.terminal.debugger import TerminalPdb
    bp = TerminalPdb(color_scheme='linux').set_trace
except AttributeError as err:
    # print('Warning: Attribute Error ({}), disabling IPython TerminalPdb.'.format(err))
    bp = lambda: None

idx = pd.IndexSlice


class FrankFileFormatError(RuntimeError):
    pass


class FrankFileFormatWarning(RuntimeWarning):
    pass


class FrankFilenameParser:

    frank_filename_re = re.compile('^(\d*)_([a-zA-Z0-9]*)_(\w*)\.(\w*)$')

    def __init__(self, filename_str):
        self.filename = filename_str
        filename_match = self.frank_filename_re.match(self.filename)

        if filename_match is not None:
            filename_groups = filename_match.groups()
            self.date = filename_groups[0]
            try:
                self.date_expand = FrankFilenameParser.expand_date_str(self.date)
            except ValueError:
                raise FrankFileFormatError('Filename ({}) date field ({}) is not a true date.'.
                                           format(self.filename, self.date))

            self.anim_name = filename_groups[1]
            self.datatype = filename_groups[2]
            self.ext = filename_groups[3]

    @staticmethod
    def expand_date_str(date_str):
        return time.strptime(date_str, '%Y%m%d')


class FrankAnimalInfo:
    def __init__(self, base_dir, anim_name):
        self.base_dir = base_dir
        self.anim_name = anim_name

        self.data_dir = FrankAnimalInfo._get_analysis_dir(self.base_dir, self.anim_name)
        self.data_paths = FrankAnimalInfo._get_data_path_df(self.data_dir)

    def get_dates(self):
        return list(self.data_paths['date'].unique())

    def get_data_path(self, date, datatype):
        path = self.data_paths[(self.data_paths['datatype'] == datatype) &
                               (self.data_paths['date'] == date)]

        if len(path) < 1:
            raise FrankFileFormatError('Animal ({}) missing ({}) data for date ({}).'.format(self.anim_name,
                                                                                             datatype,
                                                                                             date))

        if len(path) > 1:
            raise FrankFileFormatError(('Animal ({}) has more than one date ({}) for ({}) data, '
                                        'likely a bug or corruption of the animal info.').format(self.anim_name,
                                                                                                 datatype,
                                                                                                 date))

        return path.iloc[0].path

    @staticmethod
    def _get_analysis_dir(base_dir, anim_name):
        return os.path.join(base_dir, anim_name, 'analysis')

    @staticmethod
    def _get_data_path_df(data_path):
        data_path_entries = os.scandir(data_path)
        path_df = pd.DataFrame(columns=['animal', 'date', 'datatype', 'ext', 'path'])

        for path_entry in data_path_entries:    # type: os._DummyDirEntry
            filename_parser = FrankFilenameParser(path_entry.name)

            if filename_parser.ext != 'h5':
                warnings.warn('Found file that does not have h5 extension ({}), skipping.'.
                              format(filename_parser.filename))

            path_df = path_df.append(dict(zip(path_df.columns, [filename_parser.anim_name,
                                                                filename_parser.date,
                                                                filename_parser.datatype,
                                                                filename_parser.ext,
                                                                path_entry.path])),
                                     ignore_index=True)

        path_df = path_df.sort_values(['animal', 'date', 'datatype']).reset_index(drop=True)

        return path_df

    def get_base_name(self, date, datatype):
        return date + '_' + self.anim_name + '_' + datatype + '.h5'


class BaseData(metaclass=ABCMeta):

    def __enter__(self):
        return self

    def __del__(self):
        self.store.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.store.close()

    @property
    @abstractmethod
    def store(self):
        pass

    @store.setter
    @abstractmethod
    def store(self, value):
        pass

    @staticmethod
    @abstractmethod
    def get_hierarchy_base_str():
        return '/preprocessing/Module/'

    @staticmethod
    def get_hierarchy_epoch_str(epoch):
        return 'e{:02d}/'.format(epoch)


class RawPosData(BaseData):

    store = None

    def __init__(self, anim: FrankAnimalInfo, date):
        self.anim = anim
        self.date = date
        self.path = self.anim.get_data_path(date, 'rawpos')

        self.store = pd.HDFStore(self.path, mode='r')

    def get_data(self, epoch, recon_type):
        return self.store[self.get_hiearchy_str(epoch, recon_type) + 'data']

    @staticmethod
    def get_hierarchy_base_str():
        return '/preprocessing/Position/'

    @classmethod
    def get_hiearchy_str(cls, epoch, recon_type):
        return (cls.get_hierarchy_base_str() +
                cls.get_hierarchy_epoch_str(epoch) +
                recon_type + '/')


class LFPData(BaseData):

    store = None

    def __init__(self, anim: FrankAnimalInfo, date):
        self.anim = anim
        self.date = date
        self.path = self.anim.get_data_path(date, 'lfp')

        self.store = pd.HDFStore(self.path, mode='r')

    def get_data(self, epoch, ntrodes=None):
        lfp_df = self.store[self.get_hierarchy_str(epoch) + 'data']
        if ntrodes is None:
            return lfp_df
        else:
            return lfp_df.loc[:, idx[ntrodes, :]]

    @staticmethod
    def get_hierarchy_base_str():
        return '/preprocessing/LFP/'

    @classmethod
    def get_hierarchy_str(cls, epoch):
        return (cls.get_hierarchy_base_str() +
                cls.get_hierarchy_epoch_str(epoch))


class SpikeBaseData(BaseData, metaclass=ABCMeta):

    @staticmethod
    def get_hierarchy_ntrode_str(ntrode):
        return 't{:02d}/'.format(ntrode)

    def get_epochs(self):
        epoch_list = []
        epoch_re = re.compile('^e(\d*)$')
        for epoch_node in self.store.get_node(self.get_hierarchy_base_str()):
            epoch_list.append(int(epoch_re.match(epoch_node._v_name).groups()[0]))

        return epoch_list

    def get_ntrodes(self):

        ntrode_set = set()
        ntrode_re = re.compile('^t(\d*)$')
        for epoch_node in self.store.get_node(self.get_hierarchy_base_str()):
            for ntrode_node in epoch_node:
                ntrode_set.add(int(ntrode_re.match(ntrode_node._v_name).groups()[0]))

        ntrode_list = list(ntrode_set)
        ntrode_list.sort()
        return ntrode_list

    def get_data(self, epoch, ntrode):
        return self.store[self.get_hierarchy_str(epoch, ntrode) + 'data']

    @classmethod
    def get_hierarchy_str(cls, epoch, ntrode):
        return (cls.get_hierarchy_base_str() +
                cls.get_hierarchy_epoch_str(epoch) +
                cls.get_hierarchy_ntrode_str(ntrode))


class SpikeAmpData(SpikeBaseData):

    store = None

    def __init__(self, anim: FrankAnimalInfo, date):
        self.anim = anim
        self.date = date

        self.path = self.anim.get_data_path(date, 'spikeamp')

        self.store = pd.HDFStore(self.path, mode='r')

    @staticmethod
    def get_hierarchy_base_str():
        return '/preprocessing/FeatureExtraction/'


class SpikeWaveData(SpikeBaseData):

    store = None

    def __init__(self, anim: FrankAnimalInfo, date):
        self.anim = anim
        self.date = date

        self.path = self.anim.get_data_path(date, 'spikeamp')

        self.store = pd.HDFStore(self.path, mode='r')


    @staticmethod
    def get_hierarchy_base_str():
        return '/preprocessing/EventWaveform/'


