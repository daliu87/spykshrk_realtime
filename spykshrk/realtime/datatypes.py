
from enum import Enum


class Datatypes(Enum):
    CONTINUOUS = 1
    POSITION = 2
    SPIKES = 3


class SpikePoint:
    def __init__(self, timestamp, ntrode_index, data):
        self.timestamp = timestamp
        self.ntrode_index = ntrode_index
        self.data = data


class LFPPoint:
    def __init__(self, timestamp, ntrode_index, data):
        self.timestamp = timestamp
        self.ntrode_index = ntrode_index
        self.data = data

    def __str__(self):
        return '{:}({:})'.format(self.__class__.__name__, self.__dict__)


class RawPosPoint:
    def __init__(self, timestamp, x1, y1, x2, y2, camera_id):
        self.timestamp = timestamp
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.camera_id = camera_id

    def __str__(self):
        return '{:}({:})'.format(self.__class__.__name__, self.__dict__)


class PosPoint:
    def __init__(self, timestamp, x, y, camera_id):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.camera_id = camera_id

    def __str__(self):
        return '{:}({:})'.format(self.__class__.__name__, self.__dict__)


class LinPosPoint:
    def __init__(self, timestamp, data):
        self.timestamp = timestamp
        self.data = data

    def __str__(self):
        return '{:}({:})'.format(self.__class__.__name__, self.__dict__)


class DigIOStateChange:
    def __init__(self, timestamp, port, io_dir, state):
        self.timestamp = timestamp
        self.port = port
        self.io_dir = io_dir        # 1 - input, 0 - output
        self.state = state

    def __str__(self):
        return '{:}({:})'.format(self.__class__.__name__, self.__dict__)


class SystemTimePoint:
    def __init__(self, timestamp, tv_sec, tv_nsec):
        self.timestamp = timestamp
        self.tv_sec = tv_sec
        self.tv_nsec = tv_nsec

    def __str__(self):
        return '{:}({:})'.format(self.__class__.__name__, self.__dict__)

