
from enum import Enum
import struct
from mpi4py import MPI


class Datatypes(Enum):
    CONTINUOUS = 1
    POSITION = 2
    SPIKES = 3


class SpikePoint:
    def __init__(self, timestamp, ntrode_id, data):
        self.timestamp = timestamp
        self.ntrode_id = ntrode_id
        self.data = data


class LFPPoint:
    _byte_format = '=qiii'

    def __init__(self, timestamp, ntrode_index, ntrode_id, data):
        self.timestamp = timestamp
        self.ntrode_index = ntrode_index
        self.ntrode_id = ntrode_id
        self.data = data

    def pack(self):
        return struct.pack(self._byte_format, self.timestamp, self.ntrode_index, self.ntrode_id, self.data)

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, ntrode_index, ntrode_id, data = struct.unpack(cls._byte_format, message_bytes)
        message = cls(timestamp=timestamp, ntrode_index=ntrode_index, ntrode_id=ntrode_id, data=data)
        return message

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

