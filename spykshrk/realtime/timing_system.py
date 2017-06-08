import struct
from spykshrk.realtime import realtime_process

from mpi4py import MPI


class TimingSystemError(Exception):
    pass


class TimingMessage(realtime_process.RealtimeMessage):
    def __init__(self, label, timestamp=0, start_rank=-1, timing_data=None):
        """
        
        Args:
            label: 
            start_rank: 
        """

        if len(label) > 12:
            raise TimingSystemError('Timing label {}: label must be 10 characters or less.'.format(label))

        if isinstance(label, str):
            self._label = bytes(label, 'utf-8')
        elif isinstance(label, bytes) or isinstance(label, bytearray):
            self._label = label

        self._timestamp = timestamp

        if timing_data is None:
            self._timing_data = [(start_rank, MPI.Wtime())]
        else:
            self._timing_data = timing_data

    def record_time(self, rank):
        self._timing_data.append((rank, MPI.Wtime()))

    @classmethod
    def unpack(cls, message_bytes, message_len):
        num_time_rec = (message_len - 12) / 12.
        if not num_time_rec.is_integer():
            raise TimingSystemError('Unpacking timing message length {}: number of bytes invalid.'.format(message_len))

        num_time_rec = int(num_time_rec)
        unpacked = struct.unpack('=12s'+'id'*num_time_rec, message_bytes)

        timing_data = []
        for rec_ii in range(num_time_rec):
            timing_data.append((unpacked[1+rec_ii*2], unpacked[1+rec_ii*2+1]))

        return cls(label=unpacked[0], timing_data=timing_data)

    def pack(self):
        return struct.pack('=12s' +'id' * len(self._timing_data),
                           self._label, *[it for sub in self._timing_data for it in sub])

    @property
    def label(self):
        return self._label.decode()

    @label.setter
    def label(self, label):
        raise TimingSystemError('Cannot modify TimingMessage label.')

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, label):
        raise TimingSystemError('Cannot modify TimingMessage timestamp.')

    @property
    def timing_data(self):
        return self._timing_data

    @timing_data.setter
    def timing_data(self, timing_data):
        raise TimingSystemError('Cannot modify TimingMessage timing_data.')



class TimingFileWriter:

    def __init__(self, save_dir, file_prefix, mpi_rank, file_postfix):

