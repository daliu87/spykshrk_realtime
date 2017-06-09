import struct
import json
from collections import OrderedDict
from spykshrk.realtime import realtime_process

from mpi4py import MPI


class TimingSystemError(Exception):
    pass


class TimingMessage(realtime_process.RealtimeMessage):
    msgheader_format = '=12sqi'
    msgheader_format_size = struct.calcsize(msgheader_format)
    timept_format = '=id'
    timept_format_size = struct.calcsize(timept_format)

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
    def unpack(cls, message_bytes, message_len=None):
        """
        
        Args:
            message_bytes: 
            message_len: 

        Returns:
            TimingMessage: The de-serialized TimingMessage object.
        """

        if message_len is None:
            _, _, num_time_pt = struct.unpack(cls.msgheader_format,
                                              message_bytes[0:cls.msgheader_format_size])
        else:
            # shortcut to avoid extra struct.unpack command
            num_time_pt = ((message_len - cls.msgheader_format_size) /
                           float(cls.timept_format_size))
            if not num_time_pt.is_integer():
                raise TimingSystemError('Unpacking timing message length {}: '
                                        'number of bytes invalid.'.format(message_len))

        num_time_pt = int(num_time_pt)
        unpacked = struct.unpack(cls.msgheader_format + cls.timept_format[1:] * num_time_pt, message_bytes)

        timing_data = []
        for pt_ii in range(num_time_pt):
            timing_data.append((unpacked[3+pt_ii*2], unpacked[3+pt_ii*2+1]))

        return cls(label=unpacked[0], timestamp=unpacked[1], timing_data=timing_data)

    def pack(self):
        return struct.pack(self.msgheader_format + self.timept_format[1:] * len(self._timing_data),
                           self._label, self._timestamp, len(self._timing_data),
                           *[it for sub in self._timing_data for it in sub])

    @property
    def label(self):
        return self._label.split(b'\0', 1)[0].decode()

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

    header_format = '=Q'

    @staticmethod
    def format_full_path(save_dir, file_prefix, mpi_rank, file_postfix):
        file_path = os.path.join(save_dir, '{}.{:02d}.{}'.format(file_prefix, mpi_rank,
                                                                 file_postfix))

        return file_path

    def __init__(self, save_dir, file_prefix, mpi_rank, file_postfix):
        self._save_dir = save_dir
        self._file_prefix = file_prefix
        self._mpi_rank = mpi_rank
        self._file_postfix = file_postfix

        self._file_path = self.format_full_path(save_dir=self._save_dir, file_prefix=self._file_prefix,
                                                mpi_rank=self._mpi_rank, file_postfix=self._file_postfix)

        self._header_json = json.dumps(OrderedDict([('file_type', 'timing'),
                                                    ('file_prefix', self._file_prefix),
                                                    ('mpi_rank', self._mpi_rank),
                                                    ('header_fmt', self.header_format),
                                                    ('msgheader_fmt', TimingMessage.msgheader_format),
                                                    ('timept_fmt', TimingMessage.timept_format)
                                                    ]))

        self._file_handle = open(self._file_path, 'wb')

        self._file_handle.write(bytearray(self._header_json, encoding='utf-8'))

        self._rec_counter = 0

    def write_timing_message(self, timing_msg: TimingMessage):
        header_bytes = struct.pack(self.header_format, self._rec_counter)
        msg_bytes = timing_msg.pack()
        self._file_handle.write(header_bytes + msg_bytes)

        self._rec_counter += 1

    def __del__(self):
        self._file_handle.close()

    @property
    def closed(self):
        return self._file_handle.closed

    def close(self):
        self._file_handle.close()


class TimingFileReader:
    def __init__(self):
