import threading
from mpi4py import MPI
import logging
from abc import ABCMeta, abstractmethod
from enum import Enum

import spykshrk.realtime.binary_record as binary_record


class MPIMessageTag(Enum):
    COMMAND_MESSAGE = 1
    SIMULATOR_DATA = 2


class RealtimeClass(object):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.class_log = logging.getLogger(name='{}.{}'.format(self.__class__.__module__,
                                                               self.__class__.__name__))


class BinaryRecordBaseError(RuntimeError):
    pass


class BinaryRecordBase(RealtimeClass):
    def __init__(self, local_rec_manager: binary_record.RemoteBinaryRecordsManager, rec_id, rec_labels, rec_format,
                 *args, **kwds):
        super().__init__(*args, **kwds)
        self.local_rec_manager = local_rec_manager
        self.rec_id = rec_id
        self.rec_labels = rec_labels
        self.rec_struct_fmt = rec_format
        self.rec_writer = None    # type: binary_record.BinaryRecordsFileWriter
        self.rec_writer_enabled = False

    def get_record_register_message(self):
        return self.local_rec_manager.create_register_rec_type_message(rec_id=self.rec_id, rec_labels=self.rec_labels,
                                                                       rec_struct_fmt=self.rec_struct_fmt)

    def set_record_writer_from_message(self, create_message):
        self.class_log.info('Creating record from message {}'.format(create_message))
        self.set_record_writer(self.local_rec_manager.create_writer_from_message(create_message))

    def set_record_writer(self, rec_writer):
        self.class_log.info('Setting record writer {}'.format(rec_writer))
        if self.rec_writer_enabled:
            raise BinaryRecordBaseError('Can\'t change writer file while recording is on going!')
        self.rec_writer = rec_writer

    def start_record_writing(self):
        if self.rec_writer:
            if not self.rec_writer.closed:
                self.rec_writer_enabled = True
            else:
                raise BinaryRecordBaseError('Can\'t start recording, file not open!')
        else:
            raise BinaryRecordBaseError('Can\'t start recording, record file never set!')

    def stop_record_writing(self):
        self.rec_writer_enabled = False

    def close_record(self):
        self.stop_record_writing()
        if self.rec_writer:
            self.rec_writer.close()

    def write_record(self, *args):
        if self.rec_writer_enabled and not self.rec_writer.closed:
            self.rec_writer.write_rec(self.rec_id, *args)
            return True
        return False


class RealtimeMessage:

    def __str__(self):
        return '{:}({:})'.format(self.__class__.__name__, self.__dict__)


class ExceptionLoggerWrapperMeta(type):
    """
        A metaclass that wraps the run() method so exceptions are printed to sys.stdout instead of sys.stderr.

        This metaclass is built to solve a very specific bug in MPI + threads: threads' output to stderr seem to be
        lost, which causes unhandled exceptions to not be displayed.

        Once MPI + threads logging is setup, exceptions logging should be handled by the logging system, making
        this class obsolete
    """
    @staticmethod
    def wrap(func):
        def outer(self):
            try:
                func(self)
            except Exception as ex:
                logging.exception(ex.args)
                # traceback.print_exc(file=sys.stdout)

        return outer

    def __new__(mcs, name, bases, attrs):
        if 'run' in attrs:
            attrs['run'] = mcs.wrap(attrs['run'])
        if 'main_loop' in attrs:
            attrs['main_loop'] = mcs.wrap(attrs['main_loop'])

        return super(ExceptionLoggerWrapperMeta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        super(ExceptionLoggerWrapperMeta, cls).__init__(name, bases, attrs)


class RealtimeProcess(RealtimeClass, metaclass=ExceptionLoggerWrapperMeta):

    def __init__(self, comm: MPI.Comm, rank, config, ThreadClass, **kwds):

        super().__init__()

        self.comm = comm
        self.rank = rank
        self.config = config

        self.thread = ThreadClass(comm=comm, rank=rank, config=config, parent=self, **kwds)

    def main_loop(self):
        self.thread.start()


class RealtimeThread(RealtimeClass, threading.Thread, metaclass=ExceptionLoggerWrapperMeta):

    def __init__(self, comm: MPI.Comm, rank, config, parent):
        super().__init__(name=self.__class__.__name__)
        super().__init__()
        self.parent = parent
        self.comm = comm
        self.rank = rank
        self.config = config


class DataStreamIterator(RealtimeClass, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        super().__init__()
        self.source_infos = {}
        self.source_handlers = []
        self.source_enabled_list = []

    @abstractmethod
    def turn_on_stream(self):
        pass

    @abstractmethod
    def turn_off_stream(self):
        pass

    def set_sources(self, source_infos):
        self.source_infos = source_infos

    def set_enabled(self, source_enable_list):
        self.source_enabled_list = source_enable_list

    @abstractmethod
    def __next__(self):
        pass


class TerminateErrorMessage(RealtimeMessage):
    def __init__(self, message):
        self.message = message
        pass


class TerminateMessage(RealtimeMessage):
    def __init__(self):
        pass


class EnableStimulationMessage(RealtimeMessage):
    def __init__(self):
        pass


class DisableStimulationMessage(RealtimeMessage):
    def __init__(self):
        pass


class StartRecordMessage(RealtimeMessage):
    def __init__(self):
        pass


class StopRecordMessage(RealtimeMessage):
    def __init__(self):
        pass


class CloseRecordMessage(RealtimeMessage):
    def __init__(self):
        pass


class RequestStatusMessage(RealtimeMessage):
    def __init__(self):
        pass


class ResetFilterMessage(RealtimeMessage):
    def __init__(self):
        pass


class NumTrodesMessage(RealtimeMessage):
    def __init__(self, num_ntrodes):
        self.num_ntrodes = num_ntrodes


class TurnOnLFPMessage(RealtimeMessage):
    def __init__(self, lfp_enable_list):
        self.lfp_enable_list = lfp_enable_list


class TurnOffLFPMessage(RealtimeMessage):
    def __init__(self):
        pass


