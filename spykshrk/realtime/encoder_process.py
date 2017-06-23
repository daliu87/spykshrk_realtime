import os
from mpi4py import MPI
from spykshrk.realtime import realtime_base, realtime_logging, binary_record, datatypes
from spykshrk.realtime.simulator import simulator_process

from spykshrk.realtime.datatypes import SpikePoint
from spykshrk.realtime.realtime_base import ChannelSelection, TurnOnDataStream


class RStarEncoderManager(realtime_logging.LoggingClass):

    def __init__(self, rank, local_rec_manager, send_interface,
                 data_interface: simulator_process.SimulatorRemoteReceiver):

        super(RStarEncoderManager, self).__init__(rank=rank,
                                                  local_rec_manager=local_rec_manager,
                                                  rec_ids=[realtime_base.RecordIDs.ENCODER_INPUT,
                                                           realtime_base.RecordIDs.ENCODER_OUTPUT],
                                                  rec_labels=[['TBD'],
                                                              ['TBD']],
                                                  rec_formats=['i',
                                                               'i'])
        self.rank = rank
        self.mpi_send = send_interface
        self.data_interface = data_interface

    def set_num_trodes(self, message: realtime_base.NumTrodesMessage):
        self.num_ntrodes = message.num_ntrodes
        self.class_log.info('Set number of ntrodes: {:d}'.format(self.num_ntrodes))

    def select_ntrodes(self, ntrode_list):
        self.class_log.debug("Registering continuous channels: {:}.".format(ntrode_list))
        for ntrode in ntrode_list:
            self.data_interface.register_datatype_channel(channel=ntrode)

    def turn_on_datastreams(self):
        self.class_log.info("Turn on datastreams.")
        self.data_interface.start_all_streams()

    def trigger_termination(self):
        self.data_interface.stop_iterator()

    def process_next_data(self):

        msgs = self.data_interface.__next__()

        if msgs is None:
            # No data avaliable but datastreams are still running, continue polling
            pass
        else:
            datapoint = msgs[0]
            timing_msg = msgs[1]
            if isinstance(datapoint, SpikePoint):
                pass
                #self.class_log.debug('Received SpikePoint.')


class EncoderMPIRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, encoder_manager):
        super(EncoderMPIRecvInterface, self).__init__(comm=comm, rank=rank, config=config)
        self.enc_man = encoder_manager

        self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def __next__(self):
        rdy, msg = self.req.test()
        if rdy:
            self.process_request_message(msg)

            self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def process_request_message(self, message):

        if isinstance(message, realtime_base.TerminateMessage):
            self.class_log.debug("Received TerminateMessage")
            raise StopIteration()

        elif isinstance(message, realtime_base.NumTrodesMessage):
            self.class_log.debug("Received number of NTrodes Message.")
            self.enc_man.set_num_trodes(message)

        elif isinstance(message, ChannelSelection):
            self.class_log.debug("Received NTrode channel selection {:}.".format(message.ntrode_list))
            self.enc_man.select_ntrodes(message.ntrode_list)

        elif isinstance(message, TurnOnDataStream):
            self.class_log.debug("Turn on data stream")
            self.enc_man.turn_on_datastreams()


class EncoderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(EncoderMPISendInterface, self).__init__(comm=comm, rank=rank, config=config)


class EncoderProcess(realtime_base.RealtimeMPIClass, metaclass=realtime_base.ProfilerWrapperMeta):
    def __init__(self, comm: MPI.Comm, rank, config):

        super().__init__(comm, rank, config)

        self.enable_profiler = rank in self.config['rank_settings']['enable_profiler']
        self.profiler_out_path = os.path.join(config['files']['output_dir'], '{}.{:02d}.{}'.
                                              format(config['files']['prefix'],
                                                     rank,
                                                     config['files']['profile_postfix']))
        self.class_log.debug('Class init')

        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
                                                                          manager_rank=config['rank']['supervisor'])

        self.mpi_send = EncoderMPISendInterface(comm=comm, rank=rank, config=config)

        if self.config['datasource'] == 'simulator':
            data_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                       rank=self.rank,
                                                                       config=self.config,
                                                                       datatype=datatypes.Datatypes.SPIKES)

        self.enc_man = RStarEncoderManager(rank=rank,
                                           local_rec_manager=self.local_rec_manager,
                                           send_interface=self.mpi_send,
                                           data_interface=data_interface)

        self.mpi_recv = EncoderMPIRecvInterface(comm=comm, rank=rank, config=config, encoder_manager=self.enc_man)
        self.class_log.debug('Class init end')

    def main_loop(self):
        self.class_log.debug("main loop")

        while True:
            self.mpi_recv.__next__()
            self.enc_man.process_next_data()

        self.class_log.info('Terminating EncodingProcess (rank: {:})'.format(self.rank))

        self.class_log.info("Encoding Process reached end, exiting.")
