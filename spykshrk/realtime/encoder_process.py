import os
from mpi4py import MPI
from spykshrk.realtime import realtime_base, realtime_logging, binary_record, datatypes
from spykshrk.realtime.simulator import simulator_process

from spykshrk.realtime.datatypes import SpikePoint, LinearPosPoint
from spykshrk.realtime.realtime_base import ChannelSelection, TurnOnDataStream
from spykshrk.realtime.tetrode_models import kernel_encoder
import spykshrk.realtime.rst.RSTPython as RST


class EncoderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(EncoderMPISendInterface, self).__init__(comm=comm, rank=rank, config=config)

    def send_record_register_messages(self, record_register_messages):
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)


class RStarEncoderManager(realtime_base.BinaryRecordBaseWithTiming, realtime_logging.LoggingClass):

    def __init__(self, rank, local_rec_manager, send_interface: EncoderMPISendInterface,
                 spike_interface: simulator_process.SimulatorRemoteReceiver,
                 pos_interface: simulator_process.SimulatorRemoteReceiver):

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
        self.spike_interface = spike_interface
        self.pos_interface = pos_interface

        kernel = RST.kernel_param(0, 25, -1024, 1024, 1)
        pos_bin_struct = kernel_encoder.PosBinStruct([0, 450], 150)
        self.rst_param = kernel_encoder.RSTParameter(kernel, pos_bin_struct)
        self.encoders = {}

        # Register position, right now only one position channel is supported
        self.pos_interface.register_datatype_channel(-1)

        self.spk_counter = 0
        self.pos_counter = 0

        self.mpi_send.send_record_register_messages(self.get_record_register_messages())

    def set_num_trodes(self, message: realtime_base.NumTrodesMessage):
        self.num_ntrodes = message.num_ntrodes
        self.class_log.info('Set number of ntrodes: {:d}'.format(self.num_ntrodes))

    def select_ntrodes(self, ntrode_list):
        self.class_log.debug("Registering spiking channels: {:}.".format(ntrode_list))
        for ntrode in ntrode_list:
            self.spike_interface.register_datatype_channel(channel=ntrode)

            self.encoders.setdefault(ntrode, kernel_encoder.RSTKernelEncoder('/tmp/ntrode{:}'.format(ntrode),
                                                                             True, self.rst_param))

    def turn_on_datastreams(self):
        self.class_log.info("Turn on datastreams.")
        self.spike_interface.start_all_streams()
        self.pos_interface.start_all_streams()

    def trigger_termination(self):
        self.spike_interface.stop_iterator()

    def process_next_data(self):

        msgs = self.spike_interface.__next__()

        if msgs is None:
            # No data avaliable but datastreams are still running, continue polling
            pass
        else:
            datapoint = msgs[0]
            timing_msg = msgs[1]
            if isinstance(datapoint, SpikePoint):
                self.spk_counter += 1
                amp_marks = [max(x) for x in datapoint.data]

                query_result = self.encoders[datapoint.ntrode_id].query_mark_hist(amp_marks,
                                                                                  datapoint.timestamp,
                                                                                  datapoint.ntrode_id)

                #self.class_log.debug(query_result)
                self.encoders[datapoint.ntrode_id].new_mark(amp_marks)

                if self.spk_counter % 1000 == 0:
                    self.class_log.debug('Received {} spikes.'.format(self.spk_counter))
                pass
                #self.class_log.debug('Received SpikePoint.')

        msgs = self.pos_interface.__next__()
        if msgs is None:
            # No data avaliable but datastreams are still running, continue polling
            pass
        else:
            datapoint = msgs[0]
            timing_msg = msgs[1]
            if isinstance(datapoint, LinearPosPoint):
                self.pos_counter += 1
                for encoder in self.encoders.values():   # type: kernel_encoder.RSTKernelEncoder
                    encoder.update_covariate(datapoint.x)

                if self.pos_counter % 100 == 0:
                    self.class_log.debug('Received {} pos datapoints.'.format(self.pos_counter))
                pass


class EncoderMPIRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, encoder_manager: RStarEncoderManager):
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

        elif isinstance(message, binary_record.BinaryRecordCreateMessage):
            self.enc_man.set_record_writer_from_message(message)

        elif isinstance(message, realtime_base.StartRecordMessage):
            self.enc_man.start_record_writing()

        elif isinstance(message, realtime_base.StopRecordMessage):
            self.enc_man.stop_record_writing()


class EncoderProcess(realtime_base.RealtimeMPIClass, metaclass=realtime_base.ProfilerWrapperMeta):
    def __init__(self, comm: MPI.Comm, rank, config):

        super().__init__(comm, rank, config)

        self.enable_profiler = rank in self.config['rank_settings']['enable_profiler']
        self.profiler_out_path = os.path.join(config['files']['output_dir'], '{}.{:02d}.{}'.
                                              format(config['files']['prefix'],
                                                     rank,
                                                     config['files']['profile_postfix']))

        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
                                                                          manager_rank=config['rank']['supervisor'])

        self.mpi_send = EncoderMPISendInterface(comm=comm, rank=rank, config=config)

        if self.config['datasource'] == 'simulator':
            spike_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                        rank=self.rank,
                                                                        config=self.config,
                                                                        datatype=datatypes.Datatypes.SPIKES)

            pos_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                      rank=self.rank,
                                                                      config=self.config,
                                                                      datatype=datatypes.Datatypes.LINEAR_POSITION)

        self.enc_man = RStarEncoderManager(rank=rank,
                                           local_rec_manager=self.local_rec_manager,
                                           send_interface=self.mpi_send,
                                           spike_interface=spike_interface,
                                           pos_interface=pos_interface)

        self.mpi_recv = EncoderMPIRecvInterface(comm=comm, rank=rank, config=config, encoder_manager=self.enc_man)

        self.terminate = False

    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):

        try:
            while not self.terminate:
                self.mpi_recv.__next__()
                self.enc_man.process_next_data()

        except StopIteration as ex:
            self.class_log.info('Terminating EncodingProcess (rank: {:})'.format(self.rank))

        self.class_log.info("Encoding Process reached end, exiting.")
