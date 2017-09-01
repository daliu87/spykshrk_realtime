from mpi4py import MPI
import math
import numpy as np
from spykshrk.realtime import realtime_base, realtime_logging, binary_record, datatypes, encoder_process
from spykshrk.realtime.simulator import simulator_process

class DecoderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm :MPI.Comm, rank, config):
        super(DecoderMPISendInterface, self).__init__(comm=comm, rank=rank, config=config)

    def send_record_register_messages(self, record_register_messages):
        self.class_log.debug("Sending record register messages.")
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_time_sync_report(self, time):
        self.comm.send(obj=realtime_base.TimeSyncReport(time),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def all_barrier(self):
        self.comm.Barrier()


class SpikeDecodeRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(SpikeDecodeRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.msg_buffer = bytearray(50000)
        self.req = self.comm.Irecv(buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.SPIKE_DECODE_DATA)

    def __next__(self):
        rdy = self.req.Test()
        if rdy:

            msg = encoder_process.SpikeDecodeResultsMessage.unpack(self.msg_buffer)
            self.req = self.comm.Irecv(buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.SPIKE_DECODE_DATA)
            return msg

        else:
            return None


class PPDecodeManager(realtime_base.BinaryRecordBaseWithTiming):
    def __init__(self, rank, config, local_rec_manager, send_interface: DecoderMPISendInterface,
                 spike_decode_interface: SpikeDecodeRecvInterface,
                 pos_interface: simulator_process.SimulatorRemoteReceiver):
        super(PPDecodeManager, self).__init__(rank=rank,
                                              local_rec_manager=local_rec_manager,
                                              send_interface=send_interface,
                                              rec_ids=[realtime_base.RecordIDs.DECODER_OUTPUT],
                                              rec_labels=[['timestamp'] +
                                                          ['x'+str(x) for x in
                                                           range(config['encoder']['position']['bins'])]],
                                              rec_formats=['q'+'d'*config['encoder']['position']['bins']])

        self.config = config
        self.mpi_send = send_interface
        self.spike_dec_interface = spike_decode_interface
        self.pos_interface = pos_interface

        # Register position, right now only one position channel is supported
        self.pos_interface.register_datatype_channel(-1)

        # Send binary record register message
        # self.mpi_send.send_record_register_messages(self.get_record_register_messages())

        self.msg_counter = 0
        self.ntrode_list = []

        self.current_time_bin = 0
        self.observation = np.ones(self.config['encoder']['position']['bins'])
        self.occ = np.ones(self.config['encoder']['position']['bins'])
        self.posterior = np.ones(self.config['encoder']['position']['bins'])
        self.firing_rate = {}
        self.cur_pos_ind = 0
        self.pos_delta = (self.config['encoder']['position']['upper'] -
                          self.config['encoder']['position']['lower']) / self.config['encoder']['position']['bins']

        self.transition_mat = PPDecodeManager._create_transition_matrix(self.pos_delta,
                                                                        self.config['encoder']['position']['bins'])

        self.current_spike_count = 0

    @staticmethod
    def _create_transition_matrix(pos_delta, num_bins):

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    # Setup transition matrix
        x_bins = np.linspace(0, pos_delta*(num_bins-1), num_bins)

        transition_mat = np.ones([num_bins, num_bins])
        for bin_ii in range(num_bins):
            transition_mat[bin_ii, :] = gaussian(x_bins, x_bins[bin_ii], 3)

        # uniform offset
        uniform_gain = 0.01
        uniform_dist = np.ones(transition_mat.shape)

        # normalize transition matrix
        transition_mat = transition_mat/( transition_mat.sum(axis=0)[None,:])

        # normalize uniform offset
        uniform_dist = uniform_dist/( uniform_dist.sum(axis=0)[None,:])

        # apply uniform offset
        transition_mat = transition_mat * (1 - uniform_gain) + uniform_dist * uniform_gain

        return transition_mat


    def turn_on_datastreams(self):
        self.pos_interface.start_all_streams()

    def select_ntrodes(self, ntrode_list):
        self.ntrode_list = ntrode_list
        self.firing_rate = {ntrode_id: np.ones(self.config['encoder']['position']['bins'])
                            for ntrode_id in self.ntrode_list}

    def process_next_data(self):
        spike_dec_msg = self.spike_dec_interface.__next__()

        if spike_dec_msg is not None:

            self.record_timing(timestamp=spike_dec_msg.timestamp, ntrode_id=spike_dec_msg.ntrode_id,
                               datatype=datatypes.Datatypes.SPIKES, label='dec_recv')

            # Update firing rate
            self.firing_rate[spike_dec_msg.ntrode_id][self.cur_pos_ind] += 1

            if self.current_time_bin == 0:
                self.current_time_bin = math.floor(spike_dec_msg.timestamp/self.config['pp_decoder']['bin_size'])
                spike_time_bin = self.current_time_bin
            else:
                spike_time_bin = math.floor(spike_dec_msg.timestamp/self.config['pp_decoder']['bin_size'])

            if spike_time_bin == self.current_time_bin:
                # Spike is in current time bin
                self.observation *= spike_dec_msg.pos_hist
                self.observation = self.observation / np.max(self.observation)
                self.current_spike_count += 1

            elif spike_time_bin > self.current_time_bin:
                # Spike is in next time bin, advance to tracking next time bin
                self.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
                                  self.current_time_bin * self.config['pp_decoder']['bin_size'],
                                  *self.observation)
                self.current_spike_count = 1
                self.observation = spike_dec_msg.pos_hist
                self.current_time_bin = spike_time_bin

            elif spike_time_bin < self.current_time_bin:
                # Spike is in an old time bin, discard and mark as missed
                self.class_log.debug('Spike was excluded from PP decode calculation, arrived late.')
                pass

            self.msg_counter += 1
            if self.msg_counter % 1000 == 0:
                self.class_log.debug('Received {} decoded messages.'.format(self.msg_counter))

            pass

        pos_msg = self.pos_interface.__next__()

        #if isinstance(pos_msg, datatypes.LinearPosPoint):
        if pos_msg is not None:
            pos_data = pos_msg[0]

            # Convert position to bin index in histogram count
            self.cur_pos_ind = int((pos_data.x - self.config['encoder']['position']['lower']) /
                                   self.pos_delta)
            self.occ[self.cur_pos_ind] += 1
            # self.class_log.debug("Pos msg received.")


class BayesianDecodeManager(realtime_base.BinaryRecordBaseWithTiming):
    def __init__(self, rank, config, local_rec_manager, send_interface: DecoderMPISendInterface,
                 spike_decode_interface: SpikeDecodeRecvInterface):
        super(BayesianDecodeManager, self).__init__(rank=rank,
                                                    local_rec_manager=local_rec_manager,
                                                    send_interface=send_interface,
                                                    rec_ids=[realtime_base.RecordIDs.DECODER_OUTPUT],
                                                    rec_labels=[['timestamp'] +
                                                                ['x'+str(x) for x in
                                                                range(config['encoder']['position']['bins'])]],
                                                    rec_formats=['q'+'d'*config['encoder']['position']['bins']])

        self.config = config
        self.mpi_send = send_interface
        self.spike_interface = spike_decode_interface

        # Send binary record register message
        # self.mpi_send.send_record_register_messages(self.get_record_register_messages())

        self.msg_counter = 0

        self.current_time_bin = 0
        self.current_est_pos_hist = np.ones(self.config['encoder']['position']['bins'])
        self.current_spike_count = 0
        self.ntrode_list = []

    def turn_on_datastreams(self):
        # Do nothing, no datastreams for this decoder
        pass

    def select_ntrodes(self, ntrode_list):
        self.ntrode_list = ntrode_list

    def process_next_data(self):
        spike_dec_msg = self.spike_interface.__next__()

        if spike_dec_msg is not None:

            self.record_timing(timestamp=spike_dec_msg.timestamp, ntrode_id=spike_dec_msg.ntrode_id,
                               datatype=datatypes.Datatypes.SPIKES, label='dec_recv')

            if self.current_time_bin == 0:
                self.current_time_bin = math.floor(spike_dec_msg.timestamp/self.config['bayesian_decoder']['bin_size'])
                spike_time_bin = self.current_time_bin
            else:
                spike_time_bin = math.floor(spike_dec_msg.timestamp/self.config['bayesian_decoder']['bin_size'])

            if spike_time_bin == self.current_time_bin:
                # Spike is in current time bin
                self.current_est_pos_hist *= spike_dec_msg.pos_hist
                self.current_est_pos_hist = self.current_est_pos_hist / np.max(self.current_est_pos_hist)
                self.current_spike_count += 1

            elif spike_time_bin > self.current_time_bin:
                # Spike is in next time bin, advance to tracking next time bin
                self.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
                                  self.current_time_bin*self.config['bayesian_decoder']['bin_size'],
                                  *self.current_est_pos_hist)
                self.current_spike_count = 1
                self.current_est_pos_hist = spike_dec_msg.pos_hist
                self.current_time_bin = spike_time_bin

            elif spike_time_bin < self.current_time_bin:
                # Spike is in an old time bin, discard and mark as missed
                self.class_log.debug('Spike was excluded from Bayesian decode calculation, arrived late.')
                pass

            self.msg_counter += 1
            if self.msg_counter % 1000 == 0:
                self.class_log.debug('Received {} decoded messages.'.format(self.msg_counter))

            pass
            # self.class_log.debug(spike_dec_msg)


class DecoderRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, decode_manager: BayesianDecodeManager):
        super(DecoderRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.dec_man = decode_manager

        self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def __next__(self):
        rdy, msg = self.req.test()
        if rdy:
            self.process_request_message(msg)

            self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def process_request_message(self, message):
        if isinstance(message, realtime_base.TerminateMessage):
            self.class_log.debug("Received TerminateMessage")
            raise StopIteration()

        elif isinstance(message, binary_record.BinaryRecordCreateMessage):
            self.dec_man.set_record_writer_from_message(message)

        elif isinstance(message, realtime_base.ChannelSelection):
            self.class_log.debug("Received NTrode channel selection {:}.".format(message.ntrode_list))
            self.dec_man.select_ntrodes(message.ntrode_list)

        elif isinstance(message, realtime_base.TimeSyncInit):
            self.class_log.debug("Received TimeSyncInit.")
            self.dec_man.sync_time()

        elif isinstance(message, realtime_base.TurnOnDataStream):
            self.class_log.debug("Turn on data stream")
            self.dec_man.turn_on_datastreams()

        elif isinstance(message, realtime_base.TimeSyncSetOffset):
            self.dec_man.update_offset(message.offset_time)

        elif isinstance(message, realtime_base.StartRecordMessage):
            self.dec_man.start_record_writing()

        elif isinstance(message, realtime_base.StopRecordMessage):
            self.dec_man.stop_record_writing()


class DecoderProcess(realtime_base.RealtimeProcess):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(DecoderProcess, self).__init__(comm=comm, rank=rank, config=config)

        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
                                                                          manager_rank=config['rank']['supervisor'])

        self.terminate = False

        self.mpi_send = DecoderMPISendInterface(comm=comm, rank=rank, config=config)
        self.spike_decode_interface = SpikeDecodeRecvInterface(comm=comm, rank=rank, config=config)
        self.pos_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                       rank=self.rank,
                                                                       config=self.config,
                                                                       datatype=datatypes.Datatypes.LINEAR_POSITION)

        if config['decoder'] == 'bayesian_decoder':
            self.dec_man = BayesianDecodeManager(rank=rank, config=config,
                                                 local_rec_manager=self.local_rec_manager,
                                                 send_interface=self.mpi_send,
                                                 spike_decode_interface=self.spike_decode_interface)
        elif config['decoder'] == 'pp_decoder':
            self.dec_man = PPDecodeManager(rank=rank, config=config,
                                           local_rec_manager=self.local_rec_manager,
                                           send_interface=self.mpi_send,
                                           spike_decode_interface=self.spike_decode_interface,
                                           pos_interface=self.pos_interface)

        self.mpi_recv = DecoderRecvInterface(comm=comm, rank=rank, config=config, decode_manager=self.dec_man)

        # First Barrier to finish setting up nodes
        self.comm.Barrier()

    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):

        try:
            while not self.terminate:
                self.dec_man.process_next_data()
                self.mpi_recv.__next__()

        except StopIteration as ex:
            self.class_log.info('Terminating DecoderProcess (rank: {:})'.format(self.rank))

        self.class_log.info("Decoding Process reached end, exiting.")
