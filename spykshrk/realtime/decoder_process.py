from mpi4py import MPI
import math
import numpy as np
from spykshrk.realtime import realtime_base, realtime_logging, binary_record, datatypes, encoder_process


class DecoderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm :MPI.Comm, rank, config):
        super(DecoderMPISendInterface, self).__init__(comm=comm, rank=rank, config=config)

    def send_record_register_messages(self, record_register_messages):
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)


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


class BayesianDecodeManager(realtime_base.BinaryRecordBaseWithTiming):
    def __init__(self, rank, config, offset_time, local_rec_manager, send_interface: DecoderMPISendInterface,
                 spike_decode_interface: SpikeDecodeRecvInterface):
        super(BayesianDecodeManager, self).__init__(rank=rank, offset_time=offset_time,
                                                    local_rec_manager=local_rec_manager,
                                                    rec_ids=[realtime_base.RecordIDs.DECODER_OUTPUT],
                                                    rec_labels=[['timestamp'] +
                                                                ['x'+str(x) for x in
                                                                range(config['encoder']['position']['bins'])]],
                                                    rec_formats=['q'+'d'*config['encoder']['position']['bins']])

        self.config = config
        self.mpi_send = send_interface
        self.spike_interface = spike_decode_interface

        # Send binary record register message
        self.mpi_send.send_record_register_messages(self.get_record_register_messages())
        self.msg_counter = 0

        self.current_time_bin = 0
        self.current_est_pos_hist = np.ones(self.config['encoder']['position']['bins'])
        self.current_spike_count = 0

    def process_next_data(self):
        spike_dec_msg = self.spike_interface.__next__()

        if spike_dec_msg is not None:

            self.record_timing(timestamp=spike_dec_msg.timestamp, ntrode_id=spike_dec_msg.ntrode_id,
                               datatype=datatypes.Datatypes.SPIKES, label='dec_recv')

            if self.current_time_bin == 0:
                self.current_time_bin = math.floor(spike_dec_msg.timestamp/self.config['decoder']['bin_size'])
                spike_time_bin = self.current_time_bin
            else:
                spike_time_bin = math.floor(spike_dec_msg.timestamp/self.config['decoder']['bin_size'])

            if spike_time_bin == self.current_time_bin:
                # Spike is in current time bin
                self.current_est_pos_hist *= spike_dec_msg.pos_hist
                self.current_est_pos_hist = self.current_est_pos_hist / np.max(self.current_est_pos_hist)
                self.current_spike_count += 1

            elif spike_time_bin > self.current_time_bin:
                # Spike is in next time bin, advance to tracking next time bin
                self.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
                                  self.current_time_bin*self.config['decoder']['bin_size'],
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
        self.dec_man = BayesianDecodeManager(rank=rank, config=config,
                                             offset_time=self.offset_time,
                                             local_rec_manager=self.local_rec_manager,
                                             send_interface=self.mpi_send,
                                             spike_decode_interface=self.spike_decode_interface)
        self.mpi_recv = DecoderRecvInterface(comm=comm, rank=rank, config=config, decode_manager=self.dec_man)

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
