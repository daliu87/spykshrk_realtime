from mpi4py import MPI
from spykshrk.realtime import realtime_base, realtime_logging, binary_record, datatypes


class DecoderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm :MPI.Comm, rank, config):
        super(DecoderMPISendInterface, self).__init__(comm=comm, rank=rank, config=config)


class BayesianDecodeManager(realtime_base.BinaryRecordBaseWithTiming):
    def __init__(self, rank, local_rec_manager, send_interface: DecoderMPISendInterface):
        super(BayesianDecodeManager, self).__init__(rank=rank, local_rec_manager=local_rec_manager,
                                                    rec_ids=[realtime_base.RecordIDs.DECODER_OUTPUT],
                                                    rec_labels=[['TBD']],
                                                    rec_format=['i'])

        self.mpi_send = send_interface


class DecoderRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, decode_manager: BayesianDecodeManager):
        super(DecoderRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.dec_man = decode_manager

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


class DecoderProcess(realtime_base.RealtimeProcess):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(DecoderProcess, self).__init__(comm=comm, rank=rank, config=config)

        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
                                                                          manager_rank=config['rank']['supervisor'])

        self.terminate = False

        self.mpi_send = DecoderMPISendInterface(comm=comm, rank=rank, config=config)
        self.dec_man = BayesianDecodeManager(rank=rank, local_rec_manager=self.local_rec_manager,
                                             send_interface=self.mpi_send)
        self.mpi_recv = DecoderRecvInterface(comm=comm, rank=rank, config=config, decode_manager=self.dec_man)

        def trigger_termination(self):
            self.terminate = True

        def main_loop(self):

            try:
                while not self.terminate:
                    self.mpi_recv.__next__()

            except StopIteration as ex:
                self.class_log.info('Terminating DecoderProcess (rank: {:})'.format(self.rank))

            self.class_log.info("Decoding Process reached end, exiting.")
