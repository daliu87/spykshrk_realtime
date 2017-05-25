
import spykshrk.realtime.realtime_process as realtime_process
import spykshrk.realtime.simulator.simulator_process as simulator_process
import spykshrk.realtime.ripple_process as ripple_process
import spykshrk.realtime.binary_record as binary_record

from mpi4py import MPI
from time import sleep

import sys

# try:
#     __IPYTHON__
#     from IPython.terminal.debugger import TerminalPdb
#     bp = TerminalPdb(color_scheme='linux').set_trace
# except NameError as err:
#     print('Warning: NameError ({}), not using ipython (__IPYTHON__ not set), disabling IPython TerminalPdb.'.
#           format(err))
#     bp = lambda: None
# except AttributeError as err:
#     print('Warning: Attribute Error ({}), disabling IPython TerminalPdb.'.format(err))
#     bp = lambda: None


class StimDecider(realtime_process.RealtimeClass):
    def __init__(self, send_manager, ripple_n_above_thresh=sys.maxsize):
        super().__init__()
        self._send_manager = send_manager
        self._ripple_n_above_thresh = ripple_n_above_thresh
        self._ripple_thresh_states = {}
        self._enabled = False

    def reset(self):
        self._ripple_thresh_states = {}

    def enable(self):
        self.class_log.info('Enabled stim decider.')
        self._enabled = True
        self.reset()

    def disable(self):
        self.class_log.info('Disable stim decider.')
        self._enabled = False
        self.reset()

    def update_n_threshold(self, ripple_n_above_thresh):
        self._ripple_n_above_thresh = ripple_n_above_thresh

    def update_ripple_threshold_state(self, timestamp, ntrode_index, threshold_state):
        if self._enabled:
            self._ripple_thresh_states[ntrode_index] = threshold_state
            num_above = 0
            for state in self._ripple_thresh_states.values():
                num_above += state

            if num_above >= self._ripple_n_above_thresh:
                self._send_manager.start_stimulation()


class MainProcess(realtime_process.RealtimeProcess):

    def __init__(self, comm: MPI.Comm, rank, config):

        self.comm = comm    # type: MPI.Comm
        self.rank = rank
        self.config = config

        super().__init__(comm=comm, rank=rank, config=config, ThreadClass=MainThread)

        # TODO temporary measure to enable type hinting (typing.Generics is broken for PyCharm 2016.2.3)
        self.thread = self.thread   # type: MainThread

        self.terminate = False

        self.rec_manager = binary_record.BinaryRecordsManager(manager_label='realtime_replay',
                                                              save_dir=self.config['files']['output_dir'],
                                                              file_prefix=self.config['files']['prefix'],
                                                              file_postfix=self.config['files']['rec_postfix'])

    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):
        self.thread.start()
        mpi_status = MPI.Status()

        # Send ripple config to ripple ranks
        rip_param_message = ripple_process.RippleParameterMessage(**self.config['ripple']['RippleParameterMessage'])
        rip_baseline_mean_message = ripple_process. \
            CustomRippleBaselineMeanMessage(dict(map(lambda x: (int(x[0]), x[1]),
                                                     self.config['ripple']['CustomRippleBaselineMeanMessage'].items())))
        rip_baseline_std_message = ripple_process. \
            CustomRippleBaselineStdMessage(dict(map(lambda x: (int(x[0]), x[1]),
                                                    self.config['ripple']['CustomRippleBaselineStdMessage'].items())))

        for rip_rank in self.config['rank']['ripples']:
            self.comm.send(obj=rip_param_message, dest=rip_rank,
                           tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)
            self.comm.send(obj=rip_baseline_mean_message, dest=rip_rank,
                           tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)
            self.comm.send(obj=rip_baseline_std_message, dest=rip_rank,
                           tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

        mpi_status = MPI.Status()

        while not self.terminate:
            message = self.comm.recv(status=mpi_status, tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

            if isinstance(message, simulator_process.SimTrodeListMessage):
                self.class_log.debug("Received ntrode list from simulator {:}.".format(message.trode_list))
                for rip_rank in self.config['rank']['ripples']:
                    self.class_log.debug("Sending number of ntrodes to ripple rank {:}".format(rip_rank))
                    self.comm.send(realtime_process.NumTrodesMessage(len(message.trode_list)), dest=rip_rank,
                                   tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

                # Round robin allocation of channels to ripple
                enable_count = 0
                all_ripple_process_enable = [[] for _ in self.config['rank']['ripples']]
                for chan_ind, chan_id in enumerate(message.trode_list):
                    all_ripple_process_enable[enable_count % len(self.config['rank']['ripples'])].append(chan_id)
                    enable_count += 1

                # Set channel assignments for all ripple ranks
                for rank_ind, rank in enumerate(self.config['rank']['ripples']):
                    self.comm.send(obj=ripple_process.ChannelSelection(all_ripple_process_enable[rank_ind]), dest=rank,
                                   tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

                # Update binary_record file writers before starting datastream
                for rec_rank in self.config['rank_settings']['enable_rec']:
                    self.comm.send(obj=self.rec_manager.new_writer_message(), dest=rec_rank,
                                   tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)
                    self.comm.send(obj=realtime_process.StartRecordMessage(), dest=rec_rank,
                                   tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

                sleep(0.5)
                # Then turn on data streaming to ripple ranks
                for rank_ind, rank in enumerate(self.config['rank']['ripples']):
                    self.comm.send(obj=ripple_process.TurnOnDataStream(), dest=rank,
                                   tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

            elif isinstance(message, binary_record.BinaryRecordTypeMessage):
                self.rec_manager.register_rec_type_message(message)

            elif isinstance(message, realtime_process.TerminateMessage):
                self.class_log.info('Received TerminateMessage from rank {:}, now terminating all.'.
                                    format(mpi_status.source))

                terminate_ranks = list(range(self.comm.size))
                terminate_ranks.remove(self.rank)
                for rank in terminate_ranks:
                    self.comm.send(obj=realtime_process.TerminateMessage(), dest=rank,
                                   tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

                self.thread.trigger_termination()
                self.trigger_termination()

        self.class_log.info("Main Process Main reached end, exiting.")


class MainThread(realtime_process.RealtimeThread):

    def __init__(self, comm: MPI.Comm, rank, config, parent):
        super().__init__(comm=comm, rank=rank, config=config, parent=parent)
        ripple_ranks = self.config['rank']['ripples']

        self._stop_next = False

    def trigger_termination(self):
        self._stop_next = True

    def run(self):

        while not self._stop_next:
            pass

        self.class_log.info("Main Process Thread reached end, exiting.")

