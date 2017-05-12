
import spykshrk.realtime.realtime_process as realtime_process
import spykshrk.realtime.simulator.simulator_process as simulator_process

from mpi4py import MPI

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

        self.comm = comm
        self.rank = rank
        self.config = config

        super().__init__(comm=comm, rank=rank, config=config, ThreadClass=MainThread)

        # TODO temporary measure to enable type hinting (typing.Generics is broken for PyCharm 2016.2.3)
        self.thread = self.thread   # type: MainThread

        self.terminate = False

    def main_loop(self):
        self.thread.start()
        mpi_status = MPI.Status()

        while not self.terminate:
            message = self.comm.recv(tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

            if isinstance(message, simulator_process.SimTrodeListMessage):

                for rip_rank in self.config['rank']['ripples']:
                    self.comm.send(realtime_process.NumTrodesMessage(len(message.trode_list)), dest=rip_rank)

                # Round robin allocation of channels to ripple
                enable_count = 0
                all_ripple_process_enable = [[] for _ in self.config['rank']['ripples']]
                for chan_ind, chan_id in enumerate(message.trode_list):
                    all_ripple_process_enable[enable_count % len(self.config['rank']['ripples'])].append(chan_ind)
                    enable_count += 1


class MainThread(realtime_process.RealtimeThread):

    def __init__(self, comm: MPI.Comm, rank, config, parent):
        super().__init__(comm=comm, rank=rank, config=config, parent=parent)
        ripple_ranks = self.config['rank']['ripples']

        self._stop_next = False

    def stop_thread_next(self):
        self._stop_next = True

    def run(self):

        while not self._stop_next:
            pass

