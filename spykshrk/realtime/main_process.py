
import spykshrk.realtime.realtime_process as realtime_process

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

        super().__init__(MainThread, comm=comm, rank=rank, config=config)

        # TODO temporary measure to enable type hinting (typing.Generics is broken for PyCharm 2016.2.3)
        self.thread = self.thread   # type: MainThread

    def main_loop(self):
        self.thread.start()

        while True:
            pass


class MainThread(realtime_process.RealtimeThread):

    def __init__(self, parent, comm: MPI.Comm, rank, config):
        super().__init__(parent)

        self._stop_next = False

    def stop_thread_next(self):
        self._stop_next = True

    def run(self):
        while not self._stop_next:
            pass

