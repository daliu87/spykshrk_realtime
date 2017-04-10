
import spykshrk.realtime.realtime_process as realtime_process

from mpi4py import MPI


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


class MainProcess(realtime_process.RealtimeProcess):

    def __init__(self, comm: MPI.Comm, rank, ripple_ranks, latency_rank):

        super().__init__(MainThread, comm=comm, rank=rank, ripple_ranks=ripple_ranks,
                         latency_rank=latency_rank)

        # TODO temporary measure to enable type hinting (typing.Generics is broken for PyCharm 2016.2.3)
        self.thread = self.thread   # type: MainThread

    def main_loop(self):
        self.thread.start()

        while True:
            pass


class MainThread(realtime_process.RealtimeThread):

    def __init__(self, parent, comm: MPI.Comm, rank, ripple_ranks, latency_rank):
        super().__init__(parent)

        self._stop_next = False

    def stop_thread_next(self):
        self._stop_next = True

    def run(self):
        while not self._stop_next:
            pass

