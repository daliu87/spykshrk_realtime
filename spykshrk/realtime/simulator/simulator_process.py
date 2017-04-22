import sys

import spykshrk.realtime.realtime_process as realtime_process
import spykshrk.realtime.simulator.nspike_data as nspike_data

from mpi4py import MPI


class SimulatorProcess(realtime_process.RealtimeProcess):
    def __init__(self, comm: MPI.Comm, rank, config):
        super().__init__(comm=comm, rank=rank, config=config, ThreadClass=SimulatorThread)

    def main_loop(self):
        self.thread.start()


class SimulatorThread(realtime_process.RealtimeThread):

    def __init__(self, comm: MPI.Comm, rank, config, parent):
        super().__init__(comm=comm, rank=rank, config=config, parent=parent)

        self._stop_next = False

        try:
            nspike_anim = nspike_data.AnimalInfo(**config['simulator']['nspike_animal_info'])
        except TypeError as err:
            self.class_log.exception("TypeError: nspike_animal_info does not match nspike_data.AnimalInfo arguments.",
                                     exc_info=err)
            comm.send(realtime_process.TerminateOnErrorMessage("For SimulatorThread, nspike_animal_info config did"
                                                               "not match nspike_data.AnimalInfo arguments."),
                      config['rank']['supervisor'])

        nspike_data.EEGDataStream(nspike_anim, 100)
        nspike_data.PosMatDataStream(nspike_anim, 1000)

    def stop_thread_next(self):
        self._stop_next = True

    def run(self):
        while not self._stop_next:
            pass
