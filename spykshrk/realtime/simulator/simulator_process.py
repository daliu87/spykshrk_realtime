import sys

from mpi4py import MPI
import threading

import spykshrk.realtime.realtime_process as realtime_process
import spykshrk.realtime.datatypes as datatypes
import spykshrk.realtime.simulator.nspike_data as nspike_data
import spykshrk.realtime.simulator.sim_databuffer as sim_databuffer


class ReqLFPChannelData(realtime_process.RealtimeMessage):
    def __init__(self, lfp_chan):
        self.lfp_chan = lfp_chan


class SimNumTrodesMessage(realtime_process.RealtimeMessage):
    def __init__(self, num_ntrodes):
        self.num_ntrodes = num_ntrodes


class SimulatorProcess(realtime_process.RealtimeProcess):
    def __init__(self, comm: MPI.Comm, rank, config):
        super().__init__(comm=comm, rank=rank, config=config, ThreadClass=SimulatorThread)
        self.terminate = False

    def main_loop(self):
        self.thread.start()

        mpi_status = MPI.Status()
        while not self.terminate:
            message = self.comm.recv(status=mpi_status)
            if isinstance(message, ReqLFPChannelData):
                self.thread.update_lfp_chan_req(mpi_status.source, message.lfp_chan)


class SimulatorThread(realtime_process.RealtimeThread):

    def __init__(self, comm: MPI.Comm, rank, config, parent):
        super().__init__(comm=comm, rank=rank, config=config, parent=parent)

        self._stop_next = False

        try:
            nspike_anim = nspike_data.AnimalInfo(**config['simulator']['nspike_animal_info'])
            lfp_stream = nspike_data.EEGDataStream(nspike_anim, 100)
            pos_stream = nspike_data.PosMatDataStream(nspike_anim, 1000)
            self.databuffer = sim_databuffer.SimDataBuffer([lfp_stream(), pos_stream()])

            self.lfp_chan_req_dict = {}
        except TypeError as err:
            self.class_log.exception("TypeError: nspike_animal_info does not match nspike_data.AnimalInfo arguments.",
                                     exc_info=err)
            comm.send(realtime_process.TerminateErrorMessage("For SimulatorThread, nspike_animal_info config did"
                                                             "not match nspike_data.AnimalInfo arguments."),
                      config['rank']['supervisor'])

        self.comm.send(obj=SimNumTrodesMessage(len(self.config['simulator']
                                                   ['nspike_animal_info']['tetrodes'])),
                       dest=config['rank']['supervisor'],
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

        self.start_datastream = threading.Event()

    def stop_thread_next(self):
        self._stop_next = True

    def update_lfp_chan_req(self, dest_rank, lfp_chan):
        if lfp_chan in self.lfp_chan_req_dict:
            self.class_log.error(("LFP channels cannot be requested by more than one rank. Channel ({:}) requested by "
                                  "rank ({:}) but is already owned by rank ({:}). "
                                  "Overwriting previous assignment.").format(lfp_chan, dest_rank,
                                                                             self.lfp_chan_req_dict[lfp_chan]))
        self.lfp_chan_req_dict[lfp_chan] = dest_rank

    def start_datastream(self):
        self.start_datastream.set()

    def run(self):
        data_itr = self.databuffer()
        self.start_datastream.wait()
        while not self._stop_next:
            data_to_send = data_itr.__next__()
            if isinstance(data_to_send, datatypes.LFPPoint):
                try:
                    self.comm.send(obj=data_to_send, dest=self.lfp_chan_req_dict[data_to_send.ntrode_index])
                except KeyError as err:
                    self.class_log.exception(("KeyError: Tetrode index ({:}) not in lfp channel request dict,"
                                              "was likely never requested by a receiving/computing ranks.").
                                             format(data_to_send.ntrode_index), exc_info=err)

            self.start_datastream.wait()

