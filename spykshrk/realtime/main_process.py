
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


class MainProcess(realtime_process.RealtimeProcess):

    def __init__(self, comm: MPI.Comm, rank, config):

        self.comm = comm    # type: MPI.Comm
        self.rank = rank
        self.config = config

        self.send_interface = MainMPISendInterface(comm=comm, rank=rank, config=config)
        self.manager = MainSimulatorManager(rank=rank, config=config, parent=self, send_interface=self.send_interface)
        self.recv_interface = MainSimulatorMPIRecvInterface(comm=comm, rank=rank,
                                                            config=config, main_manager=self.manager)

        super().__init__(comm=comm, rank=rank, config=config, ThreadClass=MainThread)

        # TODO temporary measure to enable type hinting (typing.Generics is broken for PyCharm 2016.2.3)
        self.thread = self.thread   # type: MainThread

        self.terminate = False

    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):
        self.thread.start()

        while not self.terminate:
            self.recv_interface.process_next_message()

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


class MainMPISendInterface(realtime_process.RealtimeClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        self.comm = comm
        self.rank = rank
        self.config = config

        super().__init__()

    def send_num_ntrode(self, rank, num_ntrodes):
        self.class_log.debug("Sending number of ntrodes to rank {:}".format(rank))
        self.comm.send(realtime_process.NumTrodesMessage(num_ntrodes), dest=rank,
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_channel_selection(self, rank, channel_selects):
        self.comm.send(obj=ripple_process.ChannelSelection(channel_selects), dest=rank,
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_new_writer_message(self, rank, new_writer_message):
        self.comm.send(obj=new_writer_message, dest=rank,
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_start_rec_message(self, rank):
        self.comm.send(obj=realtime_process.StartRecordMessage(), dest=rank,
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_turn_on_datastreams(self, rank):
        self.comm.send(obj=ripple_process.TurnOnDataStream(), dest=rank,
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_ripple_parameter(self, rank, param_message):
        self.comm.send(obj=param_message, dest=rank, tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_ripple_baseline_mean(self, rank, mean_dict):
        self.comm.send(obj=ripple_process.CustomRippleBaselineMeanMessage(mean_dict=mean_dict), dest=rank,
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_ripple_baseline_std(self, rank, std_dict):
        self.comm.send(obj=ripple_process.CustomRippleBaselineStdMessage(std_dict=std_dict), dest=rank,
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def terminate_all(self):
        terminate_ranks = list(range(self.comm.size))
        terminate_ranks.remove(self.rank)
        for rank in terminate_ranks:
            self.comm.send(obj=realtime_process.TerminateMessage(), dest=rank,
                           tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)


class MainSimulatorManager(realtime_process.RealtimeClass):

    def __init__(self, rank, config, parent: MainProcess, send_interface: MainMPISendInterface):

        self.rank = rank
        self.config = config
        self.parent = parent
        self.send_interface = send_interface

        self.rec_manager = binary_record.BinaryRecordsManager(manager_label='realtime_replay',
                                                              save_dir=self.config['files']['output_dir'],
                                                              file_prefix=self.config['files']['prefix'],
                                                              file_postfix=self.config['files']['rec_postfix'])

        super().__init__()

        for rip_rank in self.config['rank']['ripples']:

            # Map json RippleParameterMessage onto python object and then send
            rip_param_message = ripple_process.RippleParameterMessage(**self.config['ripple']['RippleParameterMessage'])
            self.send_interface.send_ripple_parameter(rank=rip_rank, param_message=rip_param_message)

            # Convert json string keys into int (ntrode_id) and send
            rip_mean_base_dict = dict(map(lambda x: (int(x[0]), x[1]),
                                          self.config['ripple']['CustomRippleBaselineMeanMessage'].items()))
            self.send_interface.send_ripple_baseline_mean(rank=rip_rank, mean_dict=rip_mean_base_dict)

            # Convert json string keys into int (ntrode_id) and send
            rip_std_base_dict = dict(map(lambda x: (int(x[0]), x[1]),
                                         self.config['ripple']['CustomRippleBaselineStdMessage'].items()))
            self.send_interface.send_ripple_baseline_std(rank=rip_rank, std_dict=rip_std_base_dict)

    def handle_ntrode_list(self, trode_list):
        self.class_log.debug("Received ntrode list from simulator {:}.".format(trode_list))
        for rip_rank in self.config['rank']['ripples']:
            self.send_interface.send_num_ntrode(rank=rip_rank, num_ntrodes=len(trode_list))

        # Round robin allocation of channels to ripple
        enable_count = 0
        all_ripple_process_enable = [[] for _ in self.config['rank']['ripples']]
        for chan_ind, chan_id in enumerate(trode_list):
            all_ripple_process_enable[enable_count % len(self.config['rank']['ripples'])].append(chan_id)
            enable_count += 1

        # Set channel assignments for all ripple ranks
        for rank_ind, rank in enumerate(self.config['rank']['ripples']):
            self.send_interface.send_channel_selection(rank, all_ripple_process_enable[rank_ind])

        # Update binary_record file writers before starting datastream
        for rec_rank in self.config['rank_settings']['enable_rec']:
            self.send_interface.send_new_writer_message(rank=rec_rank,
                                                        new_writer_message=self.rec_manager.new_writer_message())

            self.send_interface.send_start_rec_message(rank=rec_rank)

        sleep(0.5)
        # Then turn on data streaming to ripple ranks
        for rank in self.config['rank']['ripples']:
            self.send_interface.send_turn_on_datastreams(rank)

    def register_rec_type_message(self, message):
        self.rec_manager.register_rec_type_message(message)

    def trigger_termination(self):
        self.send_interface.terminate_all()

        self.parent.thread.trigger_termination()
        self.parent.trigger_termination()


class MainSimulatorMPIRecvInterface(realtime_process.RealtimeClass):

    def __init__(self, comm: MPI.Comm, rank, config, main_manager: MainSimulatorManager):
        self.comm = comm
        self.rank = rank
        self.config = config
        self.main_manager = main_manager

        self.mpi_status = MPI.Status()

        super().__init__()

    def process_next_message(self):
        message = self.comm.recv(status=self.mpi_status, tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

        if isinstance(message, simulator_process.SimTrodeListMessage):
            self.main_manager.handle_ntrode_list(message.trode_list)

        elif isinstance(message, binary_record.BinaryRecordTypeMessage):
            self.main_manager.register_rec_type_message(message)

        elif isinstance(message, realtime_process.TerminateMessage):
            self.class_log.info('Received TerminateMessage from rank {:}, now terminating all.'.
                                format(self.mpi_status.source))

            self.main_manager.trigger_termination()


