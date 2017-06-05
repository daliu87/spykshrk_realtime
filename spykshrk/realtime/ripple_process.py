import struct
from collections import deque
from collections import OrderedDict

import spykshrk.realtime.realtime_process as realtime_process
import spykshrk.realtime.simulator.simulator_process as simulator_process
from spykshrk.realtime.datatypes import LFPPoint
from mpi4py import MPI
import spykshrk.realtime.binary_record as binary_record
import spykshrk.realtime.datatypes as datatypes


class ChannelSelection(realtime_process.RealtimeMessage):
    def __init__(self, ntrode_list):
        self.ntrode_list = ntrode_list


class RippleParameterMessage(realtime_process.RealtimeMessage):
    def __init__(self, rip_coeff1=1.2, rip_coeff2=0.2, ripple_threshold=5, samp_divisor=10000, n_above_thresh=1,
                 lockout_time=7500, detect_no_ripple_time=60000, dio_gate_port=None, detect_no_ripples=False,
                 dio_gate=False, enabled=False, use_custom_baseline=False, update_custom_baseline=False):
        self.rip_coeff1 = rip_coeff1
        self.rip_coeff2 = rip_coeff2
        self.ripple_threshold = ripple_threshold
        self.samp_divisor = samp_divisor
        self.n_above_thresh = n_above_thresh
        self.lockout_time = lockout_time
        self.detect_no_ripple_time = detect_no_ripple_time
        self.dio_gate_port = dio_gate_port
        self.detect_no_ripples = detect_no_ripples
        self.dio_gate = dio_gate
        self.enabled = enabled
        self.use_custom_baseline = use_custom_baseline
        self.update_custom_baseline = update_custom_baseline


class TurnOnDataStream(realtime_process.RealtimeMessage):
    def __init__(self):
        pass


class CustomRippleBaselineMeanMessage(realtime_process.RealtimeMessage):
    def __init__(self, mean_dict):
        self.mean_dict = mean_dict


class CustomRippleBaselineStdMessage(realtime_process.RealtimeMessage):
    def __init__(self, std_dict):
        self.std_dict = std_dict


class RippleStatusDictListMessage(realtime_process.RealtimeMessage):
    def __init__(self, ripple_rank, status_dict_list):
        self.ripple_rank = ripple_rank
        self.status_dict_list = status_dict_list


class RippleThresholdState(realtime_process.RealtimeMessage):
    """"Message containing whether or not at a given timestamp a ntrode's ripple filter threshold is crossed.

    This message has helper serializer/deserializer functions to be used to speed transmission.
    """
    _byte_format = 'Iii'

    def __init__(self, timestamp, ntrode_id, threshold_state):
        self.timestamp = timestamp
        self.ntrode_id = ntrode_id
        self.threshold_state = threshold_state

    def pack(self):
        return struct.pack(self._byte_format, self.timestamp, self.ntrode_id, self.threshold_state)

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, ntrode_id, threshold_state = struct.unpack(cls._byte_format, message_bytes)
        return cls(timestamp=timestamp, ntrode_id=ntrode_id, threshold_state=threshold_state)


class RippleFilter(realtime_process.RealtimeClass):
    def __init__(self, rec_base: realtime_process.BinaryRecordBase, param: RippleParameterMessage,
                 ntrode_id):
        super().__init__()
        self.rec_base = rec_base
        self.NFILT = 19
        self.NLAST_VALS = 20
        self.NUMERATOR = [2.435723358568172431e-02,
                          -1.229133831328424326e-01,
                          2.832924715801946602e-01,
                          -4.629092463232863941e-01,
                          6.834398182647745124e-01,
                          -8.526143367711925825e-01,
                          8.137704425816699727e-01,
                          -6.516133270563613245e-01,
                          4.138371933419512372e-01,
                          2.165520280363200556e-14,
                          -4.138371933419890403e-01,
                          6.516133270563868596e-01,
                          -8.137704425816841836e-01,
                          8.526143367711996879e-01,
                          -6.834398182647782871e-01,
                          4.629092463232882815e-01,
                          -2.832924715801954929e-01,
                          1.229133831328426407e-01,
                          -2.435723358568174512e-02]

        self.DENOMINATOR = [1.000000000000000000e+00,
                            -7.449887056735371438e+00,
                            2.866742370538527496e+01,
                            -7.644272470167831557e+01,
                            1.585893197862293391e+02,
                            -2.703338821178639932e+02,
                            3.898186201116285474e+02,
                            -4.840217978093359079e+02,
                            5.230782138295531922e+02,
                            -4.945387299274730140e+02,
                            4.094389697124813665e+02,
                            -2.960738943482194827e+02,
                            1.857150345772943751e+02,
                            -9.980204002570326338e+01,
                            4.505294594295533273e+01,
                            -1.655156422615593215e+01,
                            4.683913633549676270e+00,
                            -9.165841559639211766e-01,
                            9.461443242601841330e-02]

        self.ntrode_id = ntrode_id
        self.param = param

        self.stim_enabled = False

        self._custom_baseline_mean = 0.0
        self._custom_baseline_std = 0.0

        self.pos_gain = 0.0
        self.enabled = 0  # true if this Ntrode is enabled
        self.ripple_mean = 0.0
        self.ripple_std = 0.0
        self.f_x = [0.0] * self.NFILT
        self.f_y = [0.0] * self.NFILT
        self.filtind = 0
        self.last_val = deque([0.0] * self.NLAST_VALS)
        self.current_val = 0.0
        self.current_thresh = 0.0

        self.current_time = 0
        self.last_stim_time = 0
        self.in_lockout = False
        self.thresh_crossed = False

    @property
    def custom_baseline_mean(self):
        return self._custom_baseline_mean

    @custom_baseline_mean.setter
    def custom_baseline_mean(self, value):
        if value:
            self._custom_baseline_mean = value
        else:
            pass

    @property
    def custom_baseline_std(self):
        return self._custom_baseline_mean

    @custom_baseline_std.setter
    def custom_baseline_std(self, value):
        if value:
            self._custom_baseline_std = value
        else:
            pass

    def reset_data(self):
        self.class_log.debug('Reset data')
        self.pos_gain = 0.0
        self.enabled = 0  # true if this Ntrode is enabled
        self.ripple_mean = 0.0
        self.ripple_std = 0.0
        self.f_x = [0.0] * self.NFILT
        self.f_y = [0.0] * self.NFILT
        self.filtind = 0
        self.last_val = deque([0.0] * self.NLAST_VALS)
        self.current_val = 0.0
        self.current_thresh = 0.0

        self.current_time = 0
        self.last_stim_time = 0
        self.in_lockout = False
        self.thresh_crossed = False

    def update_parameter(self, param: RippleParameterMessage):
        self.param = param

    def enable_stimulation(self):
        self.stim_enabled = True

    def disable_stimulation(self):
        self.stim_enabled = False

    def set_stim_time(self, stim_time):
        self.last_stim_time = stim_time

    def update_filter(self, d):
        # return the results of filtering the current value and update the filter values
        val = 0.0
        self.f_x.pop()
        self.f_x.insert(0, d)
        self.f_y.pop()
        self.f_y.insert(0, 0.0)
        # apply the IIR filter this should be done with a dot product eventually
        for i in range(self.NFILT):
            # jind = (crf.filtind + i) % NFILT
            val = val + self.f_x[i] * self.NUMERATOR[i] - self.f_y[i] * self.DENOMINATOR[i]
        self.f_y[0] = val
        return val

    def update_envelop(self, d):
        # return the new gain for positive increments based on the gains from the last 20 points
        # mn = np.mean(crf.lastVal)
        mn = sum(self.last_val) / self.NLAST_VALS
        self.last_val.popleft()
        self.last_val.append(d)
        return mn

    def process_data(self, data_point: LFPPoint):

        self.current_time = data_point.timestamp

        if self.current_time - self.last_stim_time < self.param.lockout_time:
            self.in_lockout = True
        else:
            self.in_lockout = False

        if self.in_lockout:
            rd = self.update_filter(((self.current_time - self.last_stim_time) / self.param.lockout_time)
                                    * data_point.data)
            self.current_val = self.ripple_mean
            self.thresh_crossed = False

        else:

            rd = self.update_filter(data_point.data)

            y = abs(rd)

            if not self.stim_enabled:
                self.ripple_mean += (y - self.ripple_mean) / self.param.samp_divisor
                self.ripple_std += (abs(y - self.ripple_mean) - self.ripple_std) / self.param.samp_divisor
                if not self.param.use_custom_baseline:  # only update the threshold if we're not using a custom baseline
                    self.current_thresh = self.ripple_mean + self.ripple_std * self.param.ripple_threshold
                    # print('ntrode', crf.nTrodeId, 'mean', crf.rippleMean)

            if self.current_time % 30000 == 0:
                self.class_log.info((self.stim_enabled, self.ripple_mean, self.ripple_std))

            # track the rising and falling of the signal
            df = y - self.current_val
            if df > 0:
                gain = self.param.rip_coeff1
                self.pos_gain = self.update_envelop(gain)
                self.current_val += df * self.pos_gain
            else:
                gain = self.param.rip_coeff2
                self.pos_gain = self.update_envelop(gain)
                self.current_val += df * gain

            if self.param.use_custom_baseline:
                if self.current_val >= (self.custom_baseline_mean + self.custom_baseline_std * self.param.ripple_threshold):
                    self.thresh_crossed = True
                else:
                    self.thresh_crossed = False
            else:
                if self.current_val >= self.current_thresh:
                    self.thresh_crossed = True
                else:
                    self.thresh_crossed = False

        # rec_labels=['current_time', 'ntrode_index', 'thresh_crossed', 'lockout', 'lfp_data', 'rd','current_val'],
        # rec_format='Ii??dd',
        self.rec_base.write_record(self.current_time, self.ntrode_id, self.thresh_crossed,
                                   self.in_lockout, int(data_point.data), rd, self.current_val)

        return self.thresh_crossed

    def get_status_dict(self):
        s = OrderedDict()
        if self.param.enabled:
            s['nt'] = self.ntrode_index

            if self.param.use_custom_baseline:
                s['custom_mean'] = self.custom_baseline_mean
                s['custom_std'] = self.custom_baseline_std
            else:
                s['mean'] = self.ripple_mean
                s['std'] = self.ripple_std
        return s


class RippleMPISendInterface(realtime_process.RealtimeClass):

    def __init__(self, comm: MPI.Comm, rank, config):
        super().__init__()

        self.comm = comm
        self.rank = rank
        self.config = config
        self.num_ntrodes = None

    def send_record_register_message(self, record_register_message):
        self.comm.send(obj=record_register_message, dest=self.config['rank']['supervisor'],
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_ripple_status_message(self, status_dict_list):
        if len(status_dict_list) == 0:
            status_dict_list.append({'No ripple filters enabled.': None})

        status_dict_list.insert(0, {'mpi_rank': self.rank})
        self.comm.send(obj=RippleStatusDictListMessage(self.rank, status_dict_list),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_ripple_thresh_state(self, timestamp, ntrode_id, thresh_state):
        message = RippleThresholdState(timestamp, ntrode_id, thresh_state)

        self.comm.Send(buf=message.pack(),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_process.MPIMessageTag.FEEDBACK_DATA.value)


class RippleManager(realtime_process.BinaryRecordBase, realtime_process.RealtimeClass):
    def __init__(self, rank, local_rec_manager, send_interface: RippleMPISendInterface,
                 data_interface: simulator_process.SimulatorRemoteReceiver):
        super().__init__(rank=rank,
                         local_rec_manager=local_rec_manager,
                         rec_id=1,
                         rec_labels=['current_time',
                                     'ntrode_index',
                                     'thresh_crossed',
                                     'lockout',
                                     'lfp_data',
                                     'rd',
                                     'current_val'],
                         rec_format='Ii??ddd')

        self.rank = rank
        self.mpi_send = send_interface
        self.data_interface = data_interface

        self.num_ntrodes = None
        self.lfp_enable_list = []
        self.ripple_filters = {}
        self.param = RippleParameterMessage()
        self.custom_baseline_mean_dict = {}
        self.custom_baseline_std_dict = {}
        self.data_packet_counter = 0

        self.mpi_send.send_record_register_message(self.get_record_register_message())

    def set_num_trodes(self, message: realtime_process.NumTrodesMessage):
        self.num_ntrodes = message.num_ntrodes
        self.class_log.info('Set number of ntrodes: {:d}'.format(self.num_ntrodes))

    def select_ntrodes(self, ntrode_list):
        self.class_log.debug("Registering continuous channels: {:}.".format(ntrode_list))
        for ntrode in ntrode_list:
            self.data_interface.register_datatype_channel(datatype=datatypes.Datatypes.CONTINUOUS,
                                                          channel=ntrode)

            self.ripple_filters.setdefault(ntrode, RippleFilter(rec_base=self, param=self.param, ntrode_id=ntrode))

    def turn_on_datastreams(self):
        self.class_log.debug("Turn on datastreams.")
        self.data_interface.start_all_streams()

    def update_ripple_parameter(self, parameter: RippleParameterMessage):
        self.class_log.debug("Ripple parameter updated.")
        self.param = parameter
        for rip_filter in self.ripple_filters.values():     # type: RippleFilter
            rip_filter.update_parameter(self.param)

    def set_custom_baseline_mean(self, custom_mean_dict):
        self.class_log.debug("Custom baseline mean updated.")
        self.custom_baseline_mean_dict = custom_mean_dict
        for ntrode_index, rip_filt in self.ripple_filters.items():
            rip_filt.custom_baseline_mean = self.custom_baseline_mean_dict[ntrode_index]

    def set_custom_baseline_std(self, custom_std_dict):
        self.class_log.debug("Custom baseline std updated.")
        self.custom_baseline_std_dict = custom_std_dict
        for ntrode_index, rip_filt in self.ripple_filters.items():
            rip_filt.custom_baseline_std = self.custom_baseline_std_dict[ntrode_index]

    def enable_stimulation(self):
        for rip_filter in self.ripple_filters.values():
            rip_filter.enable_stimulation()

    def disable_stimulation(self):
        for rip_filter in self.ripple_filters.values():
            rip_filter.disable_stimulation()

    def reset_filters(self):
        for rip_filter in self.ripple_filters.values():
            rip_filter.reset_data()

    def process_status_dict_request(self):
        self.class_log.debug('processing status_dict_request.')
        self.mpi_send.send_ripple_status_message(self.get_status_dict_list())

    def get_status_dict_list(self):
        status_list = []
        for rip_filter in self.ripple_filters.values():     # type: RippleFilter
            status_dict = rip_filter.get_status_dict()
            # Don't add status dict if empty
            if status_dict:
                status_list.append(rip_filter.get_status_dict())
        return status_list

    def trigger_termination(self):
        self.data_interface.stop_iterator()

    def process_next_data(self):

        datapoint = next(self.data_interface)

        if isinstance(datapoint, LFPPoint):
            filter_state = self.ripple_filters[datapoint.ntrode_id].process_data(data_point=datapoint)
            self.mpi_send.send_ripple_thresh_state(timestamp=datapoint.timestamp,
                                                   ntrode_id=datapoint.ntrode_id,
                                                   thresh_state=filter_state)

            self.data_packet_counter += 1
            if (self.data_packet_counter % 10000) == 0:
                self.class_log.debug('Received {:} datapoints.'.format(self.data_packet_counter))

        else:
            self.class_log.warning('RippleManager should only receive LFP Data, instead received {:}'.
                                   format(type(datapoint)))


class RippleMPIRecvInterface(realtime_process.RealtimeClass):

    def __init__(self, comm: MPI.Comm, rank, ripple_manager: RippleManager, main_rank=0):
        super().__init__()

        self.comm = comm
        self.rank = rank
        self.rip_man = ripple_manager
        self.main_rank = main_rank
        self.num_ntrodes = None

    def process_next_message(self):
        message = self.comm.recv(tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

        if isinstance(message, realtime_process.TerminateMessage):
            self.class_log.debug("Received TerminateMessage")
            raise StopIteration()

        elif isinstance(message, realtime_process.NumTrodesMessage):
            self.rip_man.set_num_trodes(message)

        elif isinstance(message, ChannelSelection):
            self.rip_man.select_ntrodes(message.ntrode_list)

        elif isinstance(message, TurnOnDataStream):
            self.rip_man.turn_on_datastreams()

        elif isinstance(message, RippleParameterMessage):
            self.rip_man.update_ripple_parameter(message)

        elif isinstance(message, realtime_process.EnableStimulationMessage):
            self.rip_man.enable_stimulation()

        elif isinstance(message, realtime_process.DisableStimulationMessage):
            self.rip_man.disable_stimulation()

        elif isinstance(message, binary_record.BinaryRecordCreateMessage):
            self.rip_man.set_record_writer_from_message(message)

        elif isinstance(message, realtime_process.StartRecordMessage):
            self.rip_man.start_record_writing()

        elif isinstance(message, realtime_process.StopRecordMessage):
            self.rip_man.stop_record_writing()

        elif isinstance(message, realtime_process.CloseRecordMessage):
            self.rip_man.close_record()

        elif isinstance(message, CustomRippleBaselineMeanMessage):
            self.rip_man.set_custom_baseline_mean(message.mean_dict)

        elif isinstance(message, CustomRippleBaselineStdMessage):
            self.rip_man.set_custom_baseline_std(message.std_dict)

        elif isinstance(message, realtime_process.RequestStatusMessage):
            self.class_log.debug('Received RequestStatusMessage.')
            self.rip_man.process_status_dict_request()

        elif isinstance(message, realtime_process.ResetFilterMessage):
            self.rip_man.reset_filters()


class RippleProcess(realtime_process.RealtimeProcess):

    def __init__(self, comm: MPI.Comm, rank, config):
        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state')

        self.mpi_send = RippleMPISendInterface(comm, rank, config)

        super().__init__(comm, rank, config, ThreadClass=RippleDataThread, local_rec_manager=self.local_rec_manager)

        self.mpi_recv = RippleMPIRecvInterface(self.comm, self.rank, self.thread.rip_man, self.config['rank']['supervisor'])

        # TODO temporary measure to enable type hinting (typing.Generics is broken for PyCharm 2016.2.3)
        self.thread = self.thread   # type: RippleDataThread

    def main_loop(self):
        self.thread.start()

        try:
            while True:
                self.mpi_recv.process_next_message()

        except StopIteration as ex:
            self.class_log.info('Terminating RippleProcess (rank: {:})'.format(self.rank))

        # Program should prepare to exit
        self.thread.trigger_termination()

        self.class_log.info("Ripple Process Main Process reached end, exiting.")


class RippleDataThread(realtime_process.RealtimeThread):

    def __init__(self, comm, rank, config, parent: RippleProcess, local_rec_manager):
        super().__init__(comm, rank, config, parent=parent)
        self.local_rec_manager = local_rec_manager

        if self.config['datasource'] == 'simulator':
            data_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                       rank=self.rank,
                                                                       config=self.config)

            self.rip_man = RippleManager(rank=rank,
                                         local_rec_manager=self.local_rec_manager,
                                         send_interface=self.parent.mpi_send,
                                         data_interface=data_interface)
        else:
            raise realtime_process.DataSourceError("No valid data source selected")

        self.stop_next = False

    def trigger_termination(self):
        self.rip_man.trigger_termination()

    def run(self):

        try:
            while True:
                self.rip_man.process_next_data()

        except StopIteration as ex:

            self.class_log.info("Ripple Process Main Thread reached end, exiting.")

