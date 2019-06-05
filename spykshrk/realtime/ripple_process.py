import struct
from collections import OrderedDict
from collections import deque

from mpi4py import MPI

import spykshrk.realtime.binary_record as binary_record
import spykshrk.realtime.datatypes as datatypes
import spykshrk.realtime.realtime_base as realtime_base
import spykshrk.realtime.realtime_logging as rt_logging
import spykshrk.realtime.simulator.simulator_process as simulator_process
import spykshrk.realtime.timing_system as timing_system
from spykshrk.realtime.datatypes import LFPPoint
from spykshrk.realtime.realtime_base import ChannelSelection, TurnOnDataStream


class RippleParameterMessage(rt_logging.PrintableMessage):
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


class CustomRippleBaselineMeanMessage(rt_logging.PrintableMessage):
    def __init__(self, mean_dict):
        self.mean_dict = mean_dict


class CustomRippleBaselineStdMessage(rt_logging.PrintableMessage):
    def __init__(self, std_dict):
        self.std_dict = std_dict


class RippleStatusDictListMessage(rt_logging.PrintableMessage):
    def __init__(self, ripple_rank, status_dict_list):
        self.ripple_rank = ripple_rank
        self.status_dict_list = status_dict_list


class RippleThresholdState(rt_logging.PrintableMessage):
    """"Message containing whether or not at a given timestamp a ntrode's ripple filter threshold is crossed.

    This message has helper serializer/deserializer functions to be used to speed transmission.
    """
    _byte_format = 'Iii'

    def __init__(self, timestamp, elec_grp_id, threshold_state):
        self.timestamp = timestamp
        self.elec_grp_id = elec_grp_id
        self.threshold_state = threshold_state

    def pack(self):
        return struct.pack(self._byte_format, self.timestamp, self.elec_grp_id, self.threshold_state)

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, elec_grp_id, threshold_state = struct.unpack(cls._byte_format, message_bytes)
        return cls(timestamp=timestamp, elec_grp_id=elec_grp_id, threshold_state=threshold_state)


class RippleFilter(rt_logging.LoggingClass):
    def __init__(self, rec_base: realtime_base.BinaryRecordBase, param: RippleParameterMessage,
                 elec_grp_id):
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

        self.elec_grp_id = elec_grp_id
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
        self.class_log.debug("Custom Baseline Mean for {}, {}".format(self.elec_grp_id, value))
        if value:
            self._custom_baseline_mean = value
        else:
            pass

    @property
    def custom_baseline_std(self):
        return self._custom_baseline_mean

    @custom_baseline_std.setter
    def custom_baseline_std(self, value):
        self.class_log.debug("Custom Baseline Std for {}, {}".format(self.elec_grp_id, value))
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

    def process_data(self, timestamp, data):

        self.current_time = timestamp

        if self.current_time - self.last_stim_time < self.param.lockout_time:
            self.in_lockout = True
        else:
            self.in_lockout = False

        if self.in_lockout:
            rd = self.update_filter(((self.current_time - self.last_stim_time) / self.param.lockout_time)
                                    * data)
            self.current_val = self.ripple_mean
            self.thresh_crossed = False

        else:

            rd = self.update_filter(data)

            y = abs(rd)

            if not self.stim_enabled:
                self.ripple_mean += (y - self.ripple_mean) / self.param.samp_divisor
                self.ripple_std += (abs(y - self.ripple_mean) - self.ripple_std) / self.param.samp_divisor
                if not self.param.use_custom_baseline:  # only update the threshold if we're not using a custom baseline
                    self.current_thresh = self.ripple_mean + self.ripple_std * self.param.ripple_threshold
                    # print('ntrode', crf.nTrodeId, 'mean', crf.rippleMean)

            #if self.current_time % 30000 == 0:
            #    self.class_log.info((self.stim_enabled, self.ripple_mean, self.ripple_std))

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
                if self.current_val >= (self.custom_baseline_mean + self.custom_baseline_std *
                                        self.param.ripple_threshold):
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
        self.rec_base.write_record(realtime_base.RecordIDs.RIPPLE_STATE,
                                   self.current_time, self.elec_grp_id, self.thresh_crossed,
                                   self.in_lockout, self._custom_baseline_mean, self._custom_baseline_std,
                                   int(data), rd, self.current_val)

        return self.thresh_crossed

    def get_status_dict(self):
        s = OrderedDict()
        if self.param.enabled:
            s['nt'] = self.elec_grp_id

            if self.param.use_custom_baseline:
                s['custom_mean'] = self.custom_baseline_mean
                s['custom_std'] = self.custom_baseline_std
            else:
                s['mean'] = self.ripple_mean
                s['std'] = self.ripple_std
        return s


class RippleMPISendInterface(realtime_base.RealtimeMPIClass):

    def __init__(self, comm: MPI.Comm, rank, config):
        super().__init__(comm=comm, rank=rank, config=config)

        self.num_ntrodes = None

    def send_record_register_messages(self, record_register_messages):
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_ripple_status_message(self, status_dict_list):
        if len(status_dict_list) == 0:
            status_dict_list.append({'No ripple filters enabled.': None})

        status_dict_list.insert(0, {'mpi_rank': self.rank})
        self.comm.send(obj=RippleStatusDictListMessage(self.rank, status_dict_list),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_ripple_thresh_state(self, timestamp, elec_grp_id, thresh_state):
        message = RippleThresholdState(timestamp, elec_grp_id, thresh_state)

        self.comm.Send(buf=message.pack(),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.FEEDBACK_DATA.value)

    def forward_timing_message(self, timing_msg: timing_system.TimingMessage):
        self.comm.Send(buf=timing_msg.pack(),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.TIMING_MESSAGE.value)

    def send_time_sync_report(self, time):
        self.comm.send(obj=realtime_base.TimeSyncReport(time),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def all_barrier(self):
        self.comm.Barrier()


class RippleManager(realtime_base.BinaryRecordBaseWithTiming, rt_logging.LoggingClass):
    def __init__(self, rank, local_rec_manager, send_interface: RippleMPISendInterface,
                 data_interface: realtime_base.DataSourceReceiver):
        super().__init__(rank=rank,
                         local_rec_manager=local_rec_manager,
                         send_interface=send_interface,
                         rec_ids=[realtime_base.RecordIDs.RIPPLE_STATE],
                         rec_labels=[['timestamp',
                                      'elec_grp_id',
                                      'thresh_crossed',
                                      'lockout',
                                      'custom_mean',
                                      'custom_std',
                                      'lfp_data',
                                      'rd',
                                      'current_val']],
                         rec_formats=['Ii??ddddd'])

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

        # self.mpi_send.send_record_register_messages(self.get_record_register_messages())

    def set_num_trodes(self, message: realtime_base.NumTrodesMessage):
        self.num_ntrodes = message.num_ntrodes
        self.class_log.info('Set number of ntrodes: {:d}'.format(self.num_ntrodes))

    def select_ntrodes(self, ntrode_list):
        self.class_log.debug("Registering continuous channels: {:}.".format(ntrode_list))
        for electrode_group in ntrode_list:
            self.data_interface.register_datatype_channel(channel=electrode_group)
            self.ripple_filters.setdefault(electrode_group, RippleFilter(rec_base=self, param=self.param,
                                                                         elec_grp_id=electrode_group))

    def turn_on_datastreams(self):
        self.class_log.info("Turn on datastreams.")
        self.data_interface.start_all_streams()

    def update_ripple_parameter(self, parameter: RippleParameterMessage):
        self.class_log.info("Ripple parameter updated.")
        self.param = parameter
        for rip_filter in self.ripple_filters.values():     # type: RippleFilter
            rip_filter.update_parameter(self.param)

    def set_custom_baseline_mean(self, custom_mean_dict):
        self.class_log.info("Custom baseline mean updated.")
        self.custom_baseline_mean_dict = custom_mean_dict
        for ntrode_index, rip_filt in self.ripple_filters.items():
            rip_filt.custom_baseline_mean = self.custom_baseline_mean_dict[ntrode_index]

    def set_custom_baseline_std(self, custom_std_dict):
        self.class_log.info("Custom baseline std updated.")
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

        msgs = self.data_interface.__next__()
        if msgs is None:
            # no data available but datastream has not closed, continue polling
            pass
        else:
            datapoint = msgs[0]
            timing_msg = msgs[1]

            if isinstance(datapoint, LFPPoint):
                self.record_timing(timestamp=datapoint.timestamp, elec_grp_id=datapoint.elec_grp_id,
                                   datatype=datatypes.Datatypes.LFP, label='rip_recv')

                filter_state = (self.ripple_filters[datapoint.elec_grp_id].
                                process_data(timestamp=datapoint.timestamp,
                                             data=datapoint.data))

                self.record_timing(timestamp=datapoint.timestamp, elec_grp_id=datapoint.elec_grp_id,
                                   datatype=datatypes.Datatypes.LFP, label='rip_send')

                self.mpi_send.send_ripple_thresh_state(timestamp=datapoint.timestamp,
                                                       elec_grp_id=datapoint.elec_grp_id,
                                                       thresh_state=filter_state)

                self.data_packet_counter += 1
                if (self.data_packet_counter % 100000) == 0:
                    self.class_log.debug('Received {:} LFP datapoints.'.format(self.data_packet_counter))

            else:
                self.class_log.warning('RippleManager should only receive LFP Data, instead received {:}'.
                                       format(type(datapoint)))

            if timing_msg is not None:
                # Currently timing message is always None
                pass


class RippleMPIRecvInterface(realtime_base.RealtimeMPIClass):

    def __init__(self, comm: MPI.Comm, rank, config, ripple_manager: RippleManager):
        super().__init__(comm=comm, rank=rank, config=config)

        self.rip_man = ripple_manager
        self.main_rank = self.config['rank']['supervisor']
        self.num_ntrodes = None

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

        elif isinstance(message, realtime_base.NumTrodesMessage):
            self.rip_man.set_num_trodes(message)

        elif isinstance(message, ChannelSelection):
            self.rip_man.select_ntrodes(message.ntrode_list)

        elif isinstance(message, TurnOnDataStream):
            self.rip_man.turn_on_datastreams()

        elif isinstance(message, RippleParameterMessage):
            self.rip_man.update_ripple_parameter(message)

        elif isinstance(message, realtime_base.EnableStimulationMessage):
            self.rip_man.enable_stimulation()

        elif isinstance(message, realtime_base.DisableStimulationMessage):
            self.rip_man.disable_stimulation()

        elif isinstance(message, binary_record.BinaryRecordCreateMessage):
            self.rip_man.set_record_writer_from_message(message)

        elif isinstance(message, realtime_base.StartRecordMessage):
            self.rip_man.start_record_writing()

        elif isinstance(message, realtime_base.StopRecordMessage):
            self.rip_man.stop_record_writing()

        elif isinstance(message, realtime_base.CloseRecordMessage):
            self.rip_man.close_record()

        elif isinstance(message, CustomRippleBaselineMeanMessage):
            self.rip_man.set_custom_baseline_mean(message.mean_dict)

        elif isinstance(message, CustomRippleBaselineStdMessage):
            self.rip_man.set_custom_baseline_std(message.std_dict)

        elif isinstance(message, realtime_base.RequestStatusMessage):
            self.class_log.debug('Received RequestStatusMessage.')
            self.rip_man.process_status_dict_request()

        elif isinstance(message, realtime_base.ResetFilterMessage):
            self.rip_man.reset_filters()

        elif isinstance(message, realtime_base.TimeSyncInit):
            self.rip_man.sync_time()

        elif isinstance(message, realtime_base.TimeSyncSetOffset):
            self.rip_man.update_offset(message.offset_time)


class RippleProcess(realtime_base.RealtimeProcess):

    def __init__(self, comm: MPI.Comm, rank, config):
        super().__init__(comm, rank, config)

        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
                                                                          manager_rank=config['rank']['supervisor'])

        self.mpi_send = RippleMPISendInterface(comm, rank, config)

        if self.config['datasource'] == 'simulator':
            data_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                    rank=self.rank,
                                                                    config=self.config,
                                                                    datatype=datatypes.Datatypes.LFP)
        elif self.config['datasource'] == 'trodes':
            data_interface = simulator_process.TrodesDataReceiver(comm=self.comm,
                                                                                rank=self.rank,
                                                                                config=self.config,
                                                                                datatype=datatypes.Datatypes.LFP)
        else:
            raise realtime_base.DataSourceError("No valid data source selected")

        self.rip_man = RippleManager(rank=rank,
                                    local_rec_manager=self.local_rec_manager,
                                    send_interface=self.mpi_send,
                                    data_interface=data_interface)

        self.mpi_recv = RippleMPIRecvInterface(self.comm, self.rank, self.config, self.rip_man)

        self.terminate = False
        # config['trodes_network']['networkobject'].registerTerminateCallback(self.trigger_termination)

        # First Barrier to finish setting up nodes
        self.class_log.debug("First Barrier")
        self.comm.Barrier()

    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):

        self.rip_man.setup_mpi()

        try:
            while not self.terminate:
                self.mpi_recv.__next__()
                self.rip_man.process_next_data()

        except StopIteration as ex:
            self.class_log.info('Terminating RippleProcess (rank: {:})'.format(self.rank))

        self.class_log.info("Ripple Process Main Process reached end, exiting.")


