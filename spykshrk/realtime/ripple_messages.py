
import trodes.FSData.realtime_process as realtime_process


class NumTrodesMessage(realtime_process.RealtimeMessage):
    def __init__(self, num_ntrodes):
        self.num_ntrodes = num_ntrodes


class TurnOnLFPMessage(realtime_process.RealtimeMessage):
    def __init__(self, lfp_enable_list):
        self.lfp_enable_list = lfp_enable_list


class TurnOffLFPMessage(realtime_process.RealtimeMessage):
    def __init__(self):
        pass


class RippleParameterMessage(realtime_process.RealtimeMessage):
    def __init__(self, rip_coeff1=1.2, rip_coeff2=0.2, ripple_threshold=5, samp_divisor=10000, n_above_thresh=1,
                 lockout_time=7500, detect_no_ripple_time = 60000, dio_gate_port=None, detect_no_ripples=False,
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
    def __init__(self, timestamp, ntrode_index, threshold_state):
        self.timestamp = timestamp
        self.ntrode_index = ntrode_index
        self.threshold_state = threshold_state

