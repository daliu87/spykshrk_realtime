import spykshrk.realtime.rst.RSTPython as RST
import struct
from spykshrk.realtime.realtime_logging import PrintableMessage
import numpy as np


class PosBinStruct:
    def __init__(self, pos_range, num_bins):
        self.pos_range = pos_range
        self.num_bins = num_bins
        self.pos_bin_edges = np.linspace(pos_range[0], pos_range[1], num_bins + 1, endpoint=True, retstep=False)
        self.pos_bin_center = (self.pos_bin_edges[:-1] + self.pos_bin_edges[1:]) / 2
        self.pos_bin_delta = self.pos_bin_center[1] - self.pos_bin_center[0]

    def which_bin(self, pos):
        return np.nonzero(np.diff(self.pos_bin_edges > pos))


class RSTParameter:
    def __init__(self, kernel, pos_hist_struct):
        self.kernel = kernel
        self.pos_hist_struct = pos_hist_struct


class RSTKernelEncoderQuery(PrintableMessage):
    _header_byte_fmt = '=qiii'
    _header_byte_len = struct.calcsize(_header_byte_fmt)

    def __init__(self, query_time, ntrode_id, query_weights, query_positions, query_hist):
        self.query_time = query_time
        self.ntrode_id = ntrode_id
        self.query_weights = query_weights
        self.query_positions = query_positions
        self.query_hist = query_hist

    def pack(self):
        query_len = len(self.query_weights)
        query_byte_len = query_len * struct.calcsize('=f')
        query_hist_len = len(self.query_hist)
        query_hist_byte_len = query_hist_len * struct.calcsize('=d')

        message_bytes = struct.pack(self._header_byte_fmt,
                                    self.query_time,
                                    self.ntrode_id,
                                    query_byte_len,
                                    query_hist_byte_len)

        message_bytes = message_bytes + self.query_weights.tobytes() + \
                        self.query_positions.tobytes() + self.query_hist.tobytes()

        return message_bytes

    @classmethod
    def unpack(cls, message_bytes):
        query_time, ntrode_id, query_len, query_hist_len = struct.unpack(cls._header_byte_fmt,
                                                                         message_bytes[0:cls._header_byte_len])

        query_weights = np.frombuffer(message_bytes[cls._header_byte_len: cls._header_byte_len+query_len],
                                      dtype='float32')

        query_positions = np.frombuffer(message_bytes[cls._header_byte_len+query_len:
                                                      cls._header_byte_len+2*query_len],
                                       dtype='float32')

        query_hist = np.frombuffer(message_bytes[cls._header_byte_len+2*query_len:
                                                 cls._header_byte_len+2*query_len+query_hist_len])

        return cls(query_time=query_time, ntrode_id=ntrode_id, query_weights=query_weights,
                   query_positions=query_positions, query_hist=query_hist)


class RSTKernelEncoder:
    def __init__(self, filename, new_tree, param):
        self.param = param
        self.kernel = param.kernel
        self.filename = filename
        self.new_tree = new_tree

        self.tree = RST.RSTPython(filename.encode('utf-8'), new_tree, param.kernel)
        self.covariate = 0
        self.pos_hist = np.zeros(param.pos_hist_struct.num_bins)

    def update_covariate(self, covariate):
        self.covariate = covariate
        # bin_idx = np.nonzero((self.param.pos_hist_struct.pos_bin_edges - covariate) > 0)[0][0] - 1
        bin_idx = self.param.pos_hist_struct.which_bin(self.covariate)
        self.pos_hist[bin_idx] += 1

    def new_mark(self, mark, new_cov=None):
        # update new covariate if specified, otherwise use previous covariate state
        if new_cov:
            self.update_covariate(new_cov)

        self.tree.insert_rec(mark[0], mark[1], mark[2],
                             mark[3], self.covariate)

    def query_mark(self, mark):
        x1 = mark[0]
        x2 = mark[1]
        x3 = mark[2]
        x4 = mark[3]
        x1_l = x1 - self.kernel.stddev * 2.5
        x2_l = x2 - self.kernel.stddev * 2.5
        x3_l = x3 - self.kernel.stddev * 2.5
        x4_l = x4 - self.kernel.stddev * 2.5
        x1_h = x1 + self.kernel.stddev * 2.5
        x2_h = x2 + self.kernel.stddev * 2.5
        x3_h = x3 + self.kernel.stddev * 2.5
        x4_h = x4 + self.kernel.stddev * 2.5
        query_weights, query_positions = self.tree.query_rec(x1_l, x2_l, x3_l, x4_l,
                                                             x1_h, x2_h, x3_h, x4_h,
                                                             x1, x2, x3, x4)
        return query_weights, query_positions

    def query_mark_hist(self, mark, time, ntrode_id):
        query_weights, query_positions = self.query_mark(mark)
        query_hist, query_hist_edges = np.histogram(
            a=query_positions, bins=self.param.pos_hist_struct.pos_bin_edges,
            weights=query_weights, normed=False)
        query_hist = query_hist / (self.pos_hist + 0.001)
        query_hist = query_hist / (np.sum(query_hist) * self.param.pos_hist_struct.pos_bin_delta)

        return RSTKernelEncoderQuery(query_time=time,
                                     ntrode_id=ntrode_id,
                                     query_weights=query_weights,
                                     query_positions=query_positions,
                                     query_hist=query_hist)

