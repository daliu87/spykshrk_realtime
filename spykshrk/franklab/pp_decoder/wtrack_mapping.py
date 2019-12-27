import numpy as np
import pandas as pd
import warnings
import itertools
import copy

import scipy as sp
import scipy.stats

from spykshrk.util import AttrDict, AttrDictEnum
from spykshrk.franklab.warnings import DataInconsistentWarning
from spykshrk.franklab.data_containers import pos_col_format, SpikeObservation
from spykshrk.franklab.wtrack import WtrackArm, Direction, Rotation, Order


class WtrackLinposDecomposer(AttrDict):
    rotations = list(Rotation)
    orders = list(Order)
    prev_next = [Order.prev, Order.next]

    def __init__(self, linpos_flat, encode_settings, bin_size=1):
        super().__init__()

        self.linpos_flat = linpos_flat
        self.encode_settings = encode_settings
        self.wtrack_armcoord = self.encode_settings.wtrack_arm_coordinates

        self.bin_size = bin_size
        self.segment_order_cw = [('center', 'outbound'),
                                 ('left', 'outbound'),
                                 ('left', 'inbound'),
                                 ('right', 'outbound'),
                                 ('right', 'inbound'),
                                 ('center', 'inbound')]
        self.segment_order_ccw = [('center', 'outbound'),
                                  ('right', 'outbound'),
                                  ('right', 'inbound'),
                                  ('left', 'outbound'),
                                  ('left', 'inbound'),
                                  ('center', 'inbound')]

        self.segment_order_cw = list(itertools.product(WtrackArm, Direction))
        self.segment_order_ccw = list(itertools.product(WtrackArm, Direction))
        self.armcoord_cw = self._segment_decomposed_with_buffer(self.segment_order_cw, self.wtrack_armcoord)
        self.armcoord_ccw = self._segment_decomposed_with_buffer(self.segment_order_ccw, self.wtrack_armcoord)
        self.armcoord_cw_num_bins = max([max((buffer.x1, buffer.x2)) for arm in self.armcoord_cw.values()
                                         for direct in arm.values()
                                         for buffer in [direct.prev, direct.main, direct.next]])
        self.armcoord_ccw_num_bins = max([max((buffer.x1, buffer.x2)) for arm in self.armcoord_ccw.values()
                                          for direct in arm.values()
                                          for buffer in [direct.prev, direct.main, direct.next]])
        self.wtrack_armcoord_cw = AttrDictEnum({arm:
                                                AttrDictEnum({direct:
                                                              AttrDictEnum(x1=min(direct_v.prev.x1,
                                                                                  direct_v.main.x1, direct_v.next.x1),
                                                                           x2=max(direct_v.prev.x2,
                                                                                  direct_v.main.x2, direct_v.next.x2))
                                                              for direct, direct_v in arm_v.items()})
                                                for arm, arm_v in self.armcoord_cw.items()})
        self.wtrack_armcoord_ccw = AttrDictEnum({arm:
                                                 AttrDictEnum({direct:
                                                               AttrDictEnum(x1=min(direct_v.prev.x1,
                                                                                   direct_v.main.x1, direct_v.next.x1),
                                                                            x2=max(direct_v.prev.x2,
                                                                                   direct_v.main.x2, direct_v.next.x2))
                                                               for direct, direct_v in arm_v.items()})
                                                 for arm, arm_v in self.armcoord_ccw.items()})
        self.cw_bin_range = [min([direct.x1 for arm in self.wtrack_armcoord_cw.values() for direct in arm.values()]),
                             max([direct.x2 for arm in self.wtrack_armcoord_cw.values() for direct in arm.values()])]
        self.cw_num_bins = self.cw_bin_range[1] - self.cw_bin_range[0]
        self.ccw_bin_range = [min([direct.x1 for arm in self.wtrack_armcoord_ccw.values() for direct in arm.values()]),
                              max([direct.x2 for arm in self.wtrack_armcoord_ccw.values() for direct in arm.values()])]
        self.ccw_num_bins = self.ccw_bin_range[1] - self.ccw_bin_range[0]
        self.wtrack_armcoord_main_cw = AttrDictEnum({arm: AttrDictEnum({direct: direct_v.main
                                                                        for direct, direct_v in arm_v.items()})
                                                     for arm, arm_v in self.armcoord_cw.items()})
        self.wtrack_armcoord_main_ccw = AttrDictEnum({arm: AttrDictEnum({direct: direct_v.main
                                                                         for direct, direct_v in arm_v.items()})
                                                      for arm, arm_v in self.armcoord_ccw.items()})
        self.simple_armcoord_cw, self.simple_armcoord_bins_cw = \
            self._create_wtrack_decomposed_simple_armcoord(self.armcoord_cw, self.bin_size)
        self.simple_armcoord_ccw, self.simple_armcoord_bins_ccw = \
            self._create_wtrack_decomposed_simple_armcoord(self.armcoord_ccw, self.bin_size)
        self.simple_main_armcoord_cw, self.simple_main_armcoord_bins_cw = \
            self._create_wtrack_decomposed_simple_main_armcoord(self.armcoord_cw, self.bin_size)
        self.simple_main_armcoord_ccw, self.simple_main_armcoord_bins_ccw = \
            self._create_wtrack_decomposed_simple_main_armcoord(self.armcoord_ccw, self.bin_size)

        self.sel_data = AttrDictEnum()
        for rot_k in WtrackLinposDecomposer.rotations:
            self.sel_data[rot_k] = AttrDictEnum(main=self._sel_main(eval('self.armcoord_'+rot_k.name),
                                                                    self.wtrack_armcoord,
                                                                    eval('self.'+rot_k.name+'_num_bins'),
                                                                    self.encode_settings))
            for ord_k in WtrackLinposDecomposer.prev_next:
                self.sel_data[rot_k][ord_k] = self._sel_prev_next(ord_k, eval('self.armcoord_'+rot_k.name),
                                                                  self.wtrack_armcoord,
                                                                  eval('self.'+rot_k.name+'_num_bins'),
                                                                  self.encode_settings)

        self.decomp_linpos = self.wtrack_pos_remap_to_decomposed(self.linpos_flat,
                                                                 encode_settings.wtrack_arm_coordinates,
                                                                 self.armcoord_cw, self.armcoord_ccw)

        self.encode_settings_decomp = self._create_decomposed_encode_settings(self.armcoord_cw_num_bins,
                                                                              self.simple_armcoord_cw,
                                                                              self.wtrack_armcoord_cw,
                                                                              self.encode_settings)

    @staticmethod
    def _create_decomposed_encode_settings(decomp_num_bins, simple_armcoord, wtrack_armcoord, encode_settings):
        encode_settings_decomp = copy.deepcopy(encode_settings)
        encode_settings_decomp.pos_num_bins = decomp_num_bins
        encode_settings_decomp.pos_col_names = pos_col_format(range(decomp_num_bins), decomp_num_bins)
        encode_settings_decomp.pos_bins = np.arange(0, decomp_num_bins)
        encode_settings_decomp.pos_bin_edges = np.arange(0, decomp_num_bins + 0.0001, 1)
        encode_settings_decomp.pos_kernel = sp.stats.norm.pdf(np.arange(0, decomp_num_bins, 1), decomp_num_bins/2,
                                                                   encode_settings_decomp.pos_kernel_std)
        encode_settings_decomp.pos_col_names = pos_col_format(range(decomp_num_bins), decomp_num_bins)
        encode_settings_decomp.arm_coordinates = simple_armcoord
        encode_settings_decomp.wtrack_decomp_arm_coordinates = wtrack_armcoord

        return encode_settings_decomp

    @staticmethod
    def _segment_decomposed_with_buffer(segment_order, wtrack_arm_coord):
        seg_decomposed = AttrDictEnum()
        seg_offset = 0
        for ii, seg in enumerate(segment_order):
            prev_seg = segment_order[ii-1]
            next_seg = segment_order[(ii+1) % len(segment_order)]

            prev_seg_len = wtrack_arm_coord[prev_seg[0]][prev_seg[1]].len
            main_seg_len = wtrack_arm_coord[seg[0]][seg[1]].len
            next_seg_len = wtrack_arm_coord[next_seg[0]][next_seg[1]].len

            prev_seg_start = seg_offset
            main_seg_start = prev_seg_start + prev_seg_len
            next_seg_start = main_seg_start + main_seg_len
            next_seg_end = next_seg_start + next_seg_len
            seg_total_len = prev_seg_len + main_seg_len + next_seg_len
            seg_offset += seg_total_len

            arm_dict = seg_decomposed.setdefault(seg[0], AttrDictEnum())
            dir_dict = arm_dict.setdefault(seg[1],
                                           AttrDictEnum({Order.prev: AttrDictEnum(x1=prev_seg_start,
                                                                                  x2=main_seg_start,
                                                                                  len=prev_seg_len,
                                                                                  seg=prev_seg),
                                                         Order.main: AttrDictEnum(x1=main_seg_start,
                                                                                  x2=next_seg_start,
                                                                                  len=main_seg_len),
                                                         Order.next: AttrDictEnum(x1=next_seg_start,
                                                                                  x2=next_seg_end,
                                                                                  len=next_seg_len,
                                                                                  seg=next_seg),
                                                         "prev_seg": prev_seg,
                                                         "next_seg": next_seg}))

        return seg_decomposed

    @staticmethod
    def _create_wtrack_decomposed_simple_armcoord(decomp_armcoord, bin_size=1):
        simple_armcoord = [(min(direct.prev.x1, direct.main.x1, direct.next.x1),
                            max(direct.prev.x2, direct.main.x2, direct.next.x2))
                           for arm in decomp_armcoord.values() for direct in arm.values()]
        simple_armcoord.sort(key=lambda tup: tup[0])
        simple_armcoord_bins = np.array([])
        for seg in simple_armcoord:
            simple_armcoord_bins = np.append(simple_armcoord_bins, np.arange(seg[0], seg[1], bin_size))
        return simple_armcoord, simple_armcoord_bins

    @staticmethod
    def _create_wtrack_decomposed_simple_main_armcoord(decomp_armcoord, bin_size=1):
        simple_armcoord = [(direct.main.x1, direct.main.x2)
                           for arm in decomp_armcoord.values() for direct in arm.values()]
        simple_armcoord.sort(key=lambda tup: tup[0])
        simple_armcoord_bins = np.array([])
        for seg in simple_armcoord:
            simple_armcoord_bins = np.append(simple_armcoord_bins, np.arange(seg[0], seg[1], bin_size))
        return simple_armcoord, simple_armcoord_bins

    @staticmethod
    def _sel_main(decomp_armcoord, wtrack_armcoord, decomp_num_bins, encode_settings):
        main_sel = AttrDictEnum(decomposed=AttrDictEnum(), wtrack=AttrDictEnum())
        main_sel.decomposed['ind'] = np.concatenate([np.arange(direct_v.main.x1, direct_v.main.x2,
                                                               encode_settings.pos_bin_delta)
                                                     for arm_v in decomp_armcoord.values()
                                                     for direct_v in arm_v.values()])
        main_sel.decomposed['col'] = pos_col_format(main_sel.decomposed['ind'], decomp_num_bins)
        main_sel.wtrack['ind'] =  np.concatenate([np.arange(wtrack_armcoord[arm][direct].x1,
                                                            wtrack_armcoord[arm][direct].x2,
                                                             encode_settings.pos_bin_delta)
                                                  for arm, arm_v in decomp_armcoord.items()
                                                  for direct, direct_v in arm_v.items()])
        main_sel.wtrack['col'] = pos_col_format(main_sel.wtrack['ind'], decomp_num_bins)

        return main_sel

    @staticmethod
    def _sel_prev_next(prev_next, decomp_armcoord, wtrack_armcoord, decomp_num_bins, encode_settings):
        sel = AttrDictEnum(decomposed=AttrDictEnum(), wtrack=AttrDictEnum())
        sel.decomposed['ind'] = np.concatenate([np.arange(direct_v[prev_next].x1, direct_v[prev_next].x2,
                                                          encode_settings.pos_bin_delta)
                                                for arm_v in decomp_armcoord.values() for direct_v in arm_v.values()])
        sel.decomposed['col'] = pos_col_format(sel.decomposed['ind'], decomp_num_bins)
        sel.wtrack['ind'] = np.concatenate([np.arange(wtrack_armcoord[direct_v[prev_next.name+'_seg'][0]]
                                                      [direct_v[prev_next.name+'_seg'][1]].x1,
                                                      wtrack_armcoord[direct_v[prev_next.name+'_seg'][0]]
                                                      [direct_v[prev_next.name+'_seg'][1]].x2,
                                                      encode_settings.pos_bin_delta)
                                            for arm, arm_v in decomp_armcoord.items()
                                            for direct, direct_v in arm_v.items()])
        sel.wtrack['col'] = pos_col_format(sel.wtrack['ind'], decomp_num_bins)

        return sel

    @staticmethod
    def wtrack_pos_remap_to_decomposed(linpos_flat, wtrack_arm_coord, wtrack_decomposed_cw, wtrack_decomposed_ccw):
        linpos_arm_dir = linpos_flat.groupby(['arm', 'direction'])

        decomposed_linpos = pd.DataFrame()

        for entry in linpos_arm_dir:
            key = entry[0]
            table = entry[1]
            arm_coord_range = wtrack_arm_coord[key[0].name][key[1].name]
            decomposed_range_cw = wtrack_decomposed_cw[key[0].name][key[1].name]
            decomposed_range_ccw = wtrack_decomposed_ccw[key[0].name][key[1].name]
            decomposed_table = table.copy()
            decomposed_table.loc[:, 'linpos_cw'] = (decomposed_table.loc[:, 'linpos_flat'] - arm_coord_range.x1 +
                                                    decomposed_range_cw.main.x1)
            decomposed_table.loc[:, 'linpos_cw_next'] = (decomposed_table.loc[:, 'linpos_flat'] - arm_coord_range.x1 +
                                                         decomposed_range_cw.next.x1)
            decomposed_table.loc[:, 'linpos_cw_prev'] = (decomposed_table.loc[:, 'linpos_flat'] - arm_coord_range.x1 +
                                                         decomposed_range_cw.prev.x1)
            decomposed_table.loc[:, 'linpos_ccw'] = (decomposed_table.loc[:, 'linpos_flat'] - arm_coord_range.x1 +
                                                     decomposed_range_ccw.main.x1)
            decomposed_table.loc[:, 'linpos_ccw_next'] = (decomposed_table.loc[:, 'linpos_flat'] - arm_coord_range.x1 +
                                                          decomposed_range_ccw.next.x1)
            decomposed_table.loc[:, 'linpos_ccw_prev'] = (decomposed_table.loc[:, 'linpos_flat'] - arm_coord_range.x1 +
                                                          decomposed_range_ccw.prev.x1)

            decomposed_linpos = decomposed_linpos.append(decomposed_table)

        decomposed_linpos.sort_index(inplace=True)
        return decomposed_linpos


class WtrackLinposRecomposer(AttrDict):
    rotations = ['cw', 'ccw']
    orders = ['prev', 'next']

    def __init__(self, encoder_cw, encoder_ccw, wtrack_decomposed, encode_settings):
        super().__init__()
        self.encoder_cw = encoder_cw
        self.encoder_ccw = encoder_ccw
        self.wtrack_decomposed = wtrack_decomposed
        self.encode_settings = encode_settings
        self.observ = self._wtrack_recompose_observ(encoder_cw.observ_obj, encoder_ccw.observ_obj,
                                                    self.wtrack_decomposed, self.encode_settings)
        self.prob_no_spike = self._wtrack_recompose_prob_no_spike(encoder_cw.prob_no_spike, encoder_ccw.prob_no_spike,
                                                                  self.wtrack_decomposed, self.encode_settings)
        self.trans_mat = self._wtrack_recompose_trans_mat(encoder_cw.trans_mat['learned'],
                                                          encoder_ccw.trans_mat['learned'],
                                                          self.wtrack_decomposed, self.encode_settings)

    @staticmethod
    def decomposed_pos_remap_to_wtrack(pos, decomposed_armcoord, wtrack_armcoord):
        wtrack_pos = np.empty(pos.shape)
        wtrack_pos.fill(np.nan)
        for arm_k, arm_v in decomposed_armcoord.items():
            for direct_k, direct_v in arm_v.items():
                dir_pos_ind = (pos >= direct_v.main.x1) & (pos <= direct_v.main.x2)
                wtrack_dir_pos = pos[dir_pos_ind] - direct_v.main.x1 + wtrack_armcoord[arm_k][direct_k].x1
                wtrack_pos[dir_pos_ind] = wtrack_dir_pos

        if np.isnan(wtrack_pos).any():
            warnings.warn('Position in decomposed main coordinate system does not match wtrack coordinates.',
                          DataInconsistentWarning)

        return wtrack_pos

    @staticmethod
    def _wtrack_recompose_prob_no_spike(prob_no_spike_cw, prob_no_spike_ccw, wtrack_decomposed, encoding_settings):
        prob_no_spike = {}
        for rot_k in WtrackLinposRecomposer.rotations:
            for tet_id, prob_no_spike_tet in prob_no_spike_cw.items():
                prob_no_spike.setdefault(tet_id, np.zeros(encoding_settings.pos_num_bins))
                prob_no_spike[tet_id][wtrack_decomposed.sel_data[rot_k]['main']['wtrack']['ind']] += \
                         prob_no_spike_tet[wtrack_decomposed.sel_data[rot_k]['main']['decomposed']['ind']]
        return prob_no_spike

    @staticmethod
    def _wtrack_recompose_trans_mat(trans_mat_cw, trans_mat_ccw,
                                    wtrack_decomposed, encode_settings):
        trans_mat = np.zeros([encode_settings.pos_num_bins]*2)
        #trans_mat += WtrackRecomposer._wtrack_recompose_trans_mat_part('cw', 'main', 'main', trans_mat_cw,
        #                                                               wtrack_decomposed, encode_settings)
        for rot_k in WtrackLinposRecomposer.rotations:
            trans_mat += WtrackLinposRecomposer._wtrack_recompose_trans_mat_part(rot_k, 'main', 'main',
                                                                                 eval('trans_mat_'+rot_k),
                                                                           wtrack_decomposed, encode_settings)
            for ord_k in WtrackLinposRecomposer.orders:
                for order1, order2 in itertools.permutations([ord_k, 'main']):
                    trans_mat += WtrackLinposRecomposer._wtrack_recompose_trans_mat_part(rot_k, order1, order2,
                                                                                   eval('trans_mat_'+rot_k),
                                                                                   wtrack_decomposed, encode_settings)
        return trans_mat

    @staticmethod
    def _wtrack_recompose_trans_mat_part(rotation, order1, order2, decomp_trans_mat,
                                         wtrack_decomposed, encode_settings):
        trans_mat = np.zeros([encode_settings.pos_num_bins]*2)
        trans_main_sel = np.meshgrid(wtrack_decomposed.sel_data[rotation][order1]['decomposed']['ind'],
                                     wtrack_decomposed.sel_data[rotation][order2]['decomposed']['ind'])
        trans_wtrack_sel = np.meshgrid(wtrack_decomposed.sel_data[rotation][order2]['wtrack']['ind'],
                                       wtrack_decomposed.sel_data[rotation][order1]['wtrack']['ind'])
        trans_mat[np.ix_(wtrack_decomposed.sel_data[rotation][order2]['wtrack']['ind'],
                         wtrack_decomposed.sel_data[rotation][order1]['wtrack']['ind'])] = \
                decomp_trans_mat[trans_main_sel[1], trans_main_sel[0]]

        return trans_mat

    @staticmethod
    def _wtrack_recompose_observ(observ_cw, observ_ccw, wtrack_decomposed, encode_settings):

        observ = pd.DataFrame(np.zeros((observ_cw.shape[0],
                                        len(wtrack_decomposed.sel_data['cw']['main']['wtrack']['ind']))),
                              columns=encode_settings.pos_col_names, index=observ_cw.index)

        observ.iloc[:, wtrack_decomposed.sel_data['cw']['main']['wtrack']['ind']] = \
                observ_cw.loc[:, wtrack_decomposed.sel_data['cw']['main']['decomposed']['col']].values

        for rot_k in WtrackLinposRecomposer.rotations:
            for ord_k in WtrackLinposRecomposer.orders:
                observ.iloc[:, wtrack_decomposed.sel_data[rot_k][ord_k]['wtrack']['ind']] += \
                        eval('observ_'+rot_k).loc[:, wtrack_decomposed.
                                                  sel_data[rot_k][ord_k]['decomposed']['col']].values

        observ = observ.join(observ_cw.get_other_view())
        observ = SpikeObservation.create_default(observ, enc_settings=wtrack_decomposed.encode_settings)

        observ['position'] = WtrackLinposRecomposer.decomposed_pos_remap_to_wtrack(observ['position'],
                                                                                   wtrack_decomposed.armcoord_cw,
                                                                                   wtrack_decomposed.wtrack_armcoord)

        return observ
