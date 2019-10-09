import numpy as np
import scipy as sp
import pandas as pd
import functools
import holoviews as hv

from spykshrk.franklab.data_containers import FlatLinearPosition, SpikeFeatures


class LinPosVisualizer:
    def __init__(self, linpos_flat):
        self.linpos_flat = linpos_flat

    def plot_linpos_flat(self):
        hv.Points(self.linpos_flat['linpos_flat'], kdims=[('samples', 'Samples'), ('pos', 'Position (cm)')],
                  label='Synthetic position data for UnitGenerator and encoding algorithm'). \
            opts({'Points': {'marker': 'o', 'aspect': 2}})


class TetrodeVisualizer:

    def __init__(self, spk_amp: SpikeFeatures, linpos_flat: FlatLinearPosition, unit_spks: dict):
        self.spk_amp = spk_amp
        self.linpos_flat = linpos_flat
        self.unit_spks = unit_spks

    def mark_color_3d_plot(self, elevation, azimuth, col1='c00', col2='c01', col3='c02'):
        scatter = [hv.Scatter3D(mark_pos.loc[:, [col1, col2, col3]]).opts(dict(Scatter3D=dict(bgcolor='black', s=3)))
                   for elec_id, mark_pos in self.unit_spks.items()]
        overlay = hv.Overlay(scatter, label='Plot of spikes and their features ' +
                             col1 + ', ' + col2 + ', and ' + col3)
        overlay = overlay.opts({'Scatter3D': {'plot': {'fig_size': 400, 'azimuth': int(azimuth),
                                                       'elevation': int(elevation)},
                                              'norm': {'framewise': True}}})
        return overlay

    def plot_color_3d_dynamic(self, col1, col2, col3):
        dmap = hv.DynamicMap(callback=functools.partial(self.mark_color_3d_plot, col1=col1, col2=col2, col3=col3),
                             kdims=['elevation', 'azimuth'], cache_size=1)
        dmap = dmap.redim.values(elevation=range(0, 181, 5),
                                 azimuth=range(-90, 91, 5)).opts(norm=dict(framewise=True))
        return dmap
