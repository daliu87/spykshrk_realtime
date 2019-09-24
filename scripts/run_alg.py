
# This snippet of code properly adds the working source root path to python's path
# so you no longer have to install spykshrk through setuptools
import sys, os
root_depth = 1
notebook_dir = os.path.dirname(os.path.abspath(globals()['__file__']))
root_path = os.path.abspath(os.path.join(notebook_dir, '../'*root_depth))
# Add to python's path
try:
    while True:
        sys.path.remove(root_path)
except ValueError:
    # no more root paths
    pass
sys.path.append(root_path)
# Alternatively set root path as current working directory
#os.chdir(root_path)


import pandas as pd
import numpy as np
import scipy as sp
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import json
import os
import scipy.signal
import functools
import time

from spykshrk.util import AttrDict
import spykshrk.franklab.filterframework_util as ff_util

from spykshrk.realtime.simulator import nspike_data

from spykshrk.franklab.pp_decoder.util import gaussian, normal2D, apply_no_anim_boundary, \
                                              simplify_pos_pandas, normal_pdf_int_lookup
from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPDecoder, OfflinePPEncoder 
from spykshrk.franklab.data_containers import DataFrameClass, EncodeSettings, DecodeSettings, \
                                              SpikeObservation, LinearPosition, StimLockout, Posteriors, \
                                              FlatLinearPosition, SpikeWaves, SpikeFeatures, \
                                              pos_col_format, DayEpochTimeSeries

#from spykshrk.franklab.pp_decoder.visualization import DecodeVisualizer
#from spykshrk.franklab.pp_decoder.decode_error import LinearDecodeError

from spykshrk.franklab.franklab_data import FrankAnimalInfo, FrankFilenameParser, FrankDataInfo

import multiprocessing

import cloudpickle
        
# hv.extension('matplotlib')
# hv.extension('bokeh')
#pd.set_option('float_format', '{:,.2f}'.format)
pd.set_option('display.precision', 4)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 15)
#pd.set_option('display.width', 80)

idx = pd.IndexSlice


def run_alg():
    # Load merged rec HDF store based on config

    #config_file = '/opt/data36/daliu/realtime/spykshrk/ripple_dec/bond.config.json'
    #config_file = '/opt/data36/daliu/realtime/spykshrk/dec_60uv/bond.config.json'
    #config_file = '/home/daliu/Src/spykshrk_realtime/config/bond_single.json'
    config_file = '/g/g20/liu67/Src/spykshrk_realtime/config/bond_hpc.json'
    config = json.load(open(config_file, 'r'))

    out_dir = config['files']['output_dir']

    # Tetrode subset to use for small tests: 5, 11, 12, 14, 19
    #config['simulator']['nspike_animal_info']['tetrodes'] = [1, 2, 4, 5, 7, 10, 11, 12, 13, 14, 17, 18,
    #                                                         19, 20, 22, 23, 27, 29]
    config['simulator']['nspike_animal_info']['tetrodes'] = [5, 11, 12, 14, 19]

    day = config['simulator']['nspike_animal_info']['days'][0]
    epoch = config['simulator']['nspike_animal_info']['epochs'][0]
    time_bin_size = config['pp_decoder']['bin_size']

    # Change config
    sim_num = 3
    config['encoder']['position_kernel']['std'] = 1
    config['pp_decoder']['trans_mat_smoother_std'] = 2
    config['pp_decoder']['trans_mat_uniform_gain'] = 0.01
    config['encoder']['mark_kernel']['std'] = 10
    config['encoder']['spk_amp'] = 100
    config['encoder']['vel'] = 2

    # Extract just encode and decode settings from config
    encode_settings = EncodeSettings(config)
    decode_settings = DecodeSettings(config)

    # Grab animal linearized real position
    nspike_anim = nspike_data.AnimalInfo(**config['simulator']['nspike_animal_info'])
    pos = nspike_data.PosMatDataStream(nspike_anim)
    pos_data = pos.data

    spk = nspike_data.SpkDataStream(nspike_anim)
    spk_data = SpikeWaves.from_df(spk.data, encode_settings)

    # spk threshold
    encode_settings.spk_amp = 60

    # Encapsulate linear position
    lin_obj = LinearPosition.from_nspike_posmat(pos_data, encode_settings)
    linflat_obj = lin_obj.get_mapped_single_axis()

    spk_amp = spk_data.max(axis=1)
    spk_amp = spk_amp.to_frame().pivot_table(index=['day','epoch','elec_grp_id','timestamp','time'], 
                                             columns='channel', values=0)
    spk_amp = SpikeFeatures.create_default(df=spk_amp, sampling_rate=30000)
    spk_amp_thresh = spk_amp.get_above_threshold(encode_settings.spk_amp)


    linflat_spkindex = linflat_obj.get_irregular_resampled(spk_amp_thresh)
    linflat_spkindex_encode_velthresh = linflat_spkindex.query('abs(linvel_flat) >= @encode_settings.vel')
    linflat_spkindex_decode_velthresh = linflat_spkindex
       
    spk_amp_thresh_index_match = spk_amp_thresh

    spk_amp_thresh_encode = spk_amp_thresh_index_match.loc[linflat_spkindex_encode_velthresh.index.get_values()]
    #spk_amp_thresh_encode.set_index( 'elec_grp_id', append=True, inplace=True)
    #spk_amp_thresh_encode = spk_amp_thresh_encode.reorder_levels(['day', 'epoch', 'elec_grp_id' , 'timestamp', 'time'])
    spk_amp_thresh_encode.sort_index(inplace=True)

    spk_amp_thresh_decode = spk_amp_thresh_index_match.loc[linflat_spkindex_decode_velthresh.index.get_values()]
    #spk_amp_thresh_decode.set_index( 'elec_grp_id', append=True, inplace=True)
    #spk_mp_thresh_decode = spk_amp_thresh_decode.reorder_levels(['day', 'epoch', 'elec_grp_id' , 'timestamp', 'time'])
    spk_amp_thresh_decode.sort_index(inplace=True)


    encoder = OfflinePPEncoder(linflat=linflat_obj, enc_spk_amp=spk_amp_thresh_encode, 
                               dec_spk_amp=spk_amp_thresh_decode, encode_settings=encode_settings, 
                               decode_settings=decode_settings, chunk_size=5000, cuda=True)

    observ_obj = encoder.run_encoder()



    # Run PP decoding algorithm
    time_bin_size = 30
    #decoder = OfflinePPDecoder(observ_obj=observ_obj, trans_mat=encoder.trans_mat['simple'], 
    #                           prob_no_spike=encoder.prob_no_spike,
    #                           encode_settings=encode_settings, decode_settings=decode_settings, 
    #                           time_bin_size=time_bin_size)

    #posteriors = decoder.run_decoder()


    #posteriors._to_hdf_store('/opt/data36/daliu/pyBond/analysis/bond_decode_example.h5','/analysis', 
    #                         'example01/bond/decode/clusterless/offline/day04/epoch01/', 
    #                         'decode_sim'+str(sim_num), overwrite=True)

    return observ_obj

time_start = time.time()
run_alg()
time_end = time.time()
print('Run time:', time_end - time_start)
