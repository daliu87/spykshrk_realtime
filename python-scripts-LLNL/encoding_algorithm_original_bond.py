
# coding: utf-8

# In[1]:

import sys
sys.path.insert(0,'/home/mcoulter/spykshrk_hpc/')


# In[3]:


import pandas as pd
import numpy as np
import scipy as sp
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
import json
import os
import scipy.signal
import functools
#import holoviews as hv

from spykshrk.util import AttrDict
import spykshrk.franklab.filterframework_util as ff_util

from spykshrk.realtime.simulator import nspike_data

from spykshrk.franklab.pp_decoder.util import gaussian, normal2D, apply_no_anim_boundary, simplify_pos_pandas,                                                 normal_pdf_int_lookup
from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPDecoder, OfflinePPEncoder
from spykshrk.franklab.data_containers import DataFrameClass, EncodeSettings, DecodeSettings, SpikeObservation,                                               LinearPosition, StimLockout, Posteriors,                                               FlatLinearPosition, SpikeWaves, SpikeFeatures,                                               pos_col_format, DayEpochTimeSeries

from spykshrk.franklab.pp_decoder.visualization import DecodeVisualizer
from spykshrk.franklab.pp_decoder.decode_error import LinearDecodeError

import dask
import dask.dataframe as dd
import dask.array as da
import multiprocessing

import cloudpickle
        
#get_ipython().run_line_magic('load_ext', 'Cython')

#get_ipython().run_line_magic('matplotlib', 'inline')

#hv.extension('matplotlib')
#hv.extension('bokeh')
#pd.set_option('float_format', '{:,.2f}'.format)
pd.set_option('display.precision', 4)
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 15)
#pd.set_option('display.width', 80)

idx = pd.IndexSlice

matplotlib.rcParams.update({'font.size': 14})


# In[3]:


#from holoviews import Store
#from bokeh.models.arrow_heads import TeeHead
#Store.add_style_opts(hv.Curve, ['linestyle'], backend='matplotlib')


# In[4]:

dask.config.set(scheduler='processes')

#try:
#    cluster.close()
#    client.close()
#except:
#    print("No cluster or client running")
#    
#from dask.distributed import Client, LocalCluster
#
#cluster = LocalCluster(n_workers=20, threads_per_worker=2)
#client = Client(cluster)

min_worker_memory = 2**30
#for w in cluster.workers:
#    min_worker_memory = min(min_worker_memory, w.memory_limit)


#"""dask.set_options(get=dask.multiprocessing.get, pool=multiprocessing.pool.Pool(20))
#min_worker_memory = 10e9
#"""


# In[5]:


# Load merged rec HDF store based on config

#config_file = '/opt/data36/daliu/realtime/spykshrk/ripple_dec/bond.config.json'
#config_file = '/opt/data36/daliu/realtime/spykshrk/dec_60uv/bond.config.json'
config_file = '/home/mcoulter/spykshrk_hpc/config/bond_single.json'
config = json.load(open(config_file, 'r'))

#config['simulator']['nspike_animal_info']['tetrodes'] = [1, 2, 4, 5, 7, 10, 11, 12, 13, 14, 17, 18, 19, 20, 22, 23, 27, 29]
config['simulator']['nspike_animal_info']['tetrodes'] = [5, 11, 12, 14, 19]

day = config['simulator']['nspike_animal_info']['days'][0]
epoch = config['simulator']['nspike_animal_info']['epochs'][0]
time_bin_size = config['pp_decoder']['bin_size']

# Change config
config['encoder']['position_kernel']['std'] = 1
config['pp_decoder']['trans_mat_smoother_std'] = 2
config['pp_decoder']['trans_mat_uniform_gain'] = 0.01
config['encoder']['mark_kernel']['std'] = 10

# Extract just encode and decode settings from config
encode_settings = EncodeSettings(config)
decode_settings = DecodeSettings(config)

# Grab animal linearized real position
nspike_anim = nspike_data.AnimalInfo(**config['simulator']['nspike_animal_info'])
pos = nspike_data.PosMatDataStream(nspike_anim)
pos_data = pos.data

spk = nspike_data.SpkDataStream(nspike_anim)
spk_data = SpikeWaves.from_df(spk.data, encode_settings)

# Encapsulate linear position
lin_obj = LinearPosition.from_nspike_posmat(pos_data, encode_settings)
linflat_obj = lin_obj.get_mapped_single_axis()

ripcons = nspike_data.RipplesConsData(nspike_anim)
ripdata = ripcons.data_obj


# In[6]:


config


# In[7]:


spk_amp = spk_data.max(axis=1)
spk_amp = spk_amp.to_frame().pivot_table(index=['day','epoch','elec_grp_id','timestamp','time'], 
                                         columns='channel', values=0)
spk_amp = SpikeFeatures.create_default(df=spk_amp, sampling_rate=30000)
spk_amp_thresh = spk_amp.get_above_threshold(100)


# In[37]:


linflat_spkindex = linflat_obj.get_irregular_resampled(spk_amp_thresh)
linflat_spkindex_encode_velthresh = linflat_spkindex
linflat_spkindex_decode_velthresh = linflat_spkindex

spk_amp_thresh_index_match = spk_amp_thresh.reset_index('elec_grp_id')

spk_amp_thresh_encode = spk_amp_thresh_index_match.loc[linflat_spkindex_encode_velthresh.index]
spk_amp_thresh_encode.set_index( 'elec_grp_id', append=True, inplace=True)
spk_amp_thresh_encode = spk_amp_thresh_encode.reorder_levels(['day', 'epoch', 'elec_grp_id' , 'timestamp', 'time'])
spk_amp_thresh_encode.sort_index(inplace=True)

spk_amp_thresh_decode = spk_amp_thresh_index_match.loc[linflat_spkindex_decode_velthresh.index]
spk_amp_thresh_decode.set_index( 'elec_grp_id', append=True, inplace=True)
spk_amp_thresh_decode = spk_amp_thresh_decode.reorder_levels(['day', 'epoch', 'elec_grp_id' , 'timestamp', 'time'])
spk_amp_thresh_decode.sort_index(inplace=True)


# In[38]:


display(spk_amp_thresh_encode)
display(spk_amp_thresh_decode)


# In[39]:

encoder = OfflinePPEncoder(linflat=linflat_obj, enc_spk_amp=spk_amp_thresh_encode, dec_spk_amp=spk_amp_thresh_decode,
                           encode_settings=encode_settings, decode_settings=decode_settings,
                           dask_worker_memory=min_worker_memory)
#task = encoder.setup_encoder_dask()
results = encoder.run_encoder()

#get_ipython().run_cell_magic('time', '', '#%%prun -r -s cumulative\n\nencoder = OfflinePPEncoder(linflat=linflat_obj, enc_spk_amp=spk_amp_thresh_encode, dec_spk_amp=spk_amp_thresh_decode,\n                           encode_settings=encode_settings, decode_settings=decode_settings,\n                           dask_worker_memory=min_worker_memory)\n#task = encoder.setup_encoder_dask()\nresults = encoder.run_encoder()')


# In[40]:


#%%time
tet_ids = np.unique(spk_amp.index.get_level_values('elec_grp_id'))
observ_tet_list = []
grp = spk_amp_thresh_decode.groupby('elec_grp_id')
for tet_ii, (tet_id, grp_spk) in enumerate(grp):
    tet_result = results[tet_ii]
    tet_result.set_index(grp_spk.index, inplace=True)
    observ_tet_list.append(tet_result)

observ = pd.concat(observ_tet_list)
observ_obj = SpikeObservation.create_default(observ.sort_index(level=['day', 'epoch', 
                                                                      'timestamp', 'elec_grp_id']), 
                                             encode_settings)

observ_obj['elec_grp_id'] = observ_obj.index.get_level_values('elec_grp_id')
observ_obj.index = observ_obj.index.droplevel('elec_grp_id')

observ_obj['position'] = (lin_obj.get_irregular_resampled(observ_obj).
                          get_mapped_single_axis()['linpos_flat'])


# In[41]:

observ_obj.loc[:, 'x000':'x449'] = observ_obj.loc[:, 'x000':'x449'].values + 1e-20

#get_ipython().run_cell_magic('time', '', "\n# Add small offset to observation distributions to prevent numerical instability due to precision\nobserv_obj.loc[:, 'x000':'x449'] = observ_obj.loc[:, 'x000':'x449'].values + 1e-20")


# In[42]:


time_bin_size = 30

decoder = OfflinePPDecoder(observ_obj=observ_obj, trans_mat=encoder.trans_mat['simple'], 
                           prob_no_spike=encoder.prob_no_spike,
                           encode_settings=encode_settings, decode_settings=decode_settings, 
                           time_bin_size=time_bin_size)

posteriors = decoder.run_decoder()

#get_ipython().run_cell_magic('time', '', "# Run PP decoding algorithm\ntime_bin_size = 30\n\ndecoder = OfflinePPDecoder(observ_obj=observ_obj, trans_mat=encoder.trans_mat['simple'], \n                           prob_no_spike=encoder.prob_no_spike,\n                           encode_settings=encode_settings, decode_settings=decode_settings, \n                           time_bin_size=time_bin_size)\n\nposteriors = decoder.run_decoder()")


# In[30]:


import os
os.path.join('/analysis', 'decode/clusterless/offline/posterior', 'run1')


# In[31]:


posteriors._to_hdf_store('/data2/mcoulter/posteriors/remy_20_4_livermore_decode.h5','/analysis', 
                         'decode/clusterless/offline/posterior', 'simple_trans_mat', overwrite=False)


# In[32]:


#test1 = Posteriors._from_hdf_store('bond_decode.h5','/analysis',
#                                   'decode/clusterless/offline/posterior', 'run1')


# In[17]:


posteriors.memory_usage()[0]/1e6


# In[18]:


#hv.Image(encoder.trans_mat['simple'])


# In[33]:


#get_ipython().run_cell_magic('output', "backend='matplotlib' size=300", '%%opts Points (s=200 marker=\'^\' )\n%%opts Curve [aspect=3]\n%%opts Text (text_align=\'left\')\n\nsel_distrib = observ_obj.loc[:, pos_col_format(0,encode_settings.pos_num_bins):         \n                             pos_col_format(encode_settings.pos_num_bins-1,\n                                            encode_settings.pos_num_bins)]\n    \nsel_pos = observ_obj.loc[:, \'position\']\n\nmax_prob = sel_distrib.max().max()/2\n\ndef plot_observ(big_bin, small_bin):\n    bin_id = small_bin + 10000 * big_bin\n    spks_in_bin = sel_distrib.loc[observ_obj[\'dec_bin\'] == bin_id, :]\n    pos_in_bin = sel_pos.loc[observ_obj[\'dec_bin\'] == bin_id, :]\n    \n    num_spks = len(spks_in_bin)\n    plot_list = []\n    if num_spks == 0:\n        plot_list.append(hv.Curve((0,[max_prob-0.01]), \n                                   extents=(0, 0, encode_settings.pos_bins[-1], max_prob)))\n    for spk_observ, pos_observ in zip(spks_in_bin.values, pos_in_bin.values):\n        plot_list.append(hv.Curve(spk_observ, \n                                  extents=(0, 0, encode_settings.pos_bins[-1], max_prob)))\n\n        plot_list.append(hv.Points((pos_observ, [max_prob-0.01])))\n    return hv.Overlay(plot_list) * hv.Text(50,max_prob-0.05, "num_spks: {num_spks}\\n"\n                                           "Timestamp: {timestamp}\\nTime: {time}".\n                                           format(num_spks=num_spks, timestamp=time_bin_size*bin_id,\n                                                  time=time_bin_size*bin_id/30000))\n\n#Ind = Stream.define(\'stuff\', ind=0)\n\ndmap = hv.DynamicMap(plot_observ, kdims=[\'big_bin\', \'small_bin\'], label="test")\n#dmap = hv.DynamicMap(plot_observ, kdims=\n#                     [hv.Dimension(\'bin_id\', range=(0, observ_obj[\'dec_bin\'].iloc[-1]), step=1)])\n#dmap = hv.DynamicMap(plot_observ, kdims=\n#                     [hv.Dimension(\'bin_id\', values=observ_obj[\'dec_bin\'].unique())])\n\n#dmap.redim.values(bin_id=range(0, observ_obj[\'dec_bin\'].iloc[-1]))\ndmap.redim.range(small_bin=(0, 1000), big_bin=(0, observ_obj[\'dec_bin\'].iloc[-1]/1000 + 1))\n#dmap.redim.range(bin_id=(0, observ_obj[\'dec_bin\'].iloc[-1]))\n#dmap.redim.values(bin_id=[4,5])')


# In[34]:


lin_obj.get_mapped_single_axis()


# In[43]:


#get_ipython().run_cell_magic('output', "backend='bokeh' size=400 holomap='scrubber'", "%%opts RGB { +framewise} [height=100 width=250 aspect=2 colorbar=True]\n%%opts Points [height=100 width=250 aspect=2 ] (marker='o' color='#AAAAFF' size=2 alpha=0.7)\n%%opts Polygons (color='grey', alpha=0.5 fill_color='grey' fill_alpha=0.5)\n#%%opts Image {+framewise}\ndec_viz = DecodeVisualizer(posteriors, linpos=linflat_obj, riptimes=ripdata, enc_settings=encode_settings)\n\ndec_viz.plot_all_dynamic(stream=hv.streams.RangeXY(), plt_range=1, slide=1, values=ripdata['starttime']-.5)")


# In[22]:


#get_ipython().run_cell_magic('opts', 'NdLayout [shared_axes=False]', '%%output size=100\n\ndmap = dec_viz.plot_ripple_dynamic()\n\nplot_list = []\nplt_grp_size = 12\nplt_grps = range(math.ceil(ripdata.get_num_events()/plt_grp_size))\nplt_range_low = np.array(plt_grps) * plt_grp_size\nplt_range_high = np.append(plt_range_low[0:-1] + plt_grp_size, ripdata.get_num_events())\n\nfor plt_grp, ind_low, ind_high in zip(plt_grps, plt_range_low, plt_range_high):\n    plot_list.append(hv.NdLayout(dmap[set(range(ind_low, ind_high))]).cols(3))\n\n\n#for plt_grp in plt_grps\n#hv.NdLayout(dmap[set(range(ripdata.get_num_events()))]).cols(3)')


# In[23]:


#get_ipython().run_cell_magic('opts', 'Image {+axiswise} [height=300 width=300 aspect=3]', "%%opts Curve {+axiswise} [aspect=2] (line_dash='dashed' color='#AAAAAA' linestyle='--' alpha=0.5)\n%%opts Points {+axiswise} [aspect=2] (marker='*' size=14)\n%%opts NdLayout {+axiswise}\n%%output backend='matplotlib' size=600\n\nevent_ids = ripdata.find_events([2585.42, 2791, 2938.2, 3180.2, 3263.40, 3337.4])\nplt = hv.Layout()\nfor id in event_ids:\n    plt += dec_viz.plot_ripple_all(id)\n\nplt.cols(1)")


# In[24]:


#get_ipython().run_cell_magic('opts', 'Image {+axiswise} [height=300 width=300 aspect=1]', "%%opts Curve.arm_bound {+axiswise} [aspect=1] (line_dash='dashed' color='#AAAAAA' linestyle='--' alpha=0.5)\n%%opts Points {+axiswise} [aspect=1] (marker='*' size=14)\n%%opts NdLayout {+axiswise}\n%%output backend='matplotlib' size=200\n\ndec_viz.plot_ripple_all(2)")


# In[25]:


#linflat_obj['ripple_grp'].unique()


# In[26]:


#get_ipython().run_cell_magic('opts', 'Image {+axiswise} [height=300 width=300 aspect=1]', "%%opts Curve {+axiswise} [aspect=1] (line_dash='dashed' color='#AAAAAA' linestyle='--' alpha=0.5)\n%%opts Points {+axiswise} [aspect=1] (marker='*' size=14)\n%%opts NdLayout {+axiswise}\n%%output backend='matplotlib' size=200\n\ndec_viz = DecodeVisualizer(posteriors, linpos=linflat_obj, riptimes=ripdata.get_above_maxthresh(5), enc_settings=encode_settings)\n\nrip_plots = dec_viz.plot_ripple_grid(2)\nfor plt_grp in rip_plots:\n    display(plt_grp)")


# In[27]:


#get_ipython().run_cell_magic('output', 'size=300', 'dec_viz.plot_ripple_all(242)')


# In[28]:


#np.append(plt_range_high, [270])

