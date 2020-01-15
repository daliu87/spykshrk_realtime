import numpy as np
import scipy as sp
import scipy.stats
import torch
import cupy
from torch.utils.dlpack import to_dlpack, from_dlpack
from spykshrk.franklab.warnings import DatatypeInconsistentWarning

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def normal2D(x, y, sig):
    return np.exp(-(np.power(x, 2.) + np.power(y, 2.)) / (2 * np.power(sig, 2.)))


def normal_pdf_int_lookup(x, mean, std):
    max_amp = 3000
    norm_dist = sp.stats.norm.pdf(x=np.arange(-max_amp,max_amp), loc=0, scale=std)

    return norm_dist[x-mean+max_amp]


def normal_pdf_int_lookup_torch(x, mean, std, device=torch.device('cpu'), dtype=torch.float):
    max_amp = 3000
    norm_dist = sp.stats.norm.pdf(x=np.arange(-max_amp,max_amp), loc=0, scale=std)
    norm_dist /= norm_dist.max()
    norm_dist_torch = torch.from_numpy(norm_dist).to(device=device, dtype=dtype) 
    x_torch = torch.from_numpy(x).to(device=device)
    mean_torch = torch.from_numpy(mean).to(device=device)
    distance_torch = x_torch - mean_torch
    del x_torch, mean_torch
    distribution_index_torch = distance_torch + max_amp
    del distance_torch
    norm_dist_cu = cupy.fromDlpack(to_dlpack(norm_dist_torch))
    distribution_index_cu = cupy.fromDlpack(to_dlpack(distribution_index_torch))
    lookup_results = from_dlpack(norm_dist_cu[distribution_index_cu].toDlpack())
    return lookup_results


def apply_no_anim_boundary(x_bins, arm_coor, image, fill=0):
    """

    Args:
        x_bins: the position value for each bin
        arm_coor: the inclusive arm coordinates of valid animal positions
        image: the image or array to apply
        fill: value to fill in outside of boundary

    Returns:

    """
    # calculate no-animal boundary
    arm_coor = np.array(arm_coor, dtype='float64')
    arm_coor[:,0] -= x_bins[1] - x_bins[0]
    bounds = np.vstack([[x_bins[-1], 0], arm_coor])
    bounds = np.roll(bounds, -1)

    boundary_ind = np.searchsorted(x_bins, bounds, side='right')
    #boundary_ind[:,1] -= 1

    for bounds in boundary_ind:
        if image.ndim == 1:
            image[bounds[0]:bounds[1]] = fill
        elif image.ndim == 2:
            image[bounds[0]:bounds[1], :] = fill
            image[:, bounds[0]:bounds[1]] = fill
    return image


def simplify_pos_pandas(pos_data):
    pos_data_time = pos_data.index.get_level_values('time')

    pos_data_notebook = pos_data.loc[:,'lin_dist_well']
    pos_data_notebook.loc[:, 'lin_vel_center'] = pos_data.loc[:,('lin_vel', 'well_center')]
    pos_data_notebook.loc[:, 'seg_idx'] = pos_data.loc[:,('seg_idx', 'seg_idx')]
    pos_data_notebook.loc[:,'timestamps'] = pos_data_time*30000
    pos_data_notebook = pos_data_notebook.set_index('timestamps')

    return pos_data_notebook
