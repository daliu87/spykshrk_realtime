import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def normal2D(x, y, sig):
    return np.exp(-(np.power(x, 2.) + np.power(y, 2.)) / (2 * np.power(sig, 2.)))


def apply_no_anim_boundary(x_bins, arm_coor, image):
    """
    
    Args:
        x_bins: the position value for each bin
        arm_coor: the arm inclusive arm coordinates of valid animal positions
        image: the image or array to apply

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
            image[bounds[0]:bounds[1]] = 0
        elif image.ndim == 2:
            image[bounds[0]:bounds[1], :] = 0
            image[:, bounds[0]:bounds[1]] = 0
    return image

