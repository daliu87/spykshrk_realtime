import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def normal2D(x, y, sig):
    return np.exp(-(np.power(x, 2.) + np.power(y, 2.)) / (2 * np.power(sig, 2.)))


def apply_no_anim_boundary(x_bins, bounds, image):
    # no-animal boundary
    boundary_ind = np.searchsorted(x_bins, bounds, side='right')
    boundary_ind = np.reshape(boundary_ind, [3,2])

    for bounds in boundary_ind:
        if image.ndim == 1:
            image[bounds[0]:bounds[1]] = 0
        elif image.ndim == 2:
            image[bounds[0]:bounds[1], :] = 0
            image[:, bounds[0]:bounds[1]] = 0
    return image