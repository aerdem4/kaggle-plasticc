import numpy as np


def safe_log(x):
    return np.log(np.clip(x, 1e-4, None))


def get_hostgal_range(hostgal_photoz):
    return np.clip(hostgal_photoz//0.2, 0, 6)
