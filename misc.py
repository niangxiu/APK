from __future__ import division
import numpy as np
from pdb import set_trace


def nanarray(shape):
    _ = np.empty(shape)
    _[:] = np.nan
    return _


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.abs(np.dot(v1_u, v2_u)), -1.0, 1.0))

