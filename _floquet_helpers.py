import scipy.linalg
import numpy.matrix
import numpy as np
import scipy


def compute_distance(propagator1, propagator2):
    ret = np.dot(propagator1-propagator2, (propagator1-propagator2).T.conj())
    return numpy.matrix.trace(ret)

def compute_propagator(matrix, endtime, dt):
    return False