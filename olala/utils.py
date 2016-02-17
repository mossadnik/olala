
import numpy as np

def checkVec2d(a):
    '''
    preprocess 2d-vector specifications.
    accepted inputs are scalars and sequences of length two.
    '''
    result = np.atleast_2d(a)
    assert result.ndim == 2, "too many dimensions for 2d vector: %s" % (str(result.ndim))
    assert result.shape[1] <= 2, "length of vector must be <= 2"
    if result.shape[1] == 1:
        return np.repeat(result, 2, axis=1)
    return result