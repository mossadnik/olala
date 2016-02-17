import numpy as np
from scipy.optimize import minimize

from ..utils import checkVec2d


def intervalOverlap(x, w):
    '''
    computes the overlap of intervals with left points x and widths w.
    
    returns symmetric matrix of overlaps. Diagonal is set to zero.
    '''
    left = x
    right = x + w
    minRight = np.minimum(right[:, np.newaxis], right[np.newaxis, :])
    maxLeft = np.maximum(left[:, np.newaxis], left[np.newaxis, :])
    overlap = np.maximum(0., minRight - maxLeft) 
    np.fill_diagonal(overlap, 0)
    combinedW = np.minimum(w[:, np.newaxis], w[np.newaxis, :])
    return overlap / combinedW

def rectOverlap(x, w):
    '''
    compute the overlap of rectangles with centers x and dimensions w
    
    returns symmetric matrix of overlap areas. Diagonal is set to zero    
    '''
    ndim = x.shape[1]
    overlaps = np.array([intervalOverlap(x[:, i], w[:, i]) for i in range(ndim)])
    return np.prod(overlaps, axis=0)
    
def springEnergy(x, c):
    '''
    computes the spring potential on x being attached to centers c
    
    returns array of size 2 with components from x/y distances
    '''
    return .5 * np.sum((c - x)**2, axis=0)


from scipy import optimize

def forceLayout(x0, w, springForce=[2., .5], verbose=False):
    springConstant = checkVec2d(springForce)

    def energy(xFlat, w):
        x = xFlat.reshape(x0.shape)
        return (
            .5 * rectOverlap(x, w).sum() + 
            np.dot(springConstant, springEnergy(x, x0))
               )
    xlo = np.zeros_like(x0)
    xhi = np.ones_like(x0) - w
    constraints = ({'type': 'ineq', 'fun': lambda x: x - xlo.ravel()},
                   {'type': 'ineq', 'fun': lambda x: xhi.ravel() - x})
    res = minimize(energy, x0.ravel(), args=(w,), method='cobyla', constraints=constraints,
                           options={"maxiter": 5000})
    if verbose:
        print res
    return res['x'].reshape(x0.shape)