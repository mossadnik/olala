
import numpy as np
from scipy.optimize import minimize

from ..utils import checkVec2d


def greedyRowAssignment(x0, w, minRows=None, maxFilling=.7):
    '''
    Compute greedy row assignment for sequence of intervals
    '''
    if minRows is None:
        minRows = 0
    nRows = max(minRows, int(np.ceil(w.sum() / maxFilling)))

    right = np.zeros(nRows)
    row = np.zeros(x0.size, dtype=np.int)
    pos = np.zeros(x0.size)
    fill = np.zeros_like(right)
    for i, (xNew, wNew) in enumerate(zip(x0, w)):
        newPos = np.maximum(xNew, right)  # new position for each row
        feasible = fill + wNew < 1.
        minRight = np.min(newPos[feasible])  # assign to best row greedily
        optimal = newPos == minRight
        r = np.where(feasible & optimal)[0][0]  # assign to lowest row 
        row[i] = r
        pos[i] = newPos[r]
        right[r] = pos[i] + wNew
        fill[r] += wNew
    return row, pos

def rowOptimization(x0, w, c, verbose=False):
    def energy(x, c):
        return np.sum((x - c)**2)
    
    def overlapConstraint(x, w, i):
        return x[i] - (x[i - 1] + w[i - 1])
    
    n = x0.size    
    # no-overlap constraint
    constraints = [dict(type='ineq', fun=overlapConstraint, args=(w, i)) for i in range(1, n)]
    # bounds
    constraints += [dict(type='ineq', fun=lambda x: x[0]),
                   dict(type='ineq', fun=lambda x, w: 1 - (x[-1] + w[-1]), args=(w,))]

    res = minimize(energy, x0, args=(c,), method='slsqp', constraints=constraints)
    if verbose:
        print res
    return res['x']


def rowLayout(x0, w, minRows=None, maxFilling=.7):
    row, xr = greedyRowAssignment(x0[:, 0], w[:, 0], minRows=minRows, maxFilling=maxFilling)
    nRows = row.max() + 1
    x = np.zeros((row.size, 2))
    h = 0
    for r in range(nRows):
        ix = row == r
        x[ix, 0] = rowOptimization(xr[ix], w[ix, 0], x0[ix, 0])
        x[ix, 1] = h
        h += w[ix, 1].max()
    return x