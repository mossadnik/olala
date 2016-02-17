
import numpy as np
from .utils import checkVec2d


rectOffsets = {pos: delta for pos, delta in 
               zip(['bl', 'b', 'br', 'l', 'c', 'r', 'tl', 't', 'tr'],
               	   [np.array([i, j]) for j in [0, .5, 1.] for i in [0, .5, 1.]])
               }


def connectionPositionOffset(s, w):
    try:
        factor = rectOffsets[s]
    except ValueError:
        print "unknown connection position '%s'" % s
    return w * factor


def getLabelBoxes(ax, labels):
    '''
    return positions and widths of bboxes of labels in axis coordinates
    '''
    renderer = ax.figure.canvas.get_renderer()
    x, w = [], []
    for label in labels:
        label.draw(renderer)
        box = label.get_window_extent().inverse_transformed(ax.transAxes)
        x.append((box.x0, box.y0))
        w.append((box.width, box.height))
    x = np.array(x)
    w = np.array(w)
    return x, w


def placeLabels(ax, labels, x, w=None, arrows=None, connectionPosition='bl'):
    '''
    place labels in position x (axis coordinates). Origins of
    optional arrows are placed at connectionPosition of each label bbox
    '''
    invTransData = ax.transData.inverted()
    transAxes = ax.transAxes
    for i in range(len(labels)):
        label = labels[i]
        xy = x[i]
        pos = invTransData.transform(transAxes.transform(xy))
        label.set_position(pos)
        if arrows:
            dxy = connectionPositionOffset(connectionPosition, w[i])
            pos = invTransData.transform(transAxes.transform(xy + dxy))
            arrow = arrows[i]
            arrow.set_position(pos)


def applyLayout(ax, labels, pad=.01,
                layoutFunc=None, layoutArgs={},
                xlim=(0, 1), ylim=(0, 1),
                arrows=None, connectionPosition='bl'):
    '''
    update label positions according to layout algorithm.
    If arrows is given, arrow origins will be moved to respective labels.
    '''
    def getScaleAndOffset(xlim, ylim):
        lim = (xlim, ylim)
        scale = np.array([l[1] - l[0] for l in lim], dtype=float)[np.newaxis, :]
        offset = np.array([l[0] for l in lim], dtype=float)[np.newaxis, :]
        return scale, offset

    def toStandardBounds(x, w, xlim, ylim):
        '''
        transform bounds to standard bounds (0, 1) x (0, 1)
        '''
        scale, offset = getScaleAndOffset(xlim, ylim)
        return (x - offset) / scale, w / scale

    def fromStandardBounds(x, w, xlim, ylim):
        scale, offset = getScaleAndOffset(xlim, ylim)
        return scale * x + offset, scale * w

    pad = checkVec2d(pad)
    x0, w = toStandardBounds(*getLabelBoxes(ax, labels), xlim=xlim, ylim=ylim)
    x, w = fromStandardBounds(layoutFunc(x0, w + pad, **layoutArgs),
                              w, xlim, ylim)
    placeLabels(ax, labels, x, w, arrows, connectionPosition)
