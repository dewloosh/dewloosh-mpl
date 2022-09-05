# -*- coding: utf-8 -*-
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import numpy as np

__all__ = ['parallel']


def parallel(data, *args, labels=None, padding=0.05,
             colors=None, lw=0.2, bezier=True, figsize=None,
             title=None, **kwargs):
    """
    Parameters
    ----------
    data : list of arrays
        A list of numpy.ndarray for each column. Each array is 1d with a length of N,
        where N is the number of data records (the number of lines).
        
    labels : Iterable, Optinal
        Labels for the columns. If provided, it must have the same length as `data`.
        
    padding : float, Optional
        Controls the padding around the plot.
        
    colors : list of float, Optional
        A value for each record. Default is None.
        
    lw : float, Optional
        Linewidth.
        
    bezier : bool, Optional
        If True, bezier curves are created instead of a linear polinomials. 
        Default is True.
    
    figsize : tuple, Optional
        A tuple to control the size of the figure. Default is None.
        
    title : str, Optional
        The title of the figure.
    
    Example
    -------
    
    >>> from dewloosh.mpl import parallel
    >>> colors = np.random.rand(150, 3)
    >>> labels = [str(i) for i in range(10)]
    >>> values = [np.random.rand(150) for i in range(10)]
    >>> parallel(values, labels=labels, padding=0.05, lw=0.2,
    >>>         colors=colors, title='Parallel Plot with Random Data')
    
    """

    if isinstance(data, dict):
        if labels is None:
            labels = list(data.keys())
        ys = np.dstack(list(data.values()))[0]
    elif isinstance(data, np.ndarray):
        assert labels is not None
        ys = data
    elif isinstance(data, Iterable):
        assert labels is not None
        ys = np.dstack(data)[0]
    else:
        raise TypeError('Invalid data type!')

    ynames = labels
    N, nY = ys.shape

    if colors is not None:
        pass

    figsize = (7.5, 3) if figsize is None else figsize
    fig, host = plt.subplots(figsize=figsize)

    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * padding
    ymaxs += dys * padding
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    axes = [host] + [host.twinx() for i in range(nY - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (nY - 1)))

    host.set_xlim(0, nY - 1)
    host.set_xticks(range(nY))
    host.set_xticklabels(ynames, fontsize=8)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    if title is not None:
        host.set_title(title, fontsize=12)

    for j in range(N):
        if not bezier:
            # to just draw straight lines between the axes:
            host.plot(range(nY), zs[j, :], c=colors[j])
        else:
            # create bezier curves
            # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
            #   at one third towards the next axis; the first and last axis have one less control vertex
            # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
            # y-coordinate: repeat every point three times, except the first and last only twice
            verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                             np.repeat(zs[j, :], 3)[1:-1]))
            # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
            codes = [Path.MOVETO] + \
                [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = PathPatch(path, facecolor='none',
                              lw=lw, edgecolor=colors[j])
            host.add_patch(patch)
