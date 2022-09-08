# -*- coding: utf-8 -*-
from typing import Iterable, TypeVar, Hashable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import numpy as np

from linkeddeepdict import LinkedDeepDict
from dewloosh.core.tools import float_to_str_sig as str_sig

__all__ = ['parallel', 'aligned_parallel']


TScalar = TypeVar('TScalar', int, float, complex)
TReal = TypeVar('TReal', int, float)
TVector = Iterable[TScalar]
TRealVector = Iterable[TReal]
TColor = TypeVar('TColor', str, TRealVector)


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


def aligned_parallel(data, datapos, *args, yticks=None, labels=None,
                     sharelimits=False, texlabels=None, xticksrotation=0,
                     suptitle=None, slider=False, slider_label=None, 
                     hlines=None, vlines=None, y=None, **kwargs):
    """
    Parameters
    ----------
    data : numpy.ndarray or dict
        The values to plot. If it is a numpy array, labels must be provided
        with the argument `labels`, if it is a sictionary, the keys of the 
        dictionary are used as labels.
        
    datapos : Iterable
        Positions of the provided data values.
        
    yticks : Iterable, Optional
        Positions of ticks on the vertical axes. Default is None.
        
    labels : Iterable, Optional
        An iterable of strings specifying labels for the datasets.
        Default is None.

    sharelimits : bool, Optional
        If True, the axes share limits of the vertical axes.
        Default is False.

    texlabels : Itrable, Optional
        TeX-formatted labels. If provided, it must have the same length as
        `labels`. Default is None.
        
    xticksrotation : int, Optional
        Rotation of the ticks along the vertical axes. Expected in degrees. 
        Default is 0.
        
    suptitle : str, Optional
        See Matplotlib's docs for the details. Default is None.
        
    slider : bool, Optional
        If True, a slider is added to the figure for interactive plots.
        Default is False.
        
    slider_label : str, Optional
        A label for the slider. Only if `slider` is true. Default is None.
        
    hlines : Iterable, Optional
        A list of data values where horizontal lines are to be added to the axes.
        Default is None.
        
    vlines : Iterable, Optional
        A list of data values where vertical lines are to be added to the axes.
        Default is None.
        
    y : float or int, Optional
        Value for the vertical axis. Default is the average of the limits
        of the vertical axis (0.5*(datapos[0] + datapos[-1])).
        
    **kwargs : dict, Optional
        Extra keyword arguments are forwarded to the creator of the matplotlib figure.
        Default is None.
        
    Example
    -------

    >>> from dewloosh.mpl import aligned_parallel
    >>> labels = ['a', 'b', 'c']
    >>> values = np.array([np.random.rand(150) for _ in labels]).T
    >>> datapos = np.linspace(-1, 1, 150)
    >>> aligned_parallel(values, datapos, labels=labels, yticks=[-1, 1])
        
    """
    # init
    fig = plt.figure(**kwargs)
    suptitle = '' if suptitle is None else suptitle
    fig.suptitle(suptitle)
    plotdata = LinkedDeepDict()
    axcolor = 'lightgoldenrodyellow'
    ymin, ymax = np.min(datapos), np.max(datapos)
    if y is None:
        y0 = 0.5 * (ymin + ymax)
    else:
        y0 = y
    hlines = [] if hlines is None else hlines
    vlines = [] if vlines is None else vlines
    
    # init slider
    if slider:
        if slider_label is None:
            slider_label = ''

    # read data
    if isinstance(data, dict):
        if labels is None:
            labels = list(data.keys())
    elif isinstance(data, np.ndarray):
        nData = data.shape[1]
        if labels is None:
            labels = list(map(str, range(nData)))
        data = {labels[i]: data[:, i] for i in range(nData)}
    for lbl in labels:
        plotdata[lbl]['values'] = data[lbl]

    # set min and max values
    _min, _max = [], []
    for lbl in labels:
        plotdata[lbl]['min'] = np.min(data[lbl])
        plotdata[lbl]['max'] = np.max(data[lbl])
        _min.append(plotdata[lbl]['min'])
        _max.append(plotdata[lbl]['max'])
    # set global min and max
    vmin = np.min(_min)
    vmax = np.max(_max)
    del _min
    del _max

    # setting up figure layout
    nData = len(labels)
    if slider:
        nAxes = nData + 1  # +1 for the Slider
    else:
        nAxes = nData
    width_ratios = [1 for i in range(nData)]
    if slider:
        width_ratios.append(0.15)
    spec = gridspec.GridSpec(ncols=nAxes, nrows=1,
                             width_ratios=width_ratios, figure=fig,
                             wspace=0.2, left=0.1)

    # create axes
    for i in range(nData):
        plotid = int("{}{}{}".format(1, nAxes, i+1))
        plotid = spec[0, i]
        ax = fig.add_subplot(plotid, facecolor=axcolor)
        ax.grid(False)
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('0.5')
        if i == 0:
            if yticks is not None:
                ax.set_yticks(yticks)
                ax.set_yticklabels([str_sig(val, sig=3) for val in yticks])
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        hline = ax.axhline(y=y0, color='#d62728', linewidth=1)
        bbox = dict(boxstyle="round", ec='black', fc='yellow', alpha=0.8)
        txt = ax.text(0.0, 0.0, "NaN", size=10, ha="center", va="center",
                      visible=False, bbox=bbox)
        # horizontal lines
        ax.axhline(y=yticks[0], color='black', linewidth=0.5, linestyle='-')
        ax.axhline(y=yticks[-1], color='black', linewidth=0.5, linestyle='-')
        for hl in hlines:
            ax.axhline(y=hl, color='black', linewidth=0.5, linestyle='-')
        # a vertical lines
        for vl in vlines:
            ax.axvline(x=vl, color='black', linewidth=0.5, linestyle='-')
        # store objects
        plotdata[labels[i]]['ax'] = ax
        plotdata[labels[i]]['text'] = txt
        plotdata[labels[i]]['hline'] = hline

    # create slider
    if slider:
        sliderax = fig.add_subplot(spec[0, nAxes-1], fc=axcolor)
        """sliderax.patch.set_edgecolor('black')
        sliderax.patch.set_linewidth('1.0')
        sliderax.patch.zorder = 10"""
        slider_ = Slider(sliderax, slider_label, valmin=ymin, valmax=ymax, valinit=0.0,
                        orientation='vertical', valfmt="%.3f", closedmin=True,
                        closedmax=True) # track_color=axcolor, color=axcolor
        """slider_rect = slider_.track
        s_xy = slider_rect.get_xy() 
        s_w = slider_rect.get_width() 
        s_h = slider_rect.get_height()
        sliderax.add_patch(Rectangle(s_xy, s_w, s_h, fill=False, ec='k'))"""

    def _approx_at_y(y: float, plotkey: Hashable):
        lines2D = plotdata[plotkey]['lines']
        values = lines2D.get_xdata()
        locations = lines2D.get_ydata()
        return np.interp(y, locations, values)

    def _set_yval(y):
        for axkey in plotdata.keys():
            if 'hline' in plotdata[axkey]:
                plotdata[axkey]['hline'].set_ydata(y)
            v_at_y = _approx_at_y(y, axkey)
            plotdata[axkey]['text'].update({'visible': True,
                                            'x': v_at_y, 'y': y,
                                            'text': str_sig(v_at_y)})
        fig.canvas.draw_idle()

    def _update_slider(y=None):
        if y is None:
            y = slider.val
        _set_yval(y)
        
    def _set_xlim(axs: mpl.axes, vmin: float, vmax: float,
                  offset=0.2, **kwargs):
        voffset = (vmax - vmin) * offset
        if abs(vmin - vmax) > 1e-7:
            axs.set_xlim(vmin - voffset, vmax + voffset)
        xticks = [vmin, vmax]
        axs.set_xticks(xticks)
        rotation = kwargs.get('rotation', xticksrotation)
        axs.set_xticklabels([str_sig(val, sig=3) for val in xticks],
                            rotation=rotation)

    def _set_ylim(axs: mpl.axes, vmin: float, vmax: float,
                  offset=0.1, **kwargs):
        voffset = (vmax - vmin)*offset
        axs.set_ylim(vmin - voffset, vmax + voffset)
        
    # plot axes
    for i, axkey in enumerate(plotdata.keys()):
        axis = plotdata[axkey]['ax']
        # set limits
        if sharelimits == True:
            _set_xlim(axis, vmin, vmax)
        else:
            _set_xlim(axis, plotdata[axkey]['min'], plotdata[axkey]['max'])
        _set_ylim(axis, ymin, ymax)
        # set labels
        if texlabels is not None:
            axis.set_title(texlabels[i])
        else:
            axis.set_title(str(axkey))
        # plot
        lines = axis.plot(plotdata[axkey]['values'], datapos, picker=5)[0]
        plotdata[axkey]['lines'] = lines

    # connect events
    if slider:
        slider_.on_changed(_update_slider)
        fig._slider = slider_  # to keep reference, otherwise slider is not responsive
    _set_yval(y0)
    return fig