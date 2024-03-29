import numpy as np
from functools import partial

from dewloosh.core.typing import issequence

from neumann.utils import minmax

from polymesh.triang import triobj_to_mpl, get_triobj_data, triangulate
from polymesh.utils.tri import offset_tri
from polymesh.utils import cells_coords, explode_mesh_data_bulk

from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import PatchCollection
import matplotlib.tri as mpltri

from .utils import get_fig_axes, decorate_ax


__all__ = ['triplot']


def triplot(triobj, *args, hinton=False, data=None, title=None,
            label=None, fig=None, ax=None, axes=None, fig_kw: dict = None,
            **kwargs):
    """
    Creates plots over triangulations using `matplotlib`.

    Parameters
    ----------
    triobj : TriMeshLike
        This is either a tuple of mesh data (coordinates and topology)
        or a triangulation object understood by `dewloosh.tri.trimesh.triangulate`.
    hinton : bool, Optional
        Creates a hinton-like plot. Only wors if the provided data is along
        the cells. Default is False.
    data : `numpy.ndarray`, Optional 
        Some data to plot as an 1d or 2d numpy array. Default is None.
    title : str, Optional
        Title of the plot. See `matplotlib` for further details.
        Default is None.
    label : str, Optional
        Title of the plot. See `matplotlib` for further details.
        Default is None.
    fig : `matplotlib.Figure`, Optional
        A `matplotlib` figure to plot on. Default is None.
    ax : `matplotlib.axes.Axes` or a collection of it, Optional.
        A `matplotlib` axis, or a collection of such objects to plot on. 
        Default is None.
    kwargs : dict, Optional
        The following keyword arguments are understood and forwarded to the 
        appropriate function in `matplotlib`:

        'cmap' - colormap (if `data` is provided)   
        'lw' 
        'xlim' 
        'ylim' 
        'axis' 
        'suptitle'

    fig_kw : dict, Optional
        If there is no figure instance provided, these parameters are 
        forwarded to the ``matplotlib.pyplot.figure`` call.

    Examples
    --------
    Create a triangulation

    >>> from polymesh.grid import grid
    >>> from polymesh.topo.tr import Q4_to_T3
    >>> from polymesh.tri.trimesh import triangulate
    >>> from dewloosh.mpl import triplot
    >>> import numpy as np
    >>> gridparams = {
    >>>     'size' : (1200, 600),
    >>>     'shape' : (30, 15),
    >>>     'eshape' : (2, 2),
    >>>     'origo' : (0, 0),
    >>>     'start' : 0
    >>> }
    >>> coordsQ4, topoQ4 = grid(**gridparams)
    >>> points, triangles = Q4_to_T3(coordsQ4, topoQ4, path='grid')
    >>> triobj = triangulate(points=points[:, :2], triangles=triangles)[-1]

    If you just want to plot the mesh itself, do this

    >>> triplot(triobj)

    Plot the mesh with random data over the cells

    >>> data = np.random.rand(len(triangles))   
    >>> triplot(triobj, data=data)

    >>> data = np.random.rand(len(triangles))
    >>> triplot(triobj, hinton=True, data=data)

    Plot the mesh with random data over the points

    >>> data = np.random.rand(len(points))
    >>> triplot(triobj, data=data, cmap='bwr')    

    You can play with the arguments sent to ``matplotlib``

    >>> triplot(triobj, data=data, cmap='Set1', axis='off')

    """
    fig, axes = get_fig_axes(*args, data=data, ax=ax, axes=axes,
                             fig=fig, **kwargs)
    if isinstance(triobj, tuple):
        coords, topo = triobj
        triobj = triangulate(points=coords[:, :2], triangles=topo)[-1]
        coords, topo = None, None
    pdata = partial(triplot_data, triobj)
    pgeom = partial(triplot_geom, triobj)
    phint = partial(triplot_hinton, triobj)
    if data is not None:
        assert len(data.shape) <= 2, \
            "Data must be a 1 or 2 dimensional array."
        nD = 1 if len(data.shape) == 1 else data.shape[1]
        data = data.reshape((data.shape[0], nD))
        if not issequence(title):
            title = nD * (title, )
        if not issequence(label):
            label = nD * (label, )
        pfnc = phint if hinton else pdata
        axobj = [pfnc(ax, data[:, i], *args, fig=fig, title=title[i],
                      label=label[i], **kwargs)
                 for i, ax in enumerate(axes)]
        if nD == 1:
            data = data.reshape(data.shape[0])
    else:
        axobj = pgeom(axes[0], *args, fig=fig, title=title, **kwargs)
    return axobj


class TriPatchCollection(PatchCollection):

    def __init__(self, cellcoords, *args, **kwargs):
        pmap = map(lambda i: cellcoords[i], np.arange(len(cellcoords)))
        def fnc(points): return Polygon(points, closed=True)
        patches = list(map(fnc, pmap))
        super().__init__(patches, *args, **kwargs)


def triplot_hinton(triobj, ax, data, *args, lw=0.5, fcolor='b',
                   ecolor='k', title=None, suptitle=None, label=None,
                   **kwargs):
    axobj = None
    tri = triobj_to_mpl(triobj)
    points, triangles = get_triobj_data(tri, *args, trim2d=True, **kwargs)
    cellcoords = offset_tri(points, triangles, data)
    axobj = TriPatchCollection(cellcoords, fc=fcolor, ec=ecolor, lw=lw)
    ax.add_collection(axobj)
    decorate_ax(ax=ax, points=points, title=title, suptitle=suptitle,
                label=label, **kwargs)
    return axobj


def triplot_geom(triobj, ax, *args, lw=0.5, marker='b-',
                 zorder=None, fcolor=None, ecolor='k',
                 fig=None, title=None, suptitle=None, label=None,
                 **kwargs):
    axobj = None
    tri = triobj_to_mpl(triobj)
    points, triangles = get_triobj_data(tri, trim2d=True)

    if fcolor is None:
        if zorder is not None:
            axobj = ax.triplot(tri, marker, lw=lw, zorder=zorder, **kwargs)
        else:
            axobj = ax.triplot(tri, marker, lw=lw, **kwargs)
    else:
        cellcoords = cells_coords(points, triangles)
        axobj = TriPatchCollection(cellcoords, fc=fcolor, ec=ecolor, lw=lw)
        ax.add_collection(axobj)
    decorate_ax(fig=fig, ax=ax, points=points, title=title,
                suptitle=suptitle, label=label, **kwargs)
    return axobj


def triplot_data(triobj, ax, data, *args, cmap='winter', fig=None,
                 ecolor='k', lw=0.1, title=None, suptitle=None, label=None,
                 nlevels=None, refine=False, refiner=None, colorbar=True,
                 subdiv=3, cbpad="2%", cbsize="5%", cbpos='right', **kwargs):
    """
    Plots data over a triangulation using `matplotlib`. The provided data may be
    aligned with the points, or it can be defined over the cells. The shape of the
    data governs the behaviour of the plot.
    """

    axobj = None
    tri = triobj_to_mpl(triobj)
    points, triangles = get_triobj_data(tri, trim2d=True)

    if refiner is not None:
        refine = True

    nData = len(data)
    if nData == len(triangles):
        nD = len(data.shape)
        if nD == 1:
            axobj = ax.tripcolor(tri, facecolors=data, cmap=cmap,
                                 edgecolors=ecolor, lw=lw)
        elif nD == 2 and data.shape[1] == 3:
            points, triangles, data = \
                explode_mesh_data_bulk(points, triangles, data)
            triobj = triangulate(points=points, triangles=triangles)[-1]
            tri = triobj_to_mpl(triobj)
            axobj = ax.tripcolor(tri, data, cmap=cmap,
                                 edgecolors=ecolor, lw=lw)
    elif nData == len(points):
        if refine:
            if refiner is None:
                refiner = mpltri.UniformTriRefiner(triobj)
            tri, data = refiner.refine_field(data, subdiv=subdiv)
        if isinstance(nlevels, int):
            dmin, dmax = minmax(data)
            levels = np.linspace(dmin, dmax, nlevels+1)
            axobj = ax.tricontourf(tri, data, levels=levels, cmap=cmap)
            ax.tricontour(tri, data, levels=levels)
        else:
            axobj = ax.tripcolor(tri, data, cmap=cmap,
                                 edgecolors=ecolor, lw=lw)

    assert axobj is not None, "Failed to handle the provided data."
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbpos, size=cbsize, pad=cbpad)
        cbar = fig.colorbar(axobj, cax=cax)
        dmin, dmax = minmax(data)
        cbar.ax.set_yticks([dmin, dmax])
        cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in [dmin, dmax]]) 

    decorate_ax(fig=fig, ax=ax, points=points, title=title,
                suptitle=suptitle, label=label, **kwargs)
    return axobj
