"""
Basic Examples
==============

"""

from polymesh.grid import grid
from polymesh.topo.tr import Q4_to_T3
from polymesh.tri.trimesh import triangulate
from dewloosh.mpl import triplot
import numpy as np
gridparams = {
    'size' : (1200, 600),
    'shape' : (30, 15),
    'eshape' : (2, 2),
    'origo' : (0, 0),
    'start' : 0
}
coordsQ4, topoQ4 = grid(**gridparams)
points, triangles = Q4_to_T3(coordsQ4, topoQ4, path='grid')
triobj = triangulate(points=points[:, :2], triangles=triangles)[-1]

# %%
from dewloosh.mpl import parallel
colors = np.random.rand(150, 3)
labels = [str(i) for i in range(10)]
values = [np.random.rand(150) for i in range(10)]
parallel(values, labels=labels, padding=0.05, lw=0.2,
         colors=colors, title='Parallel Plot with Random Data')
