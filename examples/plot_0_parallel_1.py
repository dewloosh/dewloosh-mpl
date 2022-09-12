"""
Parallel Plots
==============

"""

from dewloosh.mpl import parallel
import numpy as np

colors = np.random.rand(150, 3)
labels = [str(i) for i in range(10)]
values = [np.random.rand(150) for _ in range(10)]
parallel(values, labels=labels, padding=0.05, lw=0.2,
         colors=colors, title='Parallel Plot with Random Data')
