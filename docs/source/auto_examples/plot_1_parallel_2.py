"""
Aligned Parallel Plots
======================

"""

from dewloosh.mpl import aligned_parallel
import numpy as np

labels = ['a', 'b', 'c']
values = np.array([np.random.rand(150) for _ in labels]).T
datapos = np.linspace(-1, 1, 150)
aligned_parallel(values, datapos, labels=labels, yticks=[-1, 1], y=0.8,
                 xticksrotation=0, vlines=[0., 1.], figsize=(7, 5))
