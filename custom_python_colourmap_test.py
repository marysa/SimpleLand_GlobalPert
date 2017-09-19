#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:50:10 2017

@author: mlague
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:32:12 2017

@author: mlague
"""

# In[]:

# For interactive in-line plots:
#%matplotlib nbagg  

# For inline plots:
#%matplotlib inline     

import matplotlib
import numpy as np
import os
import datetime
#import netCDF4 as nc
import xarray as xr
from scipy import interpolate
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
#import brewer2mpl as cbrew
import scipy.io as sio


#from mml_mapping_fun import mml_map, mml_map_NA, mml_neon_box


from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)

# Avoid having to restart kernel if I modify my mapping scripts.
import imp
#imp.reload(mml_map)
#imp.reload(mml_map_NA)
#imp.reload(mml_neon_box)
import matplotlib.colors as mcolors


# In[]:


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


# In[]:
c = mcolors.ColorConverter().to_rgb
wrbw = make_colormap(
    [c('white'), c('red'), 0.5, c('blue'), c('white')])
N = 1000
array_dg = np.random.uniform(0, 10, size=(N, 2))
colors = np.random.uniform(-2, 2, size=(N,))
plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=wrbw)
plt.colorbar()
plt.show()

  
# In[]:
 


