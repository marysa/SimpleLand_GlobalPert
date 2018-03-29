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


from mml_mapping_fun import mml_map, mml_map_NA, mml_neon_box


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
#c = mcolors.ColorConverter().to_rgb
#rbw = make_colormap(
#    [c('white'), c('red'), 0.5, c('blue'), c('white')])
#N = 1000
#array_dg = np.random.uniform(0, 10, size=(N, 2))
#colors = np.random.uniform(-2, 2, size=(N,))
#plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=wrbw)
#plt.colorbar()
#plt.show()

  
# In[]:
 
def mml_cmap(name):
    """ Quickly return the wrbw colour map for doing dlnd/datm sensitivity
    
    """
    c = mcolors.ColorConverter().to_rgb
    #wrbw = make_colormap([c('white'), c('red'), 0.5, c('blue'), c('white')])
    #mml_dblue = (30./255.,43./255.,156./255.)
    #mml_lblue = (84./255.,137./255.,229./255.)
    #mml_red = (191./255.,50./255.,10./255.)
    #mml_yellow = (251./255.,219./255.,15./255.)
    
    mml_dblue = (5./255.,113./255.,176./255.)
    mml_lblue = (146./255.,197./255.,222./255.)
    mml_red = (202./255.,0./255.,32./255.)
    mml_yellow = (244./255.,165./255.,130./255.)
    mml_grey = (247./255.,247./255.,247./255.)
    
    wrbw = make_colormap([c((1,1,1)), c(mml_dblue), 0.5, c(mml_red), c((1,1,1))])
    wbrw = make_colormap([c((1,1,1)), c(mml_red), 0.5, c(mml_dblue), c((1,1,1))])
    
    rwb = make_colormap([c(mml_red), c(mml_grey), 0.5, c(mml_grey), c(mml_dblue)])
    bwr = make_colormap([c(mml_dblue), c('white'), 0.5, c(mml_grey), c(mml_red)])
    
    rywbb = make_colormap([c(mml_red),c(mml_yellow),0.25,c(mml_yellow),c(mml_grey),
                           0.5,c(mml_grey),c(mml_lblue),0.75,c(mml_lblue),c(mml_dblue)])
    bbwyr = make_colormap([c(mml_dblue),c(mml_lblue),0.25,c(mml_lblue),c(mml_grey),
                           0.5,c(mml_grey),c(mml_yellow),0.75,c(mml_yellow),c(mml_red)])
    
    wyrbbw = make_colormap([c(mml_grey),c(mml_yellow),0.25,c(mml_yellow),c(mml_red),
                           0.5,c(mml_dblue),c(mml_lblue),0.75,c(mml_lblue),c(mml_grey)])
    wbbryw = make_colormap([c(mml_grey),c(mml_lblue),0.25,c(mml_lblue),c(mml_dblue),
                           0.5,c(mml_red),c(mml_yellow),0.75,c(mml_yellow),c(mml_grey)])
    
    
    
    
    if name == 'wrbw': 
        mml_color_map = wrbw
    elif name == 'wbrw':
        mml_color_map = wbrw
    elif name=='bwr':
        mml_color_map = bwr
    elif name=='rwb':
        mml_color_map = rwb
    elif name=='wyrbbw' :
        mml_color_map = wyrbbw
    elif name == 'wbbryw' :
        mml_color_map = wbbryw
    elif name == 'rywbb' :
        mml_color_map = rywbb
    elif name == 'bbwyr' :
        mml_color_map = bbwyr
    else:
        print('error! did not choose a color map!')
        
    return mml_color_map

# In[]:
#    
#temp_cb = mml_cmap('rywbb')
#N=1000
#array_dg = np.random.uniform(0, 10, size=(N, 2))
#colors = np.random.uniform(-2, 2, size=(N,))
#plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=temp_cb)
#cb = plt.colorbar()
#plt.show()
#
#temp_cb = mml_cmap('wbbryw')
#N=1000
#array_dg = np.random.uniform(0, 10, size=(N, 2))
#colors = np.random.uniform(-2, 2, size=(N,))
#plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=temp_cb)
#cb = plt.colorbar()
#plt.show()
#
#temp_cb = mml_cmap('bwr')
#N=1000
#array_dg = np.random.uniform(0, 10, size=(N, 2))
#colors = np.random.uniform(-2, 2, size=(N,))
#plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=temp_cb)
#cb = plt.colorbar()
#plt.show()
# In[]:
#    
#temp_cb = mml_cmap('rywbb')
#
#N = 1000
##fig, axes = plt.subplots(1, 1, figsize=(6,6))
#array_dg = np.random.uniform(0, 10, size=(N, 2))
#colors = np.random.uniform(-2, 2, size=(N,))
#
## define the bins and normalize
#bounds = np.linspace(0,20,21)
#norm = plt.colors.BoundaryNorm(bounds, temp_cb.N)
#
#
#ax=plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=temp_cb,norm=norm)
#cb = plt.colorbar.ColorbarBase(ax,cmap=temp_cb,norm=norm,spacing='proportional',
#                               ticks=bounds,boundaries=bounds,format='%1i')
#plt.show()

