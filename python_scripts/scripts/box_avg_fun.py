#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 08:15:52 2017

@author: mlague

Average (e.g. some flux) over a box defined by lat_bounds, lon_bounds

(purpose: look at changes in surface energy budget between experiments. This is
just for one experiment)

"""

# In[]:

# netcdf/numpy/xarray
import numpy as np
import netCDF4 as nc
import numpy.matlib
import datetime
import xarray as xr
from scipy import interpolate
from numpy import ma
from scipy import stats
import scipy.io as sio
import pickle as pickle
from sklearn import linear_model
import numpy.ma as ma

import time

from copy import copy 

from joblib import Parallel, delayed
import multiprocessing

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
#import brewer2mpl as cbrew
from matplotlib import ticker

from matplotlib.ticker import FormatStrFormatter

# OS interaction
import os
import sys

from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)

from mml_mapping_fun import mml_map, discrete_cmap
from custom_python_mml_cmap import make_colormap, mml_cmap


# Avoid having to restart the kernle if I modify my mapping scripts (or anything else)
import imp
import matplotlib.colors as mcolors



#%%
def avg_over_box(dat_to_avg=None, area_grid=None, lat=None, lon=None, lat_bounds=None, lon_bounds=None):
    #----------------------------------
    # Inputs:
    #
    # dat_to_avg = data to average. Don't feed in places with land/ocn boundary, I'm not dealing with that here.
    # area_grid = f19 areas for weighting
    # lat = vector of lats - to be used looking up lat_bounds
    # lon = vector of lons - to be used looking up lon_bounds
    # lat_bounds = latitude boundaries of box to average over  (-90 to 90)
    # lon _bounds = longitude boundaries of box to average over (0 to 360, Europe -> Asia -> Pacific -> Americas -> Atlantic)
    #
    # Outputs:
    #
    # box_avg = average of dat_to_avg averaged (area weighted) over the box lat_bounds, lon_bounds
    #----------------------------------
    
    #lat_bounds = [-4,40]
    #lon_bounds = [10,35]
    
    lat_bot = np.min(lat_bounds)
    lat_top = np.max(lat_bounds)
    lon_left = np.min(lon_bounds)
    lon_right = np.max(lon_bounds)
    
    ind_bot = np.argmin(lat<=lat_bot)
    ind_top = np.argmax(lat>=lat_top)
    ind_left = np.argmin(lon<=lon_left)
    ind_right = np.argmax(lon>=lon_right)
    
#    print(ind_left)
#    print(ind_right)
#    print(ind_bot)
#    print(ind_top)
    
#    print(lat[ind_bot])
#    print(lat[ind_top])
#    print(lon[ind_left])
#    print(lon[ind_right])
    
    area_box = area_grid[ind_bot:ind_top,ind_left:ind_right]
    data_box = dat_to_avg[ind_bot:ind_top,ind_left:ind_right]

    data_box_total = np.nansum(np.nansum(data_box*area_box,1),0)
#    print(data_box_total)
    box_avg = data_box_total / (np.nansum(np.nansum(area_box,1),0))
#    print(box_avg)
    
    return box_avg
#%%

def draw_box(mp_ax=None,lat_bounds=None,lon_bounds=None,lat=None,lon=None,line_col=None,label=None):
    
    lat_bot = np.min(lat_bounds)
    lat_top = np.max(lat_bounds)
    lon_left = np.min(lon_bounds)
    lon_right = np.max(lon_bounds)
    
    ind_bot = np.argmin(lat<=lat_bot)
    ind_top = np.argmax(lat>=lat_top)
    ind_left = np.argmin(lon<=lon_left)
    ind_right = np.argmax(lon>=lon_right)
    
    x1,y1 = mp_ax(lon[ind_left],lat[ind_bot])
    x2,y2 = mp_ax(lon[ind_right],lat[ind_bot])
    x3,y3 = mp_ax(lon[ind_right],lat[ind_top])
    x4,y4 = mp_ax(lon[ind_left],lat[ind_top])
    poly = plt.Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor=None,edgecolor=line_col,linewidth=3,fill=False,label=label)
    box_handle = plt.gca().add_patch(poly)
    
    return mp_ax, box_handle