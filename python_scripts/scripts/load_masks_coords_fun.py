#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:58:35 2017

@author: mlague

    Script with lots of pre-loading stuff for global perturbation simulations

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

def get_masks():
    #-----------------------------
    # Inputs:
    #
    # none; load things like glacier masks, landfrac, etc
    #
    #
    # Returns:
    #
    # landfrac - fraction of each gricell taken up by land
    # landmask - 1 where land, nan where not
    # ocnmask - 1 where ocn, nan where not
    # bareground_mask - 1 where non-glaciated land , nan where not
    # glc_mask - 1 where glc, nan where not
    # inv_glc_mask - 1 where NOT glc, nan where is glc, nan where is ocn
    #
    #-----------------------------
    
    #-----------------------------
    # Define filepaths
    #-----------------------------
    
    # surfdat file (has the glacier map)
    ds_surdat = xr.open_dataset('/home/disk/eos18/mlague/simple_land/PreProcessing/mml_small_clm5_surfdata.nc')
    
    # global_pert cam & clm file
    ds_cam = xr.open_dataset('/home/disk/eos18/mlague/simple_land/output/global_pert_cheyenne/global_a2_cv2_hc0.1_rs100_cheyenne/means/global_a2_cv2_hc0.1_rs100_cheyenne.cam.h0.20-50_year_avg.nc')
    ds_clm = xr.open_dataset('/home/disk/eos18/mlague/simple_land/output/global_pert_cheyenne/global_a2_cv2_hc0.1_rs100_cheyenne/means/global_a2_cv2_hc0.1_rs100_cheyenne.clm2.h0.20-50_year_avg.nc')
    
    
    #-----------------------------
    # Define predefined masks
    #-----------------------------
    landmask = ds_clm['landmask'].values[:]
    landfrac = ds_clm['landfrac'].values[:]
 
    # turn 0 -> nan in landmask
    landmask = np.where(landmask==0,np.nan,landmask)
    
    ocnmask = np.ones(np.shape(landmask))
    ocnmask = np.where(landmask==1,np.nan,ocnmask)
    
    #-----------------------------
    # Glacier masking
    #-----------------------------    
    
    # Get glacier mask
    glc_pct = (ds_surdat.variables['PCT_GLACIER']).values[:]
    # only apply the glacier mask where glc_pct >=50% 
    # (also set to be at latitudes above 60 (59 deg is closest to 60), or no? not for now...)
    glc_mask = np.ones(np.shape(glc_pct))*np.nan
    # initially, choose glc_pct >=50
    glc_mask[ glc_pct > 50 ] = 1
    inv_glc_mask = np.ones(np.shape(glc_pct))*np.nan*landmask
    inv_glc_mask [ glc_pct < 50 ] = 1   # includes ocean, so also masky by landmask
    inv_glc_mask = np.where(ocnmask==1,np.nan,inv_glc_mask)
    
    
    bareground_mask = inv_glc_mask
    
    return landfrac, landmask, ocnmask, bareground_mask, glc_mask, inv_glc_mask
  
#%%
def get_coords():
    #-----------------------------
    # Inputs:
    #
    # none; load things like lat, lon, area fractions, etc
    #
    #
    # Returns:
    #
    # area_f19 - the area grid for a 2.5 degree simulation, used for weighting
    # lat - vector latitudes
    # lon - vector longitudes
    # LT - meshgrid latitudes
    # LN - meshgrid longitudes
    #
    #-----------------------------
    
    #-----------------------------
    # Area grid
    #-----------------------------
    
    # open a cam area file produced in matlab using an EarthEllipsoid from a cam5 f19 lat/lon data set
    area_f19_mat = sio.loadmat('/home/disk/eos18/mlague/simple_land/scripts/python/analysis//f19_area.mat')
    area_f19 = area_f19_mat['AreaGrid']
    
    
    #-----------------------------
    # lats & lons
    #-----------------------------
    
    ds_cam = xr.open_dataset('/home/disk/eos18/mlague/simple_land/output/global_pert_cheyenne/global_a2_cv2_hc0.1_rs100_cheyenne/means/global_a2_cv2_hc0.1_rs100_cheyenne.cam.h0.20-50_year_avg.nc')
    ds_clm = xr.open_dataset('/home/disk/eos18/mlague/simple_land/output/global_pert_cheyenne/global_a2_cv2_hc0.1_rs100_cheyenne/means/global_a2_cv2_hc0.1_rs100_cheyenne.clm2.h0.20-50_year_avg.nc')
    
    lat = ds_cam['lat'].values[:]
    lon = ds_cam['lon'].values[:]
    lev = ds_cam['lev'].values[:]
    
    
    #-----------------------------
    # Meshgrid lats & lons
    #-----------------------------
    
    LN,LT = np.meshgrid(lon,lat)

    return area_f19, lat, lon, LT, LN, lev

#%%
def get_seasons():
    # Inputs:
    # none
    #
    # Outputs: 
    # dictionary with seasons
    
    seasons = {}
    
    seasons['names'] = ['ANN','DJF','MAM','JJA','SON']
    
    seasons['indices'] = {}
    seasons['indices']['ANN'] = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
    seasons['indices']['DJF'] = np.array([11,0,1])
    seasons['indices']['MAM'] = np.array([2,3,4])
    seasons['indices']['JJA'] = np.array([5,6,7])
    seasons['indices']['SON'] = np.array([8,9,10])
 
    
    return seasons

