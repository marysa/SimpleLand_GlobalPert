#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:32:28 2017

@author: mlague
"""


# netcdf/numpy/xarray
import numpy as np
#import netCDF4 as nc
import numpy.matlib
import datetime
import xarray as xr
from scipy import interpolate
from numpy import ma
from scipy import stats
import scipy.io as sio
#import cpickle as pickle
import pickle as pickle
from sklearn import linear_model
import numpy.ma as ma

import time

from joblib import Parallel, delayed
import multiprocessing

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import brewer2mpl as cbrew

from matplotlib.ticker import FormatStrFormatter

# OS interaction
import os
import sys

from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)

from mml_mapping_fun import mml_map, discrete_cmap
#import mml_map
from custom_python_mml_cmap import make_colormap, mml_cmap

#from planar import BoundingBox

# Avoid having to restart the kernle if I modify my mapping scripts (or anything else)
import imp
#imp.reload(mml_map)
#imp.reload(mml_map_NA)
#imp.reload(mml_neon_box)
import matplotlib.colors as mcolors

#%%
"""
Initial Setup:
    Load data paths, set figure paths, get some required function calls to 
    custom colour bars out of the way...
"""

# Load my colour bar
cm_dlnd = mml_cmap('wrbw')
cm_datm = mml_cmap('bwr')
#cm_datm = mml_cmap('rwb')   # appears to use it backwards... heh...

# Path to save figures:
figpath = '/home/disk/eos18/mlague/simple_land/scripts/python/analysis/global_pert/figures/'

# Point at the data sets
ext_dir = '/home/disk/eos18/mlague/EAGER/new_PostProc/ctrl_finiterp/means_100/'

# Coupled simulations:
#sims = ['global_a2_cv2_hc1_rs100',
#       'global_a1_cv2_hc1_rs100','global_a3_cv2_hc1_rs100',
#       'global_a2_cv2_hc0.5_rs100','global_a2_cv2_hc2_rs100',
#       'global_a2_cv2_hc1_rs30','global_a2_cv2_hc1_rs200']
sims = ['cesm13_cam5clm45_ctrlfiniterp']

# load the file paths and # Open the coupled data sets in xarray
cam_files = {}
clm_files = {}
ds_cam = {}
ds_clm = {}

for run in sims:
    #print ( ext_dir + run + '/means/' + run + '.cam.h0.05-end_year_avg.nc' )
    cam_files[run] =  ext_dir + run + '.cam.h0.20-99_year_avg.nc'
    clm_files[run] =  ext_dir + run + '.clm2.h0.20-99_year_avg.nc'
    
    ds_cam[run] = xr.open_dataset(cam_files[run])
    ds_clm[run] = xr.open_dataset(clm_files[run])

    
# open a cam area file produced in matlab using an EarthEllipsoid from a cam5 f19 lat/lon data set
area_f19_mat = sio.loadmat('/home/disk/eos18/mlague/simple_land/scripts/python/analysis//f19_area.mat')
area_f19 = area_f19_mat['AreaGrid']


# ### Load some standard variables
# lat, lon, landmask

ds = ds_clm['cesm13_cam5clm45_ctrlfiniterp']
lat = ds['lat'].values
lon = ds['lon'].values
landmask = ds['landmask'].values

LN,LT = np.meshgrid(lon,lat)

# surfdat file (has the glacier map)
surfdat_file = '/home/disk/eos18/mlague/simple_land/PreProcessing/mml_small_clm5_surfdata.nc'



# ### Load some standard variables
# lat, lon, landmask
ds0_cam = ds_cam['cesm13_cam5clm45_ctrlfiniterp']
ds0_clm = ds_clm['cesm13_cam5clm45_ctrlfiniterp']
lat = ds['lat'].values
lon = ds['lon'].values
landmask = ds['landmask'].values
landfrac = ds0_clm['landfrac'].values

bareground_mask = np.where(landmask==1,1,np.nan)



#%%

evap_frac = np.array(ds0_cam['LHFLX'].values[:]) / np.array( ds0_cam['LHFLX'].values[:] + ds0_cam['SHFLX'].values[:] )

fig, axes = plt.subplots(1, 1, figsize=(6,4))

mapdata = np.mean(evap_frac,0)
mapdata = mapdata*bareground_mask
mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

units = 'W/m2'

ttl_main = 'clm4.5 evap frac ctrl'
clim_diff = [0,1]
cmap_diff = plt.cm.viridis          
           
ax = fig.gca()

mapdata = mapdata*landmask

mp, cbar, cs = mml_map(LN,LT,mapdata,ds0_cam,'LHFLX','moll',title=ttl_main,clim=clim_diff,colmap=cmap_diff, cb_ttl='units: '+units )
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(12)

# Annotate with season, variable, date
ax.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop +', '+var,fontsize='10',
         ha = 'left',va = 'center',
         transform = ax.transAxes)

plt.show() 