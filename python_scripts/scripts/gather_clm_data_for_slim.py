#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:23:18 2017

@author: mlague

Gather data from various CLM files to make a .nc file consolidating the data needed to 
make an "as-CLM-like" setup of the simple land model as possible.


"""

#%% dictionaries, or whatever they're called.

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
import brewer2mpl as cbrew
from matplotlib import ticker

from matplotlib.ticker import FormatStrFormatter

# OS interaction
import os
import sys

from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)

# MML's functions:
from mml_mapping_fun import mml_map, discrete_cmap
from custom_python_mml_cmap import make_colormap, mml_cmap
from sensitivity_slope_fun import sensitivity_slope
#from sens_slope_fun2 import sensitivity_slope
from load_masks_coords_fun import get_masks, get_coords, get_seasons
from load_global_pert_data import make_variable_arrays, get_online, get_offline
from box_avg_fun import avg_over_box, draw_box


# Avoid having to restart the kernle if I modify my mapping scripts (or anything else)
import imp
import matplotlib.colors as mcolors


#%% File paths with relevant data

#----------------------------------------
#   Conductance / Resistance
#----------------------------------------
rs_path = "/home/disk/eos18/mlague/simple_land/intermediate_netcdfs/clm_pft_data/conductance/gridded/"
rs_file_sun =  "CLM5_MeanSunlitJJA-DJFMedlynGsValues.nc"
rs_file_sha =  "CLM5_MeanShadedJJA-DJFMedlynGsValues.nc"

# For reference: pft-level gridded data is 
pft_rs_12months = "MedlynGs_GSWP3_GSSUNL_year_avg.nc"

#----------------------------------------
#   albedo & Roughness
#       Albedo: (FSDS - FSNS) / FSDS 
#       Albedo_clear = (FSDSC - FSNSC) / FSDSC
#
#       Roughness: HTOP = canopy height
#
#----------------------------------------
clm45_path = "/home/disk/eos4/mlague/overflow/cam5_output/cam5_ctrl/means/"
clm_45_file = "cesm13_cam5clm45_2000_ctrl.clm2.h0.20-50_year_avg.nc"

#%% Load files as xarray datasets

ds_rs_sun = xr.open_dataset( rs_path + rs_file_sun )
ds_rs_sha = xr.open_dataset( rs_path + rs_file_sha )

ds_clm45 = xr.open_dataset( clm45_path + clm_45_file )

#%% Grab relevant masking data like lat, lon, and glacier masks (grab a whole separate file for the glacier mask bit)

# area grid, lat/lon/lev:
area_f19, lat, lon, LT, LN, lev = get_coords()

# dictionary of seasons:
seasons = get_seasons()

# masks:
landfrac, landmask, ocnmask, bareground_mask, glc_mask, inv_glc_mask = get_masks()

# hemispherical latitude indices (for grabbing summer of each hemisphere)
lat_ind_SH = np.array(range(0,48))
lat_ind_NH = np.array(range(48,96))

#%% Grab arrays of relevant variables. Check size (for seasonal purposes)

# NOTE: This is LEAF LEVEL rs; divide by LAI for a better canopy est... which means
# I need to go dig up LAI; trying to write it out to the same .nc file... its on yellowstone...

rs_sun = {}
rs_sun['JJA'] = np.array(ds_rs_sun['MedlynGsSunJJA'][:])
rs_sun['DJF'] = np.array(ds_rs_sun['MedlynGsSunDJF'][:])

rs_sha = {}
rs_sha['JJA'] = np.array(ds_rs_sha['MedlynGsShaJJA'][:])
rs_sha['DJF'] = np.array(ds_rs_sha['MedlynGsShaDJF'][:])

LAI = {}
LAI['DJF'] = np.array(ds_rs_sun['LaiDJF'][:])
LAI['JJA'] = np.array(ds_rs_sun['LaiJJA'][:])

htop = {}
htop['year']  = np.array(ds_clm45['HTOP'][:])
htop['JJA'] = np.nanmean(htop['year'][[5,6,7],:,:],0)
htop['DJF'] = np.nanmean(htop['year'][[11,0,1],:,:],0)

albedo = {}
albedo['year'] = np.array(ds_clm45['FSR'][:]) / np.array(ds_clm45['FSDS'][:])
albedo['JJA'] = np.nanmean(albedo['year'][[5,6,7],:,:],0)
albedo['DJF'] = np.nanmean(albedo['year'][[11,0,1],:,:],0)
albedo['Aug'] = np.nanmean(albedo['year'][[7],:,:],0)


#%% Splice together NH summer with SH summer, ie take DJF values for SH, JJA values for NH.
# If alebdos are "too bright" in the Arctic, take august value...

summer_props = {}

# note: rs is 1-degree ---> will have to interpolate to bring it up to 2.5

# albedo:
summer_props['albedo'] = np.zeros(np.shape(albedo['JJA']))
summer_props['albedo'][lat_ind_SH,:] = albedo['DJF'][lat_ind_SH,:]
summer_props['albedo'][lat_ind_NH,:] = albedo['JJA'][lat_ind_NH,:]

temp_albedo_check = np.nanmin(albedo['year'],0)
plt.imshow(temp_albedo_check,clim=[0,0.3])
plt.colorbar()
plt.title('min albedo')
plt.show()
plt.close()

# canopy height:
summer_props['htop'] = np.zeros(np.shape(htop['JJA']))
summer_props['htop'][lat_ind_SH,:] = htop['DJF'][lat_ind_SH,:]
summer_props['htop'][lat_ind_NH,:] = htop['JJA'][lat_ind_NH,:]

plt.imshow(summer_props['htop'])
plt.colorbar()
plt.title('htop [m]')
plt.show()
plt.close()

rs_hres_sun = np.zeros(np.shape(rs_sun['JJA']))
eq = 192/2
rs_hres_sun[0:eq,:] = rs_sun['DJF'][0:eq,:]
rs_hres_sun[eq:192,:] = rs_sun['JJA'][eq:192,:]
plt.imshow(rs_hres_sun)
plt.colorbar()
plt.title('rs sun (high res)')
plt.show()
plt.close()

rs_hres_sha = np.zeros(np.shape(rs_sha['JJA']))
eq = 192/2
rs_hres_sha[0:eq,:] = rs_sha['DJF'][0:eq,:]
rs_hres_sha[eq:192,:] = rs_sha['JJA'][eq:192,:]
plt.imshow(rs_hres_sha)
plt.colorbar()
plt.title('rs sha (high res)')
plt.show()
plt.close()

# Bulk rs: divide by lai
rs_hres_sun = np.zeros(np.shape(rs_sun['JJA']))
eq = 192/2
rs_hres_sun[0:eq,:] = rs_sun['DJF'][0:eq,:]
rs_hres_sun[eq:192,:] = rs_sun['JJA'][eq:192,:]
plt.imshow(rs_hres_sun/LAI['JJA'],clim=[0,500])
plt.colorbar()
plt.title('Bulk rs sun (high res)')
plt.show()
plt.close()

rs_hres_sha = np.zeros(np.shape(rs_sha['JJA']))
eq = 192/2
rs_hres_sha[0:eq,:] = rs_sha['DJF'][0:eq,:]
rs_hres_sha[eq:192,:] = rs_sha['JJA'][eq:192,:]
plt.imshow(rs_hres_sha/LAI['JJA'],clim=[0,500])
plt.colorbar()
plt.title('Bulk rs sha (high res)')
plt.show()
plt.close()

rs_hres_sha = np.zeros(np.shape(rs_sha['JJA']))
eq = 192/2
rs_hres_sha[0:eq,:] = rs_sha['DJF'][0:eq,:]
rs_hres_sha[eq:192,:] = rs_sha['JJA'][eq:192,:]
plt.imshow(LAI['JJA'])
plt.colorbar()
plt.title('Bulk rs sha (high res)')
plt.show()
plt.close()

#%%




