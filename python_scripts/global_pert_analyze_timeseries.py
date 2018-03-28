#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 10:32:22 2018

@author: mlague
"""

"""
    Load time series and calculate slopes  +  statistics for specific variables
    in both offline and online simulations of global perturbation simulations
    
    (this script is tailored to the time-series version of the output, not the means)
    
    Will return a slope value for each variable at each location, as well as some statistics
    associated with that slope. 
"""
#%% Libraries

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

import matplotlib.colors as mcolors

# MML's functions:
from mml_mapping_fun import mml_map, discrete_cmap
from custom_python_mml_cmap import make_colormap, mml_cmap
from sensitivity_slope_fun import sensitivity_slope
#from sens_slope_fun2 import sensitivity_slope
from load_masks_coords_fun import get_masks, get_coords, get_seasons
from load_global_pert_data import make_variable_arrays, get_online, get_offline
from box_avg_fun import avg_over_box, draw_box



#%% Paths & Experiment lists


"""
    Do the preliminary import of masks, area fns, etc
"""

# area grid, lat/lon/lev:
area_f19, lat, lon, LT, LN, lev = get_coords()

# dictionary of seasons:
seasons = get_seasons()

# masks:
landfrac, landmask, ocnmask, bareground_mask, glc_mask, inv_glc_mask = get_masks()


#-----------------------------
# Define filepaths
#-----------------------------
ext_dir={}
ext_dir['online'] = '/home/disk/eos18/mlague/simple_land/output/global_pert_cheyenne/'
ext_dir['offline'] = '/home/disk/eos18/mlague/simple_land/output/global_pert_offline_MML/'



props = ['alb','rs','hc']

sims = {}
for prop in props:
    sims[prop] = {}
    sims[prop]['online'] = {}
    sims[prop]['offline'] = {}


sims['alb']['online']['a3'] = 'global_a3_cv2_hc0.1_rs100_cheyenne'
sims['alb']['online']['a2'] = 'global_a2_cv2_hc0.1_rs100_cheyenne'
sims['alb']['online']['a1'] = 'global_a1_cv2_hc0.1_rs100_cheyenne' 

sims['rs']['online']['rs30'] =   'global_a2_cv2_hc0.1_rs30_cheyenne'
sims['rs']['online']['rs100'] =  'global_a2_cv2_hc0.1_rs100_cheyenne'
sims['rs']['online']['rs200'] =  'global_a2_cv2_hc0.1_rs200_cheyenne' 

sims['hc']['online']['hc0.1'] =  'global_a2_cv2_hc0.1_rs100_cheyenne'
# the hc0.5 tarball is corrupted! This is bad nes, as it means I can't make any moretime series / have lost all the restart files. 
# On the bright side, I do have a fair number of hte time series on olympus already - just not ALL 
sims['hc']['online']['hc0.5'] =  'global_a2_cv2_hc0.5_rs100_cheyenne'
sims['hc']['online']['hc1.0'] =  'global_a2_cv2_hc1.0_rs100_cheyenne'
sims['hc']['online']['hc2.0'] =  'global_a2_cv2_hc2.0_rs100_cheyenne'
sims['hc']['online']['hc10.0'] =  'global_a2_cv2_hc10.0_rs100_cheyenne'
sims['hc']['online']['hc20.0'] =  'global_a2_cv2_hc20.0_rs100_cheyenne' 
 
sims['alb']['offline']['a3'] = 'global_a3_cv2_hc0.1_rs100_offline_b07'
sims['alb']['offline']['a2'] = 'global_a2_cv2_hc0.1_rs100_offline_b07'
sims['alb']['offline']['a1'] = 'global_a1_cv2_hc0.1_rs100_offline_b07'

sims['rs']['offline']['rs30'] =   'global_a2_cv2_hc0.1_rs30_offline_b07'
sims['rs']['offline']['rs100'] =  'global_a2_cv2_hc0.1_rs100_offline_b07'
sims['rs']['offline']['rs200'] =  'global_a2_cv2_hc0.1_rs200_offline_b07' 

sims['hc']['offline']['hc0.1'] =  ''
sims['hc']['offline']['hc0.5'] =  ''
sims['hc']['offline']['hc1.0'] =  ''
sims['hc']['offline']['hc2.0'] =  ''
sims['hc']['offline']['hc10.0'] =  ''
sims['hc']['offline']['hc20.0'] =  '' 



#print(sims)

run0 = 'global_a2_cv2_hc0.1_rs100_cheyenne'
ds0_cam = xr.open_dataset( ext_dir['online'] + run0 + '/means/' + run0 + '.cam.h0.20-50_year_avg.nc')
ds0_clm = xr.open_dataset(ext_dir['online'] + run0 + '/means/' + run0 + '.clm2.h0.20-50_year_avg.nc')

ds_cam = {}
ds_clm = {}

#%%

# dictionaries where we'll store the arrays
atm_resp = {}
lnd_resp = {}

# Vairables to take slopes of (output vars)
ts_vars = {}
ts_vars['atm'] = ['TREFHT','SHFLX','LHFLX','FSNS','FSNSC','FLNS',
          'FLNSC']
#['TREFHT','SHFLX','LHFLX','FSNS','FSNSC','FLNS',
#          'FLNSC','RELHUM','PRECC','PRECL','PRECSC','PRECSL','FSNT','FSNTC','FLNT','FLNTC',
#          'FLUT','FLDS','SWCF','LWCF','CLDLOW','CLDMED','CLDHGH','U','V','WSUB','Z3']
ts_vars['lnd'] = ['MML_ts']#,'MML_water','MML_shflx','MML_lhflx','MML_fsns','MML_flns']   # note, RH is from atm - if running offline, shouldn't change?


# Variables to take slopes of that need to be derrived
derived_vars = ['PRECIP','MSEPOT','BOWEN','EVAPFRAC','TURBFLX','ALBEDO','ALBEDOC']
#   rain+snow, potentail MSE, bowen ratio, evaporative fraction, turbulent heat flux (sh + lh), albedo, clearsky albedo

#%% Load the time series indo the ds_cam and ds_clm dictionaries

# Do online first (offline runs have different sim names)
onoff = 'online'

for prop in props:
    ds_cam[prop] = {}
    ds_clm[prop] = {}
    
    onoff = 'online'
    for run in list(sims[prop][onoff].keys()):
        ds_cam[prop][run] = {}
        ds_clm[prop][run] = {}
        
        ds_cam[prop][run][onoff] = {}
        ds_clm[prop][run][onoff] = {}
        
        name = sims[prop][onoff][run]
        
        # model vars (load):
        for ts_var in ts_vars['atm']:
            # cam gets online only
            ds_cam[prop][run][onoff][ts_var] = xr.open_dataset(ext_dir[onoff] + name +
                  '/TimeSeries/' + name + '.cam.h0.ts.20-50.' + ts_var + '.nc')
        
        for ts_var in ts_vars['lnd']:
            # clm gets online and offline:
            ds_cam[prop][run][onoff][ts_var] = xr.open_dataset(ext_dir[onoff] + name +
                  '/TimeSeries/' + name + '.clm2.h0.ts.20-50.' + ts_var + '.nc')
 
    onoff = 'offline'
    for run in list(sims[prop][onoff].keys()):
        ds_clm[prop][run][onoff] = {}
        
        name = sims[prop][onoff][run]
        
        for ts_var in ts_vars['lnd']:
            # clm gets online and offline:
            ds_cam[prop][run][onoff][ts_var] = xr.open_dataset(ext_dir[onoff] + name +
                  '/TimeSeries/' + name + '.clm2.h0.ts.20-50.' + ts_var + '.nc')
 
    
    
    
