#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:11:51 2017

@author: mlague

New sensitivities plot script, also for deltas 
Has MSE, Bowen ratio, etc in this load script part

"""



#%%

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
#import cpickle as pickle
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
ext_dir = '/home/disk/eos18/mlague/simple_land/output/global_pert_cheyenne/'

# Coupled simulations:
#sims = ['global_a2_cv2_hc1_rs100',
#       'global_a1_cv2_hc1_rs100','global_a3_cv2_hc1_rs100',
#       'global_a2_cv2_hc0.5_rs100','global_a2_cv2_hc2_rs100',
#       'global_a2_cv2_hc1_rs30','global_a2_cv2_hc1_rs200']
sims = ['global_a2_cv2_hc0.1_rs100_cheyenne',
       'global_a1_cv2_hc0.1_rs100_cheyenne','global_a3_cv2_hc0.1_rs100_cheyenne',
       'global_a2_cv2_hc0.01_rs100_cheyenne','global_a2_cv2_hc0.05_rs100_cheyenne',
       'global_a2_cv2_hc0.5_rs100_cheyenne',
       'global_a2_cv2_hc1.0_rs100_cheyenne','global_a2_cv2_hc2.0_rs100_cheyenne',
       'global_a2_cv2_hc0.1_rs30_cheyenne','global_a2_cv2_hc0.1_rs200_cheyenne']


# load the file paths and # Open the coupled data sets in xarray
cam_files = {}
clm_files = {}
ds_cam = {}
ds_clm = {}

for run in sims:
    #print ( ext_dir + run + '/means/' + run + '.cam.h0.05-end_year_avg.nc' )
    cam_files[run] = ext_dir + run + '/means/' + run + '.cam.h0.20-50_year_avg.nc'
    clm_files[run] = ext_dir + run + '/means/' + run + '.clm2.h0.20-50_year_avg.nc'
    
    ds_cam[run] = xr.open_dataset(cam_files[run])
    ds_clm[run] = xr.open_dataset(clm_files[run])

# open a cam area file produced in matlab using an EarthEllipsoid from a cam5 f19 lat/lon data set
area_f19_mat = sio.loadmat('/home/disk/eos18/mlague/simple_land/scripts/python/analysis//f19_area.mat')
area_f19 = area_f19_mat['AreaGrid']


# ### Load some standard variables
# lat, lon, landmask

ds = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
lat = ds['lat'].values
lon = ds['lon'].values
landmask = ds['landmask'].values

LN,LT = np.meshgrid(lon,lat)

# surfdat file (has the glacier map)
surfdat_file = '/home/disk/eos18/mlague/simple_land/PreProcessing/mml_small_clm5_surfdata.nc'



# ### Load some standard variables
# lat, lon, landmask
ds0_clm = xr.open_dataset(ext_dir+'global_a2_cv2_hc0.1_rs100_cheyenne/means/'+'global_a2_cv2_hc0.1_rs100_cheyenne.clm2.h0.20-50_annual_avg.nc')
ds0_cam = xr.open_dataset(ext_dir+'global_a2_cv2_hc0.1_rs100_cheyenne/means/'+'global_a2_cv2_hc0.1_rs100_cheyenne.cam.h0.20-50_annual_avg.nc')
ds_glc = xr.open_dataset(surfdat_file)

ds = ds0_clm
lat = ds['lat'].values
lon = ds['lon'].values
landmask = ds['landmask'].values
landfrac = ds0_clm['landfrac'].values



# Get glacier mask
glc_pct = (ds_glc.variables['PCT_GLACIER']).values[:]
# only apply the glacier mask where glc_pct >=50% 
# (also set to be at latitudes above 60 (59 deg is closest to 60), or no? not for now...)
glc_mask = np.zeros(np.shape(glc_pct))
# initially, choose glc_pct >=50
glc_mask[ glc_pct > 50 ] = 1
inv_glc_mask = np.zeros(np.shape(glc_pct))
inv_glc_mask [ glc_pct < 50 ] = 1   # includes ocean, so also masky by landmask



#%%
#
## Define data sets and subtitles
ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
#
## perturbatinos
#ds1 = ds_cam['global_a2_cv2_hc1_rs100']
#ds2 = ds_cam['global_a2_cv2_hc2_rs100']
#
## land files
#dsl0 = ds_clm['global_a2_cv2_hc0.5_rs100']
#dsl1 = ds_clm['global_a2_cv2_hc1_rs100']
#dsl2 = ds_clm['global_a2_cv2_hc2_rs100']


#%%  Get masks:

ds = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
lat = ds['lat'].values
lon = ds['lon'].values
landmask = ds['landmask'].values
landmask_nan = np.where(landmask==1,landmask,np.nan)
ocnmask_nan = landmask_nan.copy
ocnmask_nan = np.where(landmask==1,np.nan,1)

LN,LT = np.meshgrid(lon,lat)

# Get a glacier mask, too!
temp = np.mean(ds.mean('time')['MML_cv'].values[:],0)
print(np.shape(temp))
#plt.imshow(temp)
#plt.colorbar()
temp2 = temp.copy()
is_glc = np.where(temp2>1950000,np.nan,1)*landmask_nan
no_glc = np.where(temp2>1950000,1,np.nan)


ocn_glc_mask = ocnmask_nan.copy()
ocn_glc_mask = np.where(np.isnan(is_glc)==False,1,ocn_glc_mask)

bareground_mask = ocn_glc_mask.copy()
bareground_mask = np.where(np.isnan(ocn_glc_mask),1,np.nan)

######


#plt.imshow(bareground_mask)
#plt.colorbar()


#%% 

#%% 

#%% Example line-plot of sensitivity (single point... do later, from actual slope analysis I do on full map below)

do_sens_plots = 0
do_line_plots = 1

do_slope_analysis = 1

do_line_plots_fixed_axis = 1
do_line_plots_fixed_range = 1


#%%


#------------------------------------
# setup some empty dictionaries:
atm_var = {}
prop = {}
units = {}
slope = {}
pert = {}
# annual mean:
atm_resp = {}
# seasonal:
atm_resp_ann = {}
atm_resp_djf = {}
atm_resp_mam = {}
atm_resp_jja = {}
atm_resp_son = {}
# data sets
# atm
ds_low = {}
ds_med = {}
ds_high = {}
# lnd
dsl_low = {}
dsl_med = {}
dsl_high = {}

# for extra roughness sims
ds_low1 = {}
ds_low2 = {}
ds_med1 = {}
ds_med2 = {}
ds_high1 = {}
ds_high2 = {}

dsl_low1 = {}
dsl_low2 = {}
dsl_med1 = {}
dsl_med2 = {}
dsl_high1 = {}
dsl_high2 = {}
#------------------------------------
# fill in data sets

# albedo:
ds_low['alb'] = ds_cam['global_a1_cv2_hc0.1_rs100_cheyenne']
ds_med['alb'] = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
ds_high['alb'] = ds_cam['global_a3_cv2_hc0.1_rs100_cheyenne']

dsl_low['alb'] = ds_clm['global_a1_cv2_hc0.1_rs100_cheyenne']
dsl_med['alb'] = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
dsl_high['alb'] = ds_clm['global_a3_cv2_hc0.1_rs100_cheyenne']

## roughness:
#ds_low['hc'] = ds_cam['global_a2_cv2_hc0.5_rs100']
#ds_med['hc'] = ds_cam['global_a2_cv2_hc1_rs100']
#ds_high['hc'] = ds_cam['global_a2_cv2_hc2_rs100']
#
#dsl_low['hc'] = ds_clm['global_a2_cv2_hc0.5_rs100']
#dsl_med['hc'] = ds_clm['global_a2_cv2_hc1_rs100']
#dsl_high['hc'] = ds_clm['global_a2_cv2_hc2_rs100']
#
#
## log rel'n roughness:
#ds_low['log_hc'] = ds_cam['global_a2_cv2_hc0.5_rs100']
#ds_med['log_hc'] = ds_cam['global_a2_cv2_hc1_rs100']
#ds_high['log_hc'] = ds_cam['global_a2_cv2_hc2_rs100']
#
#dsl_low['log_hc'] = ds_clm['global_a2_cv2_hc0.5_rs100']
#dsl_med['log_hc'] = ds_clm['global_a2_cv2_hc1_rs100']
#dsl_high['log_hc'] = ds_clm['global_a2_cv2_hc2_rs100']

# roughness:
ds_low1['hc'] = ds_cam['global_a2_cv2_hc0.01_rs100_cheyenne']
ds_low2['hc'] = ds_cam['global_a2_cv2_hc0.05_rs100_cheyenne']
ds_med1['hc'] = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
ds_med2['hc'] = ds_cam['global_a2_cv2_hc0.5_rs100_cheyenne']
ds_high1['hc'] = ds_cam['global_a2_cv2_hc1.0_rs100_cheyenne']
ds_high2['hc'] = ds_cam['global_a2_cv2_hc2.0_rs100_cheyenne']

dsl_low1['hc'] = ds_clm['global_a2_cv2_hc0.01_rs100_cheyenne']
dsl_low2['hc'] = ds_clm['global_a2_cv2_hc0.05_rs100_cheyenne']
dsl_med1['hc'] = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
dsl_med2['hc'] = ds_clm['global_a2_cv2_hc0.5_rs100_cheyenne']
dsl_high1['hc'] = ds_clm['global_a2_cv2_hc1.0_rs100_cheyenne']
dsl_high2['hc'] = ds_clm['global_a2_cv2_hc2.0_rs100_cheyenne']


# log rel'n roughness:
ds_low1['log_hc'] = ds_cam['global_a2_cv2_hc0.01_rs100_cheyenne']
ds_low2['log_hc'] = ds_cam['global_a2_cv2_hc0.05_rs100_cheyenne']
ds_med1['log_hc'] = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
ds_med2['log_hc'] = ds_cam['global_a2_cv2_hc0.5_rs100_cheyenne']
ds_high1['log_hc'] = ds_cam['global_a2_cv2_hc1.0_rs100_cheyenne']
ds_high2['log_hc'] = ds_cam['global_a2_cv2_hc2.0_rs100_cheyenne']

dsl_low1['log_hc'] = ds_clm['global_a2_cv2_hc0.01_rs100_cheyenne']
dsl_low2['log_hc'] = ds_clm['global_a2_cv2_hc0.05_rs100_cheyenne']
dsl_med1['log_hc'] = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
dsl_med2['log_hc'] = ds_clm['global_a2_cv2_hc0.5_rs100_cheyenne']
dsl_high1['log_hc'] = ds_clm['global_a2_cv2_hc1.0_rs100_cheyenne']
dsl_high2['log_hc'] = ds_clm['global_a2_cv2_hc2.0_rs100_cheyenne']

# evaporative resistance:
ds_low['rs'] = ds_cam['global_a2_cv2_hc0.1_rs30_cheyenne']
ds_med['rs'] = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
ds_high['rs'] = ds_cam['global_a2_cv2_hc0.1_rs200_cheyenne']

dsl_low['rs'] = ds_clm['global_a2_cv2_hc0.1_rs30_cheyenne']
dsl_med['rs'] = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
dsl_high['rs'] = ds_clm['global_a2_cv2_hc0.1_rs200_cheyenne']


# atmospheric variable to evaluate:
#atm_var= 'TREFHT'

   
#sfc_props = ['alb','rs','hc','log_hc']
sfc_props= ['alb','rs']
#sfc_prop_ranges = np.array([ [0.1, 0.2, 0.3],
#                  [0.5, 1., 2.],
#                  [30., 100., 200.],
#                  [np.log(0.5), np.log(1.), np.log(2.)]])
sfc_prop_ranges = np.array([ [0.1, 0.2, 0.3,np.nan,np.nan],
                            [30., 100., 200.,np.nan,np.nan],
                            [0.01,0.05,0.1,0.5, 1., 2.],
                            [np.log(0.01),np.log(0.05),np.log(0.1),np.log(0.5), np.log(1.), np.log(2.)]])
print(np.shape(sfc_prop_ranges))

print(sfc_prop_ranges)

seasons = ['ANN','DJF','MAM','JJA','SON']

slope_vars = ['TREFHT','SHFLX','LHFLX','FSNT','FSNTC','FLNT','FLNTC','FSNS','FSNSC','FLNS','FLNSC','PRECC','PRECL','PRECSC','PRECSL','CLDLOW','CLDMED','CLDHGH']
    
ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

delta = {}
glbl_avg = {}

for atm_var in slope_vars:

    units[atm_var] = ds0[atm_var].units
    
    atm_resp[atm_var] = {}
    delta[atm_var] = {}
    glbl_avg[atm_var] = {}
    
    slope[atm_var] = {}
    
    print(atm_var)
    
    for sea in seasons: 
        atm_resp[atm_var][sea] = {}
        delta[atm_var][sea] = {}
        glbl_avg[atm_var][sea] = {}
        slope[atm_var][sea] = {}
        

    i = 0
    for prop in sfc_props:
        pert[prop] = sfc_prop_ranges[i]
        
        if np.isnan(pert[prop][3]):
            print(prop)
            ds1 = ds_low[prop]
            ds2 = ds_med[prop]
            ds3 = ds_high[prop]
            
            # annual mean response
            atm_resp_ann[prop] = np.array([np.array(ds1.mean('time')[atm_var].values[:,:]),
                np.array(ds2.mean('time')[atm_var].values[:,:]),
                np.array(ds3.mean('time')[atm_var].values[:,:])])
        
            # seasonal responses:
            # (first, make 12 month response, then average over djf, jja, etc)
            #print(np.shape(ds1[atm_var].values))
            resp_mths = np.array([np.array(ds1[atm_var].values[:,:,:]),
                    np.array(ds2[atm_var].values[:,:,:]),
                    np.array(ds3[atm_var].values[:,:,:])])
        
        else:
            print(prop)
            ds1 = ds_low1[prop] #0.01
            ds2 = ds_low2[prop]  #0.05
            ds3 = ds_med1[prop]  #0.1
            ds4 = ds_med2[prop]    #0.5
            ds5 = ds_high1[prop]    #1
            ds6 = ds_high2[prop]    #2
            
            # annual mean response
            atm_resp_ann[prop] = np.array([np.array(ds1.mean('time')[atm_var].values[:,:]),
                np.array(ds2.mean('time')[atm_var].values[:,:]),
                np.array(ds3.mean('time')[atm_var].values[:,:]),
                np.array(ds4.mean('time')[atm_var].values[:,:]),
                np.array(ds5.mean('time')[atm_var].values[:,:]),
                np.array(ds6.mean('time')[atm_var].values[:,:]),
                ])
        
            # seasonal responses:
            # (first, make 12 month response, then average over djf, jja, etc)
            #print(np.shape(ds1[atm_var].values))
            resp_mths = np.array([np.array(ds1[atm_var].values[:,:,:]),
                    np.array(ds2[atm_var].values[:,:,:]),
                    np.array(ds3[atm_var].values[:,:,:]),
                    np.array(ds4[atm_var].values[:,:,:]),
                    np.array(ds5[atm_var].values[:,:,:]),
                    np.array(ds6[atm_var].values[:,:,:]),
                    ])
        
        #print(np.shape(resp_mths))
        #print(type(resp_mths))
        #print(resp_mths[:,[11,0,1]])
        atm_resp_djf[prop] = np.mean(resp_mths[:,[11,0,1],:,:],1).squeeze()
        atm_resp_mam[prop] = np.mean(resp_mths[:,[2,3,4],:,:],1).squeeze()
        atm_resp_jja[prop] = np.mean(resp_mths[:,[5,6,7],:,:],1).squeeze()
        atm_resp_son[prop] = np.mean(resp_mths[:,[8,9,10],:,:],1).squeeze()
        
        atm_resp[atm_var]['ANN'][prop] = atm_resp_ann[prop]
        atm_resp[atm_var]['DJF'][prop] = atm_resp_djf[prop]
        atm_resp[atm_var]['MAM'][prop] = atm_resp_mam[prop]
        atm_resp[atm_var]['JJA'][prop] = atm_resp_jja[prop]
        atm_resp[atm_var]['SON'][prop] = atm_resp_son[prop]
        
        print('making delta')
        for sea in seasons:
            if np.size(np.shape(ds1[atm_var][:])) == 3 :
                
                delta[atm_var][sea][prop] = atm_resp[atm_var][sea][prop][-1,:,:] - atm_resp[atm_var][sea][prop][0,:,:]
                
                glbl_avg[atm_var][sea][prop] = np.sum(np.sum(atm_resp[atm_var][sea][prop]*area_f19,2),1)/(np.sum(np.sum(area_f19,1),0))
                
            elif np.size(np.shape(ds1[atm_var][:])) == 4 :
                delta[atm_var][sea][prop] = atm_resp[atm_var][sea][prop][-1,:,:,:] - atm_resp[atm_var][sea][prop][0,:,:,:]
        
        
        #print(np.shape(atm_resp_djf[prop]))
        i=i+1

#%%
        """
            DERIVED VARIABLES
        """
# Make precip & Column MSE a thing:

atm_resp['PRECIP'] = {}
atm_resp['MSE'] = {}  
atm_resp['MSEC'] = {}  
atm_resp['EVAPFRAC'] = {}  
atm_resp['BOWEN'] = {}  

delta['PRECIP'] = {}
delta['MSE'] = {}  
delta['MSEC'] = {}  
delta['EVAPFRAC'] = {}  
delta['BOWEN'] = {}  
        
for sea in seasons: 
    atm_resp['PRECIP'][sea] = {}
    atm_resp['MSE'][sea] = {}
    atm_resp['MSEC'][sea] = {}
    atm_resp['EVAPFRAC'][sea] = {}
    atm_resp['BOWEN'][sea] = {}
    
    delta['PRECIP'][sea] = {}
    delta['MSE'][sea] = {}
    delta['MSEC'][sea] = {}
    delta['EVAPFRAC'][sea] = {}
    delta['BOWEN'][sea] = {}

    i = 0
    for prop in sfc_props:
        
        atm_resp['PRECIP'][sea][prop] = np.array( atm_resp['PRECC'][sea][prop]  + 
                                                 atm_resp['PRECL'][sea][prop]  +
                                                 atm_resp['PRECSC'][sea][prop] + 
                                                 atm_resp['PRECSL'][sea][prop] )
        delta['PRECIP'][sea][prop] = atm_resp['PRECIP'][sea][prop][2,:,:] - atm_resp['PRECIP'][sea][prop][0,:,:]
        
        # MSE full model                                           
        atm_resp['MSE'][sea][prop] = np.array(
                                        (atm_resp['FSNT'][sea][prop] - atm_resp['FLNT'][sea][prop]) - 
                                        ( (atm_resp['FLNS'][sea][prop] )
                                        + atm_resp['SHFLX'][sea][prop] + atm_resp['LHFLX'][sea][prop]) )
        delta['MSE'][sea][prop] = atm_resp['MSE'][sea][prop][2,:,:] - atm_resp['MSE'][sea][prop][0,:,:]
        
        # MSE clearsky (no clouds)                                           
        atm_resp['MSEC'][sea][prop] = np.array(
                                        (atm_resp['FSNTC'][sea][prop] - atm_resp['FLNTC'][sea][prop]) - 
                                        ( (atm_resp['FLNSC'][sea][prop] )
                                        + atm_resp['SHFLX'][sea][prop] + atm_resp['LHFLX'][sea][prop]) )
        delta['MSEC'][sea][prop] = atm_resp['MSEC'][sea][prop][2,:,:] - atm_resp['MSEC'][sea][prop][0,:,:]
        
        
        # EVAPORATIVE FRACTION
        atm_resp['EVAPFRAC'][sea][prop] = np.array(
                                        (atm_resp['LHFLX'][sea][prop]) / 
                                         (atm_resp['SHFLX'][sea][prop] + atm_resp['LHFLX'][sea][prop]))
        delta['EVAPFRAC'][sea][prop] = atm_resp['EVAPFRAC'][sea][prop][2,:,:] - atm_resp['EVAPFRAC'][sea][prop][0,:,:]
        
        # Bowen Ratio
        atm_resp['BOWEN'][sea][prop] = np.array(
                                        (atm_resp['SHFLX'][sea][prop]) / 
                                         atm_resp['LHFLX'][sea][prop]   )
        delta['BOWEN'][sea][prop] = atm_resp['BOWEN'][sea][prop][2,:,:] - atm_resp['BOWEN'][sea][prop][0,:,:]
        
        
        
        
        
        i=i+1
        

#%% Slope calculation!

#mean_plot_vars = ['TREFHT','SHFLX','LHFLX','FSNS','FSNSC','CLDLOW','CLDMED','CLDHGH','MSE','BOWEN','EVAPFRAC']

do_slope_analysis==1

if do_slope_analysis==1:
    # sklearn linear_model , pearson r
    
    # Test more than just TREFHT ... try SHFLX, LHFLX for example, too
    
    slope_vars = ['TREFHT','SHFLX','LHFLX','FSNT','FSNTC','FLNT','FLNTC','FSNS','FSNSC','FLNS','FLNSC','PRECIP','MSE','EVAPFRAC']
    
    # Annual and seasonal... ought to be able to nest, non? 
    slope = {}
    inv_slope = {}
    intercept = {}
    r_value = {}
    p_value = {}
    std_err = {}
    
    std_dev = {}
    
    sensitivity = {}
        
    # Linear regression
    
    for var in slope_vars:
    
        # Annual and seasonal... ought to be able to nest, non? 
        slope[var] = {}
        inv_slope[var] = {}
        intercept[var] = {}
        r_value[var] = {}
        p_value[var] = {}
        std_err[var] = {}
        
        std_dev[var] = {}
        
        sensitivity[var] = {}
        
        seasons = ['ANN','DJF','MAM','JJA','SON']
        
        for sea in seasons:
            slope[var][sea] = {}
            inv_slope[var][sea] = {}
            intercept[var][sea] = {}
            r_value[var][sea] = {}
            p_value[var][sea] = {}
            std_err[var][sea] = {}
            sensitivity[var][sea] = {}
            std_dev[var][sea] = {}
        
        ''' I should write these loops in parallel. It doesn't look too hard, if I had 
            tried to do that from the get-go. As its coded now, I think I'd have to move 
            everything into definition functions, and pass in sea and prop... which 
            might not be too hard, or I can just let the script run overnight... :|
        
        '''
        for prop in sfc_props:
            
            for sea in seasons:
            
                #-----------------------------
                #  Do the regression
                #-----------------------------
                
                # Get the perturbation values, make an np.array (default is list)
                xvals = np.array(pert[prop])
                k = np.size(xvals)  
                print(k)
                if np.isnan(xvals[4]):  # they were forced to all be size 6 b/c of roughness. If those >3 are nan, set k to 3.
                    k = 3
                    xvals = xvals[0:3]
                    
                print(xvals)
                    
                print(k)
                print(np.max(xvals))
                
                
                # grab atmospheric response data for current property, make an np.array
                raw_data = np.array(atm_resp[var][sea][prop])
                print(np.shape(raw_data))
                # flatten response data into a single long vector (Thanks to Andre for showing me how to do this whole section)
                raw_data_v = raw_data.reshape(k, -1)
                #print(np.shape(raw_data_v))
                #print(np.shape(xvals[:,None]))
                
                
                # create an "empty" model
                model = linear_model.LinearRegression()
                
                # Fit the model to tmp_data
                model.fit(xvals[:, None], raw_data_v)
                
                #  grab the linear fit vector
                slope_vector = model.coef_
                intercept_vector = model.intercept_
                
                # put back into lat/lon (hard coded to be 2.5 degrees right now...)
                slope[var][sea][prop] = slope_vector.reshape(96,144)
                intercept[var][sea][prop] = intercept_vector.reshape(96,144)
                
                #-----------------------------
                #   Calculate the r^2 value
                #-----------------------------
            
                # grab the linear fit using the slope and intercept, so we can calculate the correlation coefficient
                fit_data_v = np.transpose(slope_vector*xvals)
                #print(np.shape(fit_data_v))
                #print(np.shape(raw_data_v))
             
                # Calculate r value
                #r_v, p_v = stats.pearsonr(raw_data_v, fit_data_v)
                #get_ipython().magic('%timeit')
                #r_v = np.corrcoef(x=raw_data_v,y=fit_data_v,rowvar=0)
                
                # Going to do this by hand until I figure out how to do it on a matrix...
                #print(np.size(raw_data_v,1))
                x_bar = np.mean(xvals)
                std_x = stats.tstd(xvals)
                #print(x_bar)
                #print(std_x)
                
                #print((np.shape(raw_data_v[1,:])))
                #print(np.shape(raw_data_v))
                #print(np.shape(fit_data_v))
                r_v = np.zeros(np.shape(raw_data_v[0,:]))
                p_v = np.zeros(np.shape(raw_data_v[0,:]))
                #print(np.shape(r_v))
                
                for j in range(np.size(raw_data_v,1)):
                   
                    # compare to using the pearson-r function:
                    r, p = stats.pearsonr(raw_data_v[:,j],fit_data_v[:,j])
                    r_v[j] = r
                    p_v[j] = p
            
              
                #print(np.shape(r_v.reshape(96,144)))
                
                r_value[var][sea][prop] = r_v.reshape(96,144)
                p_value[var][sea][prop] = p_v.reshape(96,144)
        
                
                
                # Do standard deviation analysis
                
                sigma = np.std(raw_data_v,axis=0)
                
                std_dev[var][sea][prop] = sigma.reshape(96,144)
                
                
                # actually calculate inverse slope
                # create an "empty" model
                model = linear_model.LinearRegression()
                
                # Fit the model to tmp_data
                model.fit(raw_data_v,xvals[:, None])
                
                #  grab the linear fit vector
                slope_vector = model.coef_
                intercept_vector = model.intercept_
                
                # put back into lat/lon (hard coded to be 2.5 degrees right now...)
                inv_slope[var][sea][prop] = slope_vector.reshape(96,144)
                
                del raw_data, raw_data_v
            
        
#%%

"""
    Selected mean-state plots
"""

mean_plot_vars = ['TREFHT','SHFLX','LHFLX','FSNS','FSNSC','CLDLOW','CLDMED','CLDHGH','MSE','BOWEN','EVAPFRAC']
mean_plot_vars = ['TREFHT','SHFLX','LHFLX','FSNT','FSNTC','FLNT','FLNTC','FSNS','FSNSC','FLNS','FLNSC','PRECIP','MSE','EVAPFRAC']

for var in mean_plot_vars:
    
    sea = 'ANN'
    
    mapdata = atm_resp[var][sea]['alb'][1,:,:]
    
    mapdata = mapdata*bareground_mask
    mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
    
    fig, axes = plt.subplots(1, 1, figsize=(6,4))
            
    ax = fig.gca()
    
    units = '?'
    
    ttl_main = 'background ' + var
    clim_abs = [np.min(mapdata),np.max(mapdata)]
    cmap_abs = plt.cm.viridis
    
    if (var == 'CLDLOW') or (var == 'CLDMED') or (var == 'CLDHGH'):
                abs_max = np.max([np.abs(np.min(mapdata)),np.abs(np.max(mapdata))])
                clim_abs = [0,0.7]
                #cmap_abs = plt.cm.RdBu
    elif (var == 'SHFLX'):
        abs_max = np.max([np.abs(np.min(mapdata)),np.abs(np.max(mapdata))])
        clim_abs = [-5,60]
        cmap_abs = plt.cm.viridis
    elif (var == 'LHFLX'):
        abs_max = np.max([np.abs(np.min(mapdata)),np.abs(np.max(mapdata))])
        clim_abs = [-5,60]
        cmap_abs = plt.cm.viridis
    elif (var == 'TREFHT'):  
                abs_max = np.max([np.abs(np.min(mapdata)),np.abs(np.max(mapdata))])
                clim_abs = [260,300]
                #cmap_abs = plt.cm.RdBu_r
    elif (var == 'FSNS') or (var == 'FSNSC'):
                abs_max = np.max([np.abs(np.min(mapdata)),np.abs(np.max(mapdata))])
                clim_abs = [0,240]
                #cmap_abs = plt.cm.RdBu_r
    elif (var == 'BOWEN'):
                clim_abs = [0,10]
    elif (var == 'EVAPFRAC'):
                clim_abs = [0,0.8]
    elif (var == 'MSE'):
                clim_abs = [-200,-60]
    
    
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,var,'moll',title=ttl_main,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units )
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)

    # Annotate with season, variable, date
    ax.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop +', '+var,fontsize='10',
             ha = 'left',va = 'center',
             transform = ax.transAxes)
    
    plt.show() 
    
    filename = var +'_'+ sea + '_mean'
    fig_name = figpath+'/sensitivity/'+filename+'.png'
    fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
    
    plt.close()

 #%%

"""
    Selected delta max-min plots
"""

delta_plot_vars = ['TREFHT','SHFLX','LHFLX','FSNS','MSE','PRECIP','BOWEN','EVAPFRAC']

"""
        Plot some deltas
"""

#delta_plot_vars = ['U10','TREFHT','SHFLX','LHFLX','FSNS','FSNSC','CLDLOW','CLDMED','CLDHGH','MSE','BOWEN','EVAPFRAC']
delta_plot_vars = ['TREFHT','SHFLX','LHFLX','FSNS','FSNSC','CLDLOW','CLDMED','CLDHGH','MSE','BOWEN','EVAPFRAC']



for var in delta_plot_vars:

    units = '?'#ds0_cam[var].units
    
    for prop in sfc_props:
        
        for sea in ['ANN']: #seas:
            
            mapdata = delta[var][sea][prop]
            
            mapdata = mapdata*bareground_mask
            mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
            
            ttl_main = '$\Delta$ '+ var + ', ' + prop + ', ' + sea
            filename = 'delta_'+ var + '_' + prop + '_' + sea
            
            cmap_abs = plt.cm.RdBu_r
            cmap_diff = plt.cm.RdBu_r
            
            clim_abs = [np.min(mapdata),np.max(mapdata)]
            
            if prop == 'hc':
                abs_max = np.max([np.abs(np.min(mapdata)),np.abs(np.max(mapdata))])
                clim_abs = [-abs_max, abs_max]
                cmap_abs = plt.cm.RdBu_r
            
            if (var == 'CLDLOW') or (var == 'CLDMED') or (var == 'CLDHGH') or (var == 'LHFLX'):
                abs_max = np.max([np.abs(np.min(mapdata)),np.abs(np.max(mapdata))])
                clim_abs = [-abs_max, abs_max]
                cmap_abs = plt.cm.RdBu
            
            if (var == 'SHFLX') or (var == 'TREFHT') or (var == 'FSHS') or (var == 'FSNSC'):
                abs_max = np.max([np.abs(np.min(mapdata)),np.abs(np.max(mapdata))])
                clim_abs = [-abs_max, abs_max]
                cmap_abs = plt.cm.RdBu_r
            
            if (var == 'BOWEN'):
                clim_abs = [-100,100]
            elif (var == 'EVAPFRAC'):
                clim_abs = [-1,1]
            elif (var == 'MSE'):
                clim_abs = [-150, 0]
                
            fig, axes = plt.subplots(1, 1, figsize=(6,4))
            
            ax = fig.gca()
            
            mapdata = mapdata*bareground_mask
            mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
            
            mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,var,'moll',title=ttl_main,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units )
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)

            # Annotate with season, variable, date
            ax.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop +', '+var,fontsize='10',
                     ha = 'left',va = 'center',
                     transform = ax.transAxes)
            
            plt.show() 
            
            fig_name = figpath+'/sensitivity/'+filename+'.png'
            fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches='tight', 
                        pad_inches=0.1,frameon=None)
    
            
            plt.close()


 #%%

#%% Sensitivity Plots
            
cm_lnd = plt.cm.viridis
cm_atm = plt.cm.viridis

do_sens_plots=1

sens_plot_vars = ['TREFHT','SHFLX','LHFLX','FSNS','MSE','PRECIP','BOWEN','EVAPFRAC']
            
if do_sens_plots==1:
        
    for var in sens_plot_vars:
        # Loop over properties:
        for prop in sfc_props: 
            
            # set appropriate colour limits
            if prop =='alb':
                clim_dlnd = [-0.01, 0.01]
                clim_datm = [-25,25]
    
                
                units='unitless'
                
                cm_dlnd = plt.cm.viridis_r
                cm_datm = plt.cm.viridis_r
                
            elif prop =='hc':
                clim_dlnd = [-0.5,0.5]
                clim_datm = [-0.5,0.5]
                units='m'
                
                cm_dlnd = plt.cm.viridis_r
                cm_datm = plt.cm.viridis_r
                
            elif prop=='rs' :
                clim_dlnd = [-15.,15.]
                clim_datm = [-.025,.025]
                
                cm_dlnd = plt.cm.viridis_r
                cm_datm = plt.cm.viridis_r
                
                units='s/m'
            elif prop =='log_hc':
                clim_dlnd = [-0.5,0.5]
                clim_datm = [-0.25,0.25]
                units='m'
                
                cm_dlnd = plt.cm.viridis_r
                cm_datm = plt.cm.viridis_r
            
            # Loop over seasons:
            seasons = ['ANN']
            for sea in seasons:
             #   #%% ALBEDO - Unmasked
                
                myvar = var    
             
                cm_dlnd = plt.cm.viridis_r
                cm_datm = plt.cm.viridis_r
    
                #prop = 'alb'
                #myvar = 'TREFHT'
                ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
                mask_name = 'nomask'
                
                #sea = 'ANN'
                
                mapdata_slope = slope[var][sea][prop]
                mapdata_inv = slope[var][sea][prop]**(-1)
               # mapdata_inv = inv_slope[var][sea][prop]
                mapdata_r2 = r_value[var][sea][prop]**2
                
                # Loop through properties and reset ranges to reflect most areas; 
                # also set datm/dlnd to be a change per reasonable amount of sfc
                # and multiply mapdata accordingly, e.g. albedo instead of datm/d(1 alb), do datm/d(0.1 alb) 
                # for dlnd/datm, do dlnd requred for an 0.1 delta T
                if prop =='alb':
                    clim_dlnd = [-0.01, 0.0]
                    clim_datm = [-0.25, 0.0]
                    
                    mapdata_inv = mapdata_inv*0.1   # dlnd/datm = per 1K, make it per 0.1 K 
                    mapdata_slope = mapdata_slope*0.01   # datm/d(1 alb) -> datm/(d(0.01 alb))
                    
                    increment = 0.01
                    
                    cm_dlnd = copy(plt.cm.viridis)
                    cm_datm = copy(plt.cm.viridis_r)
                    
                    #cm_dlnd = plt.cm.hsv
                    #cm_datm = plt.cm.hsv
                    
                    units='unitless'
                elif prop =='hc':
                    clim_dlnd = [-2.,2.]
                    clim_datm = [-0.3,0.3]
                    
                    mapdata_inv = mapdata_inv*0.1   # dlnd/datm = per 1K, make it per 0.1 K 
                    mapdata_slope = mapdata_slope*0.5   # datm/d(1 m hc) -> datm/(d(0.5 m hc))
                    
                    increment = 0.5
                    
    #                cm_dlnd = plt.cm.viridis_r
    #                cm_datm = plt.cm.viridis_r
                    
                
                    cm_dlnd = plt.cm.terrain
                    cm_datm = plt.cm.terrain
                    
                    cm_dlnd = plt.cm.hsv#mml_cmap('wrbw')
                    cm_datm = copy(plt.cm.RdBu_r)
                    cm_dlnd = copy(plt.cm.RdBu_r)
                    #cm_dlnd = mml_cmap('wrbw')
                
                    
                    #cm_dlnd = plt.cm.plasma_r
                    #cm_datm = plt.cm.plasma_r
                    
                    units='m'
                elif prop=='rs' :
                    clim_dlnd = [0.0, 50.0]
                    clim_datm = [0.0, 0.18]
                    
                    mapdata_inv = mapdata_inv*0.1   # dlnd/datm = per 1K, make it per 0.1 K 
                    mapdata_slope = mapdata_slope*10   # datm/d(1 s/m rs ) -> datm/(d(10 s/m rs))
                    
                    increment = 10
                    
                    cm_dlnd = copy(plt.cm.viridis_r)
                    cm_datm = copy(plt.cm.viridis)
                    
                    cm_datm.set_under('k',0.0)
                    cm_dlnd.set_under('k',0.0)
                    
                    #cm_dlnd = plt.cm.terrain
                    #cm_datm = plt.cm.terrain
                    
                    units='s/m'
                elif prop =='log_hc':
                    clim_dlnd = [-2.5,2.5]
                    clim_datm = [-0.25,0.25]
                    
                    mapdata_inv = mapdata_inv*0.1   # dlnd/datm = per 1K, make it per 0.1 K 
                    mapdata_slope = mapdata_slope*0.5   # datm/d(1 m hc) -> datm/(d(0.5 m hc))
                    
                    increment = 0.5
                    
                    cm_dlnd = plt.cm.viridis_r
                    cm_datm = plt.cm.viridis_r
                    
                    #cm_dlnd = plt.cm.plasma_r
                    #cm_datm = plt.cm.plasma_r
                    
                    units='m'
                
                
                ttl_main = prop #'Albedo'
                filename = 'sens_slopes_'+var+'_'+prop+'_'+mask_name+'_'+sea
                
                
                cmap_abs = plt.cm.viridis
                cmap_diff = plt.cm.RdBu_r
                
                
                fig, axes = plt.subplots(1, 3, figsize=(18,6))
                
                NCo = 21
                NTik_dlnd = 5
                NTik_datm = 9#np.floor(NCo/2)
                NTik_r2 = 6
                
                ax0 = axes.flatten()[0]
                plt.sca(ax0)
                ttl = '$\delta$ '+prop+' per 0.1K change in T2m'
                #units = 'unitless'
                #clim_diff = [-.01,.01]
                mapdata = mapdata_inv
                #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units )   #plt.cm.BuPu_r
                #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
                mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units,ext='both')# ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
                ax=ax0
                
                if prop == 'alb':
                    # ticks are running into each other
                    tick_locator = ticker.MaxNLocator(nbins=6)
                    cbar.locator = tick_locator
                    cbar.update_ticks()
                
                #if (prop =='hc') or (prop=='log_hc'):
                #    cbar.norm()
                
                
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
                #cbar.set_ticklabels(np.linspace(-0.01,0.01,9))
                ax1 = axes.flatten()[1]
                plt.sca(ax1)
                ttl = '$\delta$ T2m per '+np.str(increment) +' [' +units+'] change in '+prop
                units_K = 'K'
               # clim_diff = [-25,25]
                #clim_abs = clim_diff
                mapdata = mapdata_slope
                #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units )   #plt.cm.BuPu_r
                #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_datm,ext='both',disc=True )
                mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,var,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units_K,ext='both')
                ax=ax1
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
                
                #if prop == 'rs':
                if prop == prop:
                    # ticks are running into each other
                    tick_locator = ticker.MaxNLocator(nbins=6)
                    cbar.locator = tick_locator
                    cbar.update_ticks()
                
                ax2 = axes.flatten()[2]
                plt.sca(ax2)
                ttl = 'r^2'
                units_r = 'r^2'
                clim_abs = [0.5,1]
                mapdata = mapdata_r2
                #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units )   #plt.cm.BuPu_r
                #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_r2,ext='min')
                mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units_r, ncol=NCo,nticks=NTik_r2,ext='min')
                ax=ax2
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
                
                fig.subplots_adjust(top=1.15)
                fig.suptitle(ttl_main, fontsize=20)   
                
                # Annotate with season, variable, date
                ax0.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                         ha = 'left',va = 'center',
                         transform = ax0.transAxes)
                
    #            fig_name = figpath+'/sensitivity/'+filename+'.eps'
    #            fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
    #                        edgecolor='w',orientation='portrait',bbox_inches='tight', 
    #                        pad_inches=0.1,frameon=None)
                
                plt.show()
                
                fig_name = figpath+'/sensitivity/new_colourbar/'+filename+'.png'
                fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches='tight', 
                            pad_inches=0.1,frameon=None)
    
               
                plt.close()
                
                
            #    #%% ALBEDO - Land Mask
            
            
                #prop = 'alb'
                #myvar = 'TREFHT'
                ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
                mask_name = 'lndmask'
                #sea = 'ANN'
                
    #            mapdata_slope = slope[sea][prop]
    #            mapdata_inv = slope[sea][prop]**(-1)
    #            mapdata_r2 = r_value[sea][prop]**2
                
                
                ttl_main = prop #'Albedo'
                filename = 'sens_slopes_'+var+'_'+prop+'_'+mask_name+'_'+sea
                
                
                cmap_abs = plt.cm.get_cmap('viridis',11)#plt.cm.viridis()
                cmap_diff = plt.cm.RdBu_r
                
                fig, axes = plt.subplots(1, 3, figsize=(18,6))
                
                ax0 = axes.flatten()[0]
                plt.sca(ax0)
                ttl = '$\delta$ '+prop+' per 0.1K change in T2m'
                #units = 'unitless'
                #clim_diff = [-.01,.01]
                mapdata = mapdata_inv*bareground_mask
                mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
                #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik,ext='both',disc=True )   #plt.cm.BuPu_r
                mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units,ext='both')#, ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
                ax=ax0
               # mml_map(LN,LT,mapdata,ds,myvar,proj,title=None,clim=None,colmap=None,cb_ttl=None,disc=None,ncol=None,nticks=None,ext=None):
               
                if prop == 'alb':
                    # ticks are running into each other
                    tick_locator = ticker.MaxNLocator(nbins=6)
                    cbar.locator = tick_locator
                    cbar.update_ticks()
                    
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
                
                ax1 = axes.flatten()[1]
                plt.sca(ax1)
                ttl = '$\delta$ T2m per '+np.str(increment) +' [' +units+'] change in '+prop
                units_K = 'K'
                #clim_diff = [-25,25]
                #clim_abs = clim_diff
                mapdata = mapdata_slope*bareground_mask
                mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
                #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units )   #plt.cm.BuPu_r
                mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units_K,ext='both')#, ncol=NCo,nticks=NTik_datm,ext='both',disc=True )
                ax=ax1
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
                    
                #if prop == 'rs':
                if prop == prop:
                    # ticks are running into each other
                    tick_locator = ticker.MaxNLocator(nbins=6)
                    cbar.locator = tick_locator
                    cbar.update_ticks()
                    
                    
                ax2 = axes.flatten()[2]
                plt.sca(ax2)
                ttl = 'r^2'
                units_r = 'r^2'
                clim_abs = [0.5,1]
                mapdata = mapdata_r2*bareground_mask
                mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
                #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units )   #plt.cm.BuPu_r
                mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units_r, ncol=NCo,nticks=NTik_r2,ext='min')
                ax=ax2
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
                
                fig.subplots_adjust(top=1.15)
                fig.suptitle(ttl_main, fontsize=20)    
                
                # Annotate with season, variable, date
                ax0.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                         ha = 'left',va = 'center',
                         transform = ax0.transAxes)
                
                
    #            fig_name = figpath+'/sensitivity/'+filename+'.eps'
    #            fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
    #                        edgecolor='w',orientation='portrait',bbox_inches='tight', 
    #                        pad_inches=0.1,frameon=None)
    #            
                fig_name = figpath+'/sensitivity/new_colourbar/'+filename+'.png'
                fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches='tight', 
                            pad_inches=0.1,frameon=None)
                
                
                # Save the sub-plots as individual panels
                
                # (a) dlnd/datm
                extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                bbx = extent.extents
                bbx[0]=bbx[1]-0.25
                bbx[1]=bbx[1]-0.5
                bbx[2]=bbx[2]+0.25
                bbx[3]=bbx[3]+0.2
                fig_png = figpath+'/sensitivity/subplots/'+filename+'_a.png'
                fig_eps = figpath+'/sensitivity/subplots/'+filename+'_a.eps'
                vals = extent.extents
                new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
                fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                            frameon=None)
                fig.savefig(fig_eps,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                            frameon=None)
                
                # (b) datm/dlnd
                # add datetime tag
                # Annotate with season, variable, date
                ax1.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                         ha = 'left',va = 'center',
                         transform = ax1.transAxes)
                extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig_png = figpath+'/sensitivity/subplots/'+filename+'_b.png'
                fig_eps = figpath+'/sensitivity/subplots/'+filename+'_b.eps'
                vals = extent.extents
                new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
                fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                            frameon=None)
    #            fig.savefig(fig_eps,dpi=600,transparent=True,facecolor='w',
    #                        edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
    #                        frameon=None)
                
                # (c) r^2
                # Annotate with season, variable, date
                ax2.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                         ha = 'left',va = 'center',
                         transform = ax2.transAxes)
                extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig_png = figpath+'/sensitivity/subplots/'+filename+'_c.png'
                fig_eps = figpath+'/sensitivity/subplots/'+filename+'_c.eps'
                vals = extent.extents
                new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
                fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                            frameon=None)
    #            fig.savefig(fig_eps,dpi=600,transparent=True,facecolor='w',
    #                        edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
    #                        frameon=None)
                
                plt.close()
            




 #%%






 #%%






        