#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:39:05 2017

@author: mlague

Scatter Plots of sensitivities in Global Perturbation experiments, focussing on 
how standard deviation (how BIG is the response?) correlates with r^2 values
(are places with relatively low responses places with bad linearity?)

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:15:47 2017

@author: mlague

Sensitivities datm/dlnd and dlnd/datm (and r2) for global perturbation simulations
of albedo, roughness, and evaporative resistance (lid resistance; do total 
effective resistance elsewhere), for annual mean and seasonal. 

Or thats the goal, anyhow.

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
figpath = '/home/disk/eos18/mlague/simple_land/scripts/python/analysis/global_pert/figures_old/'

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
prop = {}
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
sfc_props = ['alb','rs']
#sfc_prop_ranges = np.array([ [0.1, 0.2, 0.3],
#                  [0.5, 1., 2.],
#                  [30., 100., 200.],
#                  [np.log(0.5), np.log(1.), np.log(2.)]])
sfc_prop_ranges = np.array([ [0.1, 0.2, 0.3,np.nan,np.nan],
                            [30., 100., 200.,np.nan,np.nan],
                            [0.01,0.05,0.1,0.5, 1., 2.],
                            [np.log(0.01),np.log(0.05),np.log(0.1),np.log(0.5), np.log(1.), np.log(2.)]])
sfc_prop_ranges = np.array([ [0.1, 0.2, 0.3],
                            [30., 100., 200.]])

    
print(np.shape(sfc_prop_ranges))

print(sfc_prop_ranges)

seasons = ['ANN','DJF','MAM','JJA','SON']

slope_vars = ['TREFHT','SHFLX','LHFLX','FSNT','FSNTC','FLNT','FLNTC','FSNS','FSNSC','FLNS','FLNSC','PRECC','PRECL','PRECSC','PRECSL']
    
ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']



for atm_var in slope_vars:

    units[atm_var] = ds0[atm_var].units
    
    atm_resp[atm_var] = {}
    print(atm_var)
    
    for sea in seasons: 
        atm_resp[atm_var][sea] = {}

    i = 0
    for prop in sfc_props:
        pert[prop] = sfc_prop_ranges[i]
        
#        if np.isnan(pert[prop][3]):
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
        
#        else:
#            print(prop)
#            ds1 = ds_low1[prop] #0.01
#            ds2 = ds_low2[prop]  #0.05
#            ds3 = ds_med1[prop]  #0.1
#            ds4 = ds_med2[prop]    #0.5
#            ds5 = ds_high1[prop]    #1
#            ds6 = ds_high2[prop]    #2
#            
#            # annual mean response
#            atm_resp_ann[prop] = np.array([np.array(ds1.mean('time')[atm_var].values[:,:]),
#                np.array(ds2.mean('time')[atm_var].values[:,:]),
#                np.array(ds3.mean('time')[atm_var].values[:,:]),
#                np.array(ds4.mean('time')[atm_var].values[:,:]),
#                np.array(ds5.mean('time')[atm_var].values[:,:]),
#                np.array(ds6.mean('time')[atm_var].values[:,:]),
#                ])
#        
#            # seasonal responses:
#            # (first, make 12 month response, then average over djf, jja, etc)
#            #print(np.shape(ds1[atm_var].values))
#            resp_mths = np.array([np.array(ds1[atm_var].values[:,:,:]),
#                    np.array(ds2[atm_var].values[:,:,:]),
#                    np.array(ds3[atm_var].values[:,:,:]),
#                    np.array(ds4[atm_var].values[:,:,:]),
#                    np.array(ds5[atm_var].values[:,:,:]),
#                    np.array(ds6[atm_var].values[:,:,:]),
#                    ])
        
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
        
        
        #print(np.shape(atm_resp_djf[prop]))
        i=i+1

#%%
        """
            DERIVED VARIABLES
        """
# Make precip & Column MSE a thing:

atm_resp['PRECIP'] = {}
atm_resp['MSE'] = {}  
atm_resp['EVAPFRAC'] = {}  
atm_resp['BOWEN'] = {}  
        
for sea in seasons: 
    atm_resp['PRECIP'][sea] = {}
    atm_resp['MSE'][sea] = {}
    atm_resp['EVAPFRAC'][sea] = {}
    atm_resp['BOWEN'][sea] = {}

    i = 0
    for prop in sfc_props:
        
        atm_resp['PRECIP'][sea][prop] = np.array(   atm_resp['PRECC'][sea][prop]  + 
                                                    atm_resp['PRECL'][sea][prop]  +
                                                    atm_resp['PRECSC'][sea][prop] + 
                                                    atm_resp['PRECSL'][sea][prop] )
        
        # check if MSE should include SW at sfc                                            
        atm_resp['MSE'][sea][prop] = np.array(
                                        (atm_resp['FSNT'][sea][prop] - atm_resp['FLNT'][sea][prop]) - 
                                        ( (atm_resp['FLNS'][sea][prop] )
                                        + atm_resp['SHFLX'][sea][prop] + atm_resp['LHFLX'][sea][prop]) )
        
        # EVAPORATIVE FRACTION
        atm_resp['EVAPFRAC'][sea][prop] = np.array(
                                        (atm_resp['LHFLX'][sea][prop]) / 
                                         (atm_resp['SHFLX'][sea][prop] + atm_resp['LHFLX'][sea][prop]))
        
        # Bowen Ratio
        atm_resp['BOWEN'][sea][prop] = np.array(
                                        (atm_resp['SHFLX'][sea][prop]) / 
                                         atm_resp['LHFLX'][sea][prop]   )
        
        
        
        i=i+1
        
        
    
plt.imshow(atm_resp['FSNT']['ANN']['alb'][1,:,:])
plt.colorbar()    
plt.show()
plt.close()

plt.imshow(atm_resp['FLNT']['ANN']['alb'][1,:,:])
plt.colorbar()  
plt.show()
plt.close()  

plt.imshow(atm_resp['FSNS']['ANN']['alb'][1,:,:])
plt.colorbar()    
plt.show()
plt.close()

plt.imshow(atm_resp['FLNS']['ANN']['alb'][1,:,:])
plt.colorbar()  
plt.show()
plt.close() 

plt.imshow(atm_resp['MSE']['ANN']['alb'][1,:,:])
plt.colorbar()  
plt.show()
plt.close()  

#print(prop)
#print(pert)
#print(atm_resp)
#print(pert['alb'])



#%%

if do_slope_analysis==1:
    # sklearn linear_model , pearson r
    
    # Test more than just TREFHT ... try SHFLX, LHFLX for example, too
    
    slope_vars = ['TREFHT','SHFLX','LHFLX','FSNT','FSNTC','FLNT','FLNTC','FSNS','FSNSC','FLNS','FLNSC','PRECIP','MSE','EVAPFRAC']
    
    # Annual and seasonal... ought to be able to nest, non? 
    slope = {}
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
        intercept[var] = {}
        r_value[var] = {}
        p_value[var] = {}
        std_err[var] = {}
        
        std_dev[var] = {}
        
        sensitivity[var] = {}
        
        seasons = ['ANN','DJF','MAM','JJA','SON']
        
        for sea in seasons:
            slope[var][sea] = {}
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
#                if np.isnan(xvals[4]):  # they were forced to all be size 6 b/c of roughness. If those >3 are nan, set k to 3.
#                    k = 3
#                    xvals = xvals[0:3]
                    
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
                
                del raw_data, raw_data_v
            
            

#%% Plot max/min and std dev to make sure they look plausible (same order of magnitude, at least)

for prop in sfc_props: 

    cm_sig = plt.cm.viridis
    clim_diff = [0,0.5]
    
    var = 'TREFHT'
           
    fig, axes = plt.subplots(1, 1, figsize=(5,3))
                
    #ax0 = axes.flatten()[0]
    ax0 = axes
    plt.sca(ax0)
    ttl = '$\sigma$ T2m, '+prop
    units = 'K'
    #clim_diff = [-.01,.01]
    #mapdata = mapdata_inv*0.1*ocn_glc_mask
    mapdata = std_dev[var][sea][prop]
    mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
    #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units )   #plt.cm.BuPu_r
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,var,'moll',title=ttl,clim=clim_diff,colmap=cm_sig, cb_ttl='units: '+units)
    ax=ax0
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
     
#    ax1 = axes.flatten()[1]
#    plt.sca(ax1)
#    ttl = 'max-min perturbation T2m, '+prop
#    units = 'K'
#    #clim_diff = [-.01,.01]
#    #mapdata = mapdata_inv*0.1*ocn_glc_mask
#    resp = np.array(atm_resp[sea][prop])
#    mapdata = np.squeeze(resp[-1,:,:] - resp[0,:,:])/2
#    mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
#    #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units )   #plt.cm.BuPu_r
#    mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_diff,colmap=cm_sig, cb_ttl='units: '+units)
#    ax=ax1
#    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#        item.set_fontsize(12)
 
               
#%%
    
"""
        New metrics: 
            1.  scatter plot sensitivity to evaporative resistance against 
                annual mean bucket "fullness" ...
            2.  
"""





#%%
"""
        LAND AREAS ONLY
"""
 #%%
"""
        Sensitivity to evaporative resistane against bucket fullness
            yvar: dT2m / d rs
            xvar: annual mean water ( in baseline a2 hc0.1 rs100 run )
"""
    
units = {}

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


var = 'TREFHT'



for prop in sfc_props:
    
    for sea in ['ANN']:
    
        yvar = std_dev[var][sea][prop].flatten()
        
        xvar = r_value[var][sea][prop].flatten()
        units['y'] = 'K'
        units['x'] = 'r$^2$'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        p1 = ax.scatter(xvar,yvar,s=2,color='blue',alpha=0.1)
        
        ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('r^2 ['+units['x']+']')
        ax.set_ylabel('$\sigma$ ['+units['y']+']')
        plt.title('Scatter: standard deviation vs linearity ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.15,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_std_vs_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')

"""
Next: take std in time, see what average temperature variability in time of these locations is... though 
maybe that changes when I change hc? Should still just run a t-test and see what falls 
out as significant. 
"""        
#%%
    
#%%
"""
        Sensitivity in T vs P space, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: dT2m / d lnd
"""
    
units = {}

var = 'TREFHT'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_precip = ds0_cam['PRECC'].values + ds0_cam['PRECL'].values + ds0_cam['PRECSC'].values + ds0_cam['PRECSL'].values
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
yvar = ann_precip.flatten()

ctrl_t2m = ds0_cam['TREFHT'].values 
ann_t2m = np.nanmean(ctrl_t2m,0) * no_glc
xvar = ann_t2m.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'mm/yr'
        units['x'] = 'K'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.viridis_r
        if prop == 'alb':
            cmap_scat = plt.cm.viridis
            scale_factor = 0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 10.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=2,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=2,alpha=0.9,cmap=cmap_scat , edgecolors='none')
        
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+np.str(scale_factor)+' ' +prop+'); [K]/['+prop_units+']')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean Temperature ['+units['x']+']')
        ax.set_ylabel('Annual Mean Precipitation['+units['y']+']')
        plt.title('Scatter: sensitivity dT2m / d ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_precip_vs_T_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')

#%%
"""
        Sensitivity in T vs P space, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: dT2m / d lnd
"""
    
units = {}

var = 'TREFHT'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_precip = ds0_cam['PRECC'].values + ds0_cam['PRECL'].values + ds0_cam['PRECSC'].values + ds0_cam['PRECSL'].values
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
yvar = ann_precip.flatten()

ctrl_t2m = ds0_cam['FSNS'].values 
ann_t2m = np.nanmean(ctrl_t2m,0) * no_glc
xvar = ann_t2m.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'mm/yr'
        units['x'] = 'W/m^2'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.viridis_r
        if prop == 'alb':
            cmap_scat = plt.cm.viridis
            scale_factor = 0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 10.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=2,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=2,alpha=0.9,cmap=cmap_scat , edgecolors='none')
        
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+np.str(scale_factor)+' ' +prop+'); [K]/['+prop_units+']')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean Shortwave Absorbed ['+units['x']+']')
        ax.set_ylabel('Annual Mean Precipitation['+units['y']+']')
        plt.title('Scatter: sensitivity dT2m / d ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_precip_vs_fsns_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')

#%%
"""
        Sensitivity in cloudiness vs SW in at TOA, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: dT2m / d lnd
"""
    
units = {}

var = 'TREFHT'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_toa_alb = ds0_cam['FSUTOA'].values[:] / (ds0_cam['FSNTOA'].values[:] + ds0_cam['FSUTOA'].values[:]) 
plt.imshow(np.mean(ctrl_toa_alb,0))
plt.colorbar()
plt.show()
plt.close()
ann_toa_alb = np.nanmean(ctrl_toa_alb,0) * no_glc 
yvar = ann_toa_alb.flatten()

ctrl_fsntoa = ds0_cam['FSNTOA'].values 
ann_fsntoa = np.nanmean(ctrl_fsntoa,0) * no_glc
xvar = ann_fsntoa.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'albedo'
        units['x'] = 'W/m^2'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.get_cmap('RdPu',10)
        if prop == 'alb':
           # cmap_scat = plt.cm.viridis
            cmap_scat = plt.get_cmap('RdPu_r',10)
            scale_factor = 0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 10.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=3,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=3,alpha=0.9,cmap=cmap_scat , edgecolors='none')
        
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+np.str(scale_factor)+' ' +prop+'); [K]/['+prop_units+']')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_ylabel('Annual Mean TOA albedo')
        ax.set_xlabel('Annual Mean FSNTOA ['+units['x']+']')
        plt.title('Scatter: sensitivity dT2m / d ('+prop+')')
    
        ax.set_xlim([50,350])
        ax.set_ylim([0.1,0.7])


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_toa_alb_vs_fsntoa_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')

#%%
"""
        Sensitivity in cloudiness vs SW in at TOA, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: dT2m / d lnd
            
            I think the above may be compounding snow effects. This *shoudl* avoid that. Just be clouds.
"""
    
units = {}

var = 'TREFHT'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_toa_alb = ds0_cam['FSNTOAC'].values[:]- ds0_cam['FSNTOA'].values[:]
plt.imshow(np.mean(ctrl_toa_alb,0))
plt.colorbar()
plt.show()
plt.close()
ann_toa_alb = np.nanmean(ctrl_toa_alb,0) * no_glc 
yvar = ann_toa_alb.flatten()

ctrl_fsntoa = ds0_cam['FSNS'].values 
ann_fsntoa = np.nanmean(ctrl_fsntoa,0) * no_glc
xvar = ann_fsntoa.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'W/m$^2$'
        units['x'] = 'W/m^2'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.BuPu
        cmap_scat = plt.get_cmap('RdPu',10)
        if prop == 'alb':
            cmap_scat = plt.cm.BuPu_r
            cmap_scat = plt.get_cmap('RdPu_r',10)
            scale_factor = 0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 10.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=2,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=2,alpha=0.9,cmap=cmap_scat , edgecolors='none')
        
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+np.str(scale_factor)+' ' +prop+'); [K]/['+prop_units+']')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_ylabel('Annual Mean FSNTOAC - FSNTOA (cloud forcing, W/m$^2$)')
        ax.set_xlabel('Annual Mean FSNS ['+units['x']+']')
        plt.title('Scatter: sensitivity dT2m / d ('+prop+')')
    
        ax.set_xlim([0,270])
        ax.set_ylim([0,140])

        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_fsntc_m_fsnt_vs_fsns_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')

#%%
"""
        Sensitivity in cloudiness vs SW in at TOA, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: dT2m / d lnd
            
            I think the above may be compounding snow effects. This *shoudl* avoid that. Just be clouds.
"""
    
units = {}

var = 'TREFHT'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_toa_alb = ds0_cam['FSNSC'].values[:]- ds0_cam['FSNS'].values[:]
plt.imshow(np.mean(ctrl_toa_alb,0))
plt.colorbar()
plt.show()
plt.close()
ann_toa_alb = np.nanmean(ctrl_toa_alb,0) * no_glc 
yvar = ann_toa_alb.flatten()

ctrl_fsntoa = ds0_cam['FSNS'].values 
ann_fsntoa = np.nanmean(ctrl_fsntoa,0) * no_glc
xvar = ann_fsntoa.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'W/m$^2$'
        units['x'] = 'W/m^2'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.BuPu
        cmap_scat = plt.get_cmap('RdPu',10)
        if prop == 'alb':
            cmap_scat = plt.cm.BuPu_r
            cmap_scat = plt.get_cmap('RdPu_r',10)
            scale_factor = 0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 10.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=2,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=2,alpha=0.9,cmap=cmap_scat , edgecolors='none')
        
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+np.str(scale_factor)+' ' +prop+'); [K]/['+prop_units+']')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_ylabel('Annual Mean FSNSC - FSNS (cloud forcing, W/m$^2$)')
        ax.set_xlabel('Annual Mean FSNS ['+units['x']+']')
        plt.title('Scatter: sensitivity dT2m / d ('+prop+')')
    
        ax.set_xlim([0,270])
        ax.set_ylim([0,140])

        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_fsnsc_m_fsns_vs_fsns_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')
        
#%%
"""
        Sensitivity in T vs P space, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: dT2m / d lnd
"""
    
units = {}

var = 'TREFHT'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_precip = ds0_cam['PRECC'].values + ds0_cam['PRECL'].values + ds0_cam['PRECSC'].values + ds0_cam['PRECSL'].values
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
yvar = ann_precip.flatten()

ctrl_t2m = ds0_clm['MML_water'].values 
ann_t2m = np.nanmean(ctrl_t2m,0) * no_glc
xvar = ann_t2m.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'mm/yr'
        units['x'] = 'mm'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.RdPu
        if prop == 'alb':
            cmap_scat = plt.cm.RdPu
            scale_factor = -0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 50.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=3,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
        if prop == 'alb':
            p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=3,alpha=0.9,cmap=cmap_scat , edgecolors='none',vmin=0,vmax=3)
        else:
            p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=3,alpha=0.9,cmap=cmap_scat , edgecolors='none',vmin=0,vmax=.8)
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+np.str(scale_factor)+' ' +prop+'); [K]/['+prop_units+']')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
#        if prop=='alb':
#            cbar.set_clim(0,3)
            
#            cs = plt.get_cmap
#            cs.set_clim(0,3)
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean water in bucket ['+units['x']+']')
        ax.set_ylabel('Annual Mean Precipitation['+units['y']+']')
        plt.title('Scatter: sensitivity dT2m / d ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_precip_vs_bucket_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')

#%%
"""
        Sensitivity in T vs P space, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: dT2m / d lnd
"""
    
units = {}

var = 'TREFHT'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_precip = ds0_cam['PRECC'].values + ds0_cam['PRECL'].values + ds0_cam['PRECSC'].values + ds0_cam['PRECSL'].values
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
yvar = ann_precip.flatten()

ctrl_t2m = ds0_cam['TREFHT'].values 
ann_t2m = np.nanmean(ctrl_t2m,0) * no_glc
xvar = ann_t2m.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'mm/yr'
        units['x'] = 'mm'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.RdPu
        if prop == 'alb':
            cmap_scat = plt.cm.RdPu
            scale_factor = -0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 10.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=3,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
#        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=3,alpha=0.9,cmap=cmap_scat , edgecolors='none')
        if prop == 'alb':
            p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=3,alpha=0.9,cmap=cmap_scat , edgecolors='none',vmin=0,vmax=3)
        else:
            p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=3,alpha=0.9,cmap=cmap_scat , edgecolors='none',vmin=0,vmax=.8)
        
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+np.str(scale_factor)+' ' +prop+'); [K]/['+prop_units+']')
        
#       # cb2.set_label('Non-linear (r^2 < 0.8)')
#       if prop=='alb':
#            cbar.set_clim(0,3)
#            cmap_scat.set_clim(0,3)
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean TREFHT ['+units['x']+']')
        ax.set_ylabel('Annual Mean Precipitation['+units['y']+']')
        plt.title('Scatter: sensitivity dT2m / d ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_precip_vs_TREFHT_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')

#%%
"""
        Sensitivity in T vs P space, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: dT2m / d lnd
"""
    
units = {}

var = 'TREFHT'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_precip = ds0_cam['PRECC'].values + ds0_cam['PRECL'].values + ds0_cam['PRECSC'].values + ds0_cam['PRECSL'].values
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
yvar = ann_precip.flatten()

ctrl_t2m = ds0_cam['FSNS'].values 
ann_t2m = np.nanmean(ctrl_t2m,0) * no_glc
xvar = ann_t2m.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'mm/yr'
        units['x'] = 'mm'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.RdPu
        if prop == 'alb':
            cmap_scat = plt.cm.RdPu
            scale_factor = -0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 10.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=3,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
#        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=3,alpha=0.9,cmap=cmap_scat , edgecolors='none')
        if prop == 'alb':
            p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=3,alpha=0.9,cmap=cmap_scat , edgecolors='none',vmin=0,vmax=3)
        else:
            p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=3,alpha=0.9,cmap=cmap_scat , edgecolors='none',vmin=0,vmax=.8)
        
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+np.str(scale_factor)+' ' +prop+'); [K]/['+prop_units+']')
        
        # cb2.set_label('Non-linear (r^2 < 0.8)')
#        if prop=='alb':
#            cbar.set_clim(0,3)
#            cmap_scat.set_clim(0,3)
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean FSNS ['+units['x']+']')
        ax.set_ylabel('Annual Mean Precipitation['+units['y']+']')
        plt.title('Scatter: sensitivity dT2m / d ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_precip_vs_FSNS_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')


#%%
"""
        Sensitivity in T vs P space, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: dT2m / d lnd
"""
    
units = {}

var = 'TREFHT'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_precip = ds0_cam['TREFHT'].values # + ds0_cam['PRECL'].values + ds0_cam['PRECSC'].values + ds0_cam['PRECSL'].values
ann_precip = np.nanmean(ctrl_precip,0) * no_glc 
yvar = ann_precip.flatten()

ctrl_t2m = ds0_clm['MML_water'].values 
ann_t2m = np.nanmean(ctrl_t2m,0) * no_glc
xvar = ann_t2m.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'K'
        units['x'] = 'mm'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.viridis_r
        if prop == 'alb':
            cmap_scat = plt.cm.viridis
            scale_factor = 0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 10.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=2,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=2,alpha=0.9,cmap=cmap_scat , edgecolors='none')
        
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+np.str(scale_factor)+' ' +prop+'); [K]/['+prop_units+']')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean water in bucket ['+units['x']+']')
        ax.set_ylabel('Annual Mean Temperature ['+units['y']+']')
        plt.title('Scatter: sensitivity dT2m / d ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_T_vs_bucket_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')        
#%%
"""
        Sensitivity in T vs P space, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: dT2m / d lnd
"""
    
units = {}

var = 'TREFHT'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_precip = ds0_cam['TREFHT'].values #+ ds0_cam['PRECL'].values + ds0_cam['PRECSC'].values + ds0_cam['PRECSL'].values
ann_precip = np.nanmean(ctrl_precip,0) * no_glc 
yvar = ann_precip.flatten()

ctrl_t2m = ds0_cam['FSNS'].values 
ann_t2m = np.nanmean(ctrl_t2m,0) * no_glc
xvar = ann_t2m.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'K'
        units['x'] = 'W/m^2'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.viridis_r
        if prop == 'alb':
            cmap_scat = plt.cm.viridis
            scale_factor = 0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 10.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=2,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=2,alpha=0.9,cmap=cmap_scat , edgecolors='none')
        
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+np.str(scale_factor)+' ' +prop+'); [K]/['+prop_units+']')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean Shortwave Absorbed ['+units['x']+']')
        ax.set_ylabel('Annual Mean Temperature ['+units['y']+']')
        plt.title('Scatter: sensitivity dT2m / d ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_T_vs_fsns_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')


#%%
        
        """
            SENSIBLE HEAT FLUX
        """
        
        
#%%
 #%%
"""
        Sensitivity to evaporative resistane against bucket fullness
            yvar: dT2m / d rs
            xvar: annual mean water ( in baseline a2 hc0.1 rs100 run )
"""
     
units = {}

var = 'SHFLX'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_precip = ds0_cam['PRECC'].values + ds0_cam['PRECL'].values + ds0_cam['PRECSC'].values + ds0_cam['PRECSL'].values
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
yvar = ann_precip.flatten()

ctrl_t2m = ds0_cam['TREFHT'].values 
ann_t2m = np.nanmean(ctrl_t2m,0) * no_glc
xvar = ann_t2m.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'mm/yr'
        units['x'] = 'K'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.viridis_r
        if prop == 'alb':
            cmap_scat = plt.cm.viridis
            scale_factor = 0.1
            prop_units = 'unitless'
        if prop == 'rs':
            scale_factor = 10.
            prop_units = 's/m'
        if ( prop == 'hc' ) or (prop == 'log_hc'):
            scale_factor = 0.1
            prop_units = 'm'
        
        pcol_dat = pcol_dat * scale_factor
        grey_dat = grey_dat * scale_factor
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=2,cmap=plt.cm.gray_r,alpha=0.6, edgecolors='none')
        
        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=2,alpha=0.9,cmap=cmap_scat , edgecolors='none')
        
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (SH) / d ('+np.str(scale_factor)+' ' +prop+'); [W/m$^2$]/['+prop_units+']')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean Temperature ['+units['x']+']')
        ax.set_ylabel('Annual Mean Precipitation['+units['y']+']')
        plt.title('Scatter: sensitivity dSH / d ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_precip_vs_T_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')
        
#%%
"""
        Sensitivity in T vs P space, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann precip
            xvar: ann T2m
            pcolor: d SHFLX / d lnd
"""
    
units = {}

var = 'SHFLX'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_precip = ds0_cam['PRECC'].values + ds0_cam['PRECL'].values + ds0_cam['PRECSC'].values + ds0_cam['PRECSL'].values
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
yvar = ann_precip.flatten()

ctrl_t2m = ds0_cam['TREFHT'].values 
ann_t2m = np.nanmean(ctrl_t2m,0) * no_glc
xvar = ann_t2m.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'mm/yr'
        units['x'] = 'K'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.viridis_r
        if prop == 'alb':
            cmap_scat = plt.cm.viridis
        
        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=10,alpha=0.3,cmap=cmap_scat , edgecolors='none')
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=10,cmap=plt.cm.gray_r,alpha=0.2, edgecolors='none')
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+prop+') ')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean Temperature ['+units['x']+']')
        ax.set_ylabel('Annual Mean Precipitation['+units['y']+']')
        plt.title('Scatter ' +var+': sensitivity dT2m / d ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_precip_vs_T_masked_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')
#%%
#%%
"""
        Sensitivity in T vs P space, only colouring slope of linear (r^2 > 0.8 ) gridcells. Others in gray. 
            yvar: ann SHFLX
            xvar: ann T2m
            pcolor: d SHFLX / d lnd
"""
    
units = {}

var = 'SHFLX'

ds0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']


ds0_clm = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
ds0_cam = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365

ctrl_precip = ds0_cam['SHFLX'].values # + ds0_cam['PRECL'].values + ds0_cam['PRECSC'].values + ds0_cam['PRECSL'].values
ann_precip = np.nanmean(ctrl_precip,0) * no_glc #* ms2mmyr
yvar = ann_precip.flatten()

ctrl_t2m = ds0_cam['TREFHT'].values 
ann_t2m = np.nanmean(ctrl_t2m,0) * no_glc
xvar = ann_t2m.flatten()


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        #yvar = std_dev[sea][prop].flatten()
        
        #xvar = r_value[sea][prop].flatten()

        
        #pcol_mask = np.where(r_value[sea][prop]>0.8)
        #grey_mask = np.where(r_value[sea][prop]<=0.8)
        
        pcol_mask = np.where(r_value[var][sea][prop]>0.8,1.0,np.nan)
        grey_mask = np.where(r_value[var][sea][prop]<=0.8,1.0,np.nan)
        
        pcol_dat = (slope[var][sea][prop]*pcol_mask*landmask)
        grey_dat = (slope[var][sea][prop]*grey_mask*landmask)
        
        units['y'] = 'W/m2'
        units['x'] = 'W/m2'    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        cmap_scat = plt.cm.viridis_r
        if prop == 'alb':
            cmap_scat = plt.cm.viridis
        
        p1 = ax.scatter(x=xvar,y=yvar,c=pcol_dat.flatten(),s=10,alpha=0.3,cmap=cmap_scat , edgecolors='none')
        
        p2 = ax.scatter(x=xvar,y=yvar,c=grey_dat.flatten(),s=10,cmap=plt.cm.gray_r,alpha=0.2, edgecolors='none')
        
        cb = plt.colorbar(p1)
       # cb2 = plt.colorbar(p2)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+prop+') ')
        
       # cb2.set_label('Non-linear (r^2 < 0.8)')
    
        
        #ax.set_xlim([0,1])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean FSNS ['+units['x']+']')
        ax.set_ylabel('Annual Mean SHFLX ['+units['y']+']')
        plt.title('Scatter ' +var+': sensitivity d'+var+' / d ('+prop+')')
    


        # Annotate with season, variable, date
        ax.text(0.,-0.25,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        plt.show()    
        
        fig_png = figpath + '/sensitivity/std/'+var+'_'+prop+'_'+sea+'_shflx_vs_fsns_for_r2.png'
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight')
#%%
"""
    Next: take std in time, see what average temperature variability in time of these locations is... though 
    maybe that changes when I change hc? Should still just run a t-test and see what falls 
    out as significant. 
"""         
        
#%%
"""
    Plotting deltas. This isn't the script, but I already loaded stuff here...
"""
    
#%%
"""
        Plot some deltas brute force MSE
"""


var = 'MSE'

units = 'W/m^2'
    
for prop in sfc_props:
        
    for sea in ['ANN']: #seas:
        
        mapdata = atm_resp[var][sea][prop][-1,:,:] - atm_resp[var][sea][prop][1,:,:]
        
        ttl_main = '$\Delta$ '+ var + ', ' + prop + ', ' + sea
        filename = 'delta_'+ var + '_' + prop + '_' + sea
        
        #cmap_abs = plt.cm.viridis
        #cmap_diff = plt.cm.viridis
        
        #clim_abs = [np.min(mapdata),np.max(mapdata)]
        
        abs_max = np.max([np.abs(np.min(mapdata)),np.abs(np.max(mapdata))])
        clim_diff = [-abs_max, abs_max]
        
        cmap_diff = plt.cm.RdBu_r
        
        clim_diff = [-10, 10]
        
        fig, axes = plt.subplots(1, 1, figsize=(6,4))
        
        ax = fig.gca()
        
        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,var,'moll',title=ttl_main,clim=clim_diff,colmap=cmap_diff, cb_ttl='units: '+units )
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
"""
        Plot some deltas brute force PRECIP
"""

ms2mmday = 60*60*24*1000
var = 'PRECIP'

units = 'mm/day'
    
for prop in sfc_props:
        
    for sea in ['ANN']: #seas:
        
        mapdata = (atm_resp[var][sea][prop][-1,:,:]) - atm_resp[var][sea][prop][1,:,:]
        
        mapdata = mapdata*ms2mmday
        
        ttl_main = '$\Delta$ '+ var + ', ' + prop + ', ' + sea
        filename = 'delta_'+ var + '_' + prop + '_' + sea
        
        #cmap_abs = plt.cm.viridis
        #cmap_diff = plt.cm.viridis
        
        #clim_abs = [np.min(mapdata),np.max(mapdata)]
        
        abs_max = np.max([np.abs(np.min(mapdata)),np.abs(np.max(mapdata))])
        clim_diff = [-abs_max, abs_max]
        
        cmap_diff = plt.cm.RdBu
        
        clim_diff = [-1,1]
        
        fig, axes = plt.subplots(1, 1, figsize=(6,4))
        
        ax = fig.gca()
        
        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,var,'moll',title=ttl_main,clim=clim_diff,colmap=cmap_diff, cb_ttl='units: '+units )
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