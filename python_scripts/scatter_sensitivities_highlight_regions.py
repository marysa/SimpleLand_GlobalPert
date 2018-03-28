#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:42:01 2017

@author: mlague
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
atm_var= 'TREFHT'
units[atm_var] = ds0[atm_var].units
   
sfc_props = ['alb','rs','hc','log_hc']
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

i=0

for sea in seasons: 
    atm_resp[sea] = {}


for prop in sfc_props:
    pert[prop] = sfc_prop_ranges[i]
    
    if np.isnan(pert[prop][3]):
        print(prop)
        ds1 = ds_low[prop]
        ds2 = ds_med[prop]
        ds3 = ds_high[prop]
        
        # annual mean response
        atm_resp_ann[prop] = [ds1.mean('time')[atm_var].values[:,:],
            ds2.mean('time')[atm_var].values[:,:],
            ds3.mean('time')[atm_var].values[:,:]]
    
        # seasonal responses:
        # (first, make 12 month response, then average over djf, jja, etc)
        #print(np.shape(ds1[atm_var].values))
        resp_mths = np.array([ds1[atm_var].values[:,:,:],
                ds2[atm_var].values[:,:,:],
                ds3[atm_var].values[:,:,:]])
    
    else:
        print(prop)
        ds1 = ds_low1[prop] #0.01
        ds2 = ds_low2[prop]  #0.05
        ds3 = ds_med1[prop]  #0.1
        ds4 = ds_med2[prop]    #0.5
        ds5 = ds_high1[prop]    #1
        ds6 = ds_high2[prop]    #2
        
        # annual mean response
        atm_resp_ann[prop] = [ds1.mean('time')[atm_var].values[:,:],
            ds2.mean('time')[atm_var].values[:,:],
            ds3.mean('time')[atm_var].values[:,:],
            ds4.mean('time')[atm_var].values[:,:],
            ds5.mean('time')[atm_var].values[:,:],
            ds6.mean('time')[atm_var].values[:,:],
            ]
    
        # seasonal responses:
        # (first, make 12 month response, then average over djf, jja, etc)
        #print(np.shape(ds1[atm_var].values))
        resp_mths = np.array([ds1[atm_var].values[:,:,:],
                ds2[atm_var].values[:,:,:],
                ds3[atm_var].values[:,:,:],
                ds4[atm_var].values[:,:,:],
                ds5[atm_var].values[:,:,:],
                ds6[atm_var].values[:,:,:],
                ])
    
    print(np.shape(resp_mths))
    #print(type(resp_mths))
    #print(resp_mths[:,[11,0,1]])
    atm_resp_djf[prop] = np.mean(resp_mths[:,[11,0,1],:,:],1).squeeze()
    atm_resp_mam[prop] = np.mean(resp_mths[:,[2,3,4],:,:],1).squeeze()
    atm_resp_jja[prop] = np.mean(resp_mths[:,[5,6,7],:,:],1).squeeze()
    atm_resp_son[prop] = np.mean(resp_mths[:,[8,9,10],:,:],1).squeeze()
    
    atm_resp['ANN'][prop] = atm_resp_ann[prop]
    atm_resp['DJF'][prop] = atm_resp_djf[prop]
    atm_resp['MAM'][prop] = atm_resp_mam[prop]
    atm_resp['JJA'][prop] = atm_resp_jja[prop]
    atm_resp['SON'][prop] = atm_resp_son[prop]
    
    
    print(np.shape(atm_resp_djf[prop]))
    i=i+1



    
    

#print(prop)
#print(pert)
#print(atm_resp)
#print(pert['alb'])



#%%

if do_slope_analysis==1:
    # sklearn linear_model , pearson r
    
    # Linear regression
    
    # Annual and seasonal... ought to be able to nest, non? 
    slope = {}
    intercept = {}
    r_value = {}
    p_value = {}
    std_err = {}
    
    sensitivity = {}
    
    seasons = ['ANN','DJF','MAM','JJA','SON']
    
    for sea in seasons:
        slope[sea] = {}
        intercept[sea] = {}
        r_value[sea] = {}
        p_value[sea] = {}
        std_err[sea] = {}
        sensitivity[sea] = {}
    
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
            raw_data = np.array(atm_resp[sea][prop])
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
            slope[sea][prop] = slope_vector.reshape(96,144)
            intercept[sea][prop] = intercept_vector.reshape(96,144)
            
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
            
            r_value[sea][prop] = r_v.reshape(96,144)
            p_value[sea][prop] = p_v.reshape(96,144)
    
        
            del raw_data, raw_data_v
            
            


        
#%%
    
    """
        New metrics: 
            1.  scatter plot sensitivity to evaporative resistance against 
                annual mean bucket "fullness" ...
            2.  
    """
    
#%%
"""
        Identify boxes for different regions, e.g. Australia, Sahara, Amazon, etc
"""  

regions = {}
region_names = ['Australia' , 'Amazon', 'Sahara' ,'Siberia']    

for reg in region_names:
    regions[reg] = {}
    

###############################    
reg = 'Australia'

regions[reg]['lat_bot'] = -40
regions[reg]['lat_top'] = -11
regions[reg]['lon_west'] = 113
regions[reg]['lon_east'] = 153

regions[reg]['col'] = 'gold'

###############################    
reg = 'Amazon'

regions[reg]['lat_bot'] = -15
regions[reg]['lat_top'] = 5
regions[reg]['lon_west'] = -70
regions[reg]['lon_east'] = -48

regions[reg]['col'] = 'springgreen'

###############################    
reg = 'Siberia'

regions[reg]['lat_bot'] = 62
regions[reg]['lat_top'] =71
regions[reg]['lon_west'] = 97
regions[reg]['lon_east'] = 176

regions[reg]['col'] = 'forestgreen'


###############################    
reg = 'Sahara'

# problem: having trouble crossing prime meridian. Sunk a lot of time. 
# will work around at present by taking subset of Sahara, east of prime meridian
# for now
regions[reg]['lat_bot'] = 15
regions[reg]['lat_top'] = 31
regions[reg]['lon_west'] = 1
regions[reg]['lon_east'] = 34

regions[reg]['col'] = 'orange'


###############################  

for reg in region_names:
    
    regions[reg]['ind_bot'] = np.where(lat>regions[reg]['lat_bot'])[0][0]
    regions[reg]['ind_top'] = np.where(lat>regions[reg]['lat_top'])[0][0]
    
    if regions[reg]['lon_west'] > 0:
        regions[reg]['ind_west'] = np.where(lon> ( regions[reg]['lon_west']) )[0][0]
    else:
        regions[reg]['ind_west'] =  np.where(lon> ( 360 + regions[reg]['lon_west']) )[0][0]
    
    if regions[reg]['lon_east'] > 0:
        regions[reg]['ind_east'] = np.where(lon> ( regions[reg]['lon_east']) )[0][0]
    else:
        regions[reg]['ind_east'] =  np.where(lon> (360 + regions[reg]['lon_east']) )[0][0]

    # Swap east and west if west is bigger than east. 
#    if regions[reg]['ind_east'] < regions[reg]['ind_west']:
#        old_east = regions[reg]['ind_east']
#        old_west = regions[reg]['ind_west']
#        regions[reg]['ind_west'] = old_east
#        regions[reg]['ind_east'] = old_west
        
    # if left longitude index is negative (western hemisphere) AND right lon
    # index is positive (eastern hemisphere), set 
# for places in the western hemisphere, need to convert negative lons to positive in the 0-360 sense.
# 1:180 is East
# 181 is -179
# -1 is 359


#%% Plot boundaries of each region ... or pcolor in gridboxes that fall in there...
        
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
mp.drawcoastlines()
mp.drawmapboundary(fill_color='1.')  # make map background white
parallels = np.arange(-90.,90,20.)
mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
meridians = np.arange(0.,360.,20.)
mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])


for reg in region_names :
    
    col = regions[reg]['col']
    
    x1,y1 = mp(lon[regions[reg]['ind_west']],lat[regions[reg]['ind_bot']])
    x2,y2 = mp(lon[regions[reg]['ind_east']],lat[regions[reg]['ind_bot']])
    x3,y3 = mp(lon[regions[reg]['ind_east']],lat[regions[reg]['ind_top']])
    x4,y4 = mp(lon[regions[reg]['ind_west']],lat[regions[reg]['ind_top']])
   
    
    #if reg == 'Sahara':
    #    x1,y1 = mp(lon[regions[reg]['ind_east']],lat[regions[reg]['ind_bot']])
    #    x2,y2 = mp(lon[regions[reg]['ind_west']],lat[regions[reg]['ind_bot']])
    #    x3,y3 = mp(lon[regions[reg]['ind_west']],lat[regions[reg]['ind_top']])
    #    x4,y4 = mp(lon[regions[reg]['ind_east']],lat[regions[reg]['ind_top']])
   
        
    poly = plt.Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor=None,edgecolor=col,linewidth=3,fill=False)
    plt.gca().add_patch(poly)
    
    #mp.plot(x,y,'D-', markersize=8, linewidth=4, color='k', markerfacecolor='m')



ttl = 'asewefawefa'
print(ttl)
plt.title(ttl,fontsize=12)




# annotate with date/time
ax = plt.gca()
ax.text(-0.05,-0.05, time.strftime("%x")+'\n' ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)


plt.show()
filename = 'locations_for_scatter'
fig_png = figpath+'/sensitivity/scatter_plots/'+filename+'.png'

fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)

plt.close()
    

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

ctrl_water = ds0['MML_water'].values
ann_water = np.nanmean(ctrl_water,0) * no_glc
xvar = ann_water.flatten()

xgrid = ctrl_water

units['water'] = ds0['MML_water'].units


for prop in ['rs']:
    
    for sea in ['ANN']:
    
        #datm_dlnd = np.array(slope[sea][prop]).flatten() * 10
        ygrid = np.array(slope[sea][prop]) * 10
        
        datm_dlnd = ygrid.flatten()
        
        yvar = datm_dlnd
        units['datm_dlnd'] = '[K] / [10 s/m]'
    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        p1 = ax.scatter(xvar,yvar,s=10,color='cornflowerblue',alpha=0.1)
        
        ax.set_xlim([0,200])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean Water in Bucket ('+units['water']+')')
        ax.set_ylabel('Sensitivity d(T2m)/d(10 r_s) ('+units['datm_dlnd']+')')
        plt.title('Sensitivity to rs vs bucket fullness')
    
        
        # add on scatter of region locations
        for reg in region_names:
            
            ind_bot = regions[reg]['ind_bot']
            ind_top = regions[reg]['ind_top']
            ind_left = regions[reg]['ind_west']
            ind_right = regions[reg]['ind_east']
            
            col = regions[reg]['col']
            
            
            reg_xgrid = ( np.nanmean(xgrid,0)[ind_bot:ind_top,ind_left:ind_right] )# *
                             #no_glc[ind_bot:ind_top,ind_left:ind_right] )
            reg_xvar = reg_xgrid.flatten()
        
            reg_ygrid = (ygrid[ind_bot:ind_top,ind_left:ind_right] )#*
                             #no_glc[ind_bot:ind_top,ind_left:ind_right] )
            reg_yvar = reg_ygrid.flatten()

            plt.scatter(x=reg_xvar,y=reg_yvar,s=20,alpha=0.9, edgecolors=col,facecolors='none' , label = reg)  #,cmap = cm_scat)
        
        
        # get handles and plot legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,labels,bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
        
        plt.show()   
        
        filename = 'sensitivity_rs_vs_bucket_fullness_locations'
        fig_png = figpath+'/sensitivity/scatter_plots/'+filename+'.png'

        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)

        plt.close()
    

        #%%
"""
        Sensitivity to evaporative resistane against precip (should look like fullness)
            yvar: dT2m / d rs
            xvar: annual mean precip ( in baseline a2 hc0.1 rs100 run )
"""
    
units = {}

ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ms2mmyr = 1000*60*60*24*365


ctrl_precip = (  ds0['PRECC'].values + ds0['PRECL'].values + 
                  ds0['PRECSC'].values + ds0['PRECSL'].values ) 
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
xgrid = ann_precip
xvar = ann_precip.flatten()



units['precip'] = 'mm/year' #ds0['PRECC'].units


for prop in ['rs']:
    
    for sea in ['ANN']:
    
        ygrid = np.array(slope[sea][prop]) * 10
        datm_dlnd = ygrid.flatten()
        
        yvar = datm_dlnd
        units['datm_dlnd'] = '[K] / [10 s/m]'
    
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
        
        # plot scatter:
        p1 = ax.scatter(xvar,yvar,s=10,color='blue',alpha=0.1)
        
        ax.set_xlim([np.nanmin(xvar),np.nanmax(xvar)])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean Precip ('+units['precip']+')')
        ax.set_ylabel('Sensitivity d(T2m)/d(r_s) ('+units['datm_dlnd']+')')
        plt.title('Sensitivity to rs vs precip')
    
        
        
        # add on scatter of region locations
        for reg in region_names:
            
            ind_bot = regions[reg]['ind_bot']
            ind_top = regions[reg]['ind_top']
            ind_left = regions[reg]['ind_west']
            ind_right = regions[reg]['ind_east']
            
            col = regions[reg]['col']
            
            
            reg_xgrid = ( xgrid[ind_bot:ind_top,ind_left:ind_right] )# *
                             #no_glc[ind_bot:ind_top,ind_left:ind_right] )
            reg_xvar = reg_xgrid.flatten()
        
            reg_ygrid = (ygrid[ind_bot:ind_top,ind_left:ind_right] )#*
                             #no_glc[ind_bot:ind_top,ind_left:ind_right] )
            reg_yvar = reg_ygrid.flatten()

            plt.scatter(x=reg_xvar,y=reg_yvar,s=20,alpha=0.9, edgecolors=col,facecolors='none' , label = reg)  #,cmap = cm_scat)
        
        
        # get handles and plot legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,labels,bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
        
        plt.show()   
        
        filename = 'sensitivity_rs_vs_precip_locations'
        fig_png = figpath+'/sensitivity/scatter_plots/'+filename+'.png'

        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)

        plt.close()
        
#%%
"""
        Sensitivity to evaporative resistane against bucket fullness
            yvar: annual mean precip
            xvar: annual mean water ( in baseline a2 hc0.1 rs100 run )
            color: dT2m / d rs
"""
    
units = {}

dsl0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']

ctrl_water = dsl0['MML_water'].values
ann_water = np.nanmean(ctrl_water,0) * no_glc
xgrid = ann_water
xvar = xgrid.flatten()

units['water'] = dsl0['MML_water'].units

ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ctrl_precip = (  ds0['PRECC'].values + ds0['PRECL'].values + 
                  ds0['PRECSC'].values + ds0['PRECSL'].values )
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
ygrid = ann_precip
yvar = ygrid.flatten()

#yvar = yvar*ms2mmyr

units['precip'] = 'mm/yr'    #ds0['PRECC'].units


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        zgrid = np.array(slope[sea][prop]) * 10
        datm_dlnd = zgrid.flatten()
        
        cvar = datm_dlnd
        units['datm_dlnd'] = '[K] / [10 prop units]'
        
    
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        
        cm_scat = plt.cm.get_cmap('viridis')
        cm_scat = plt.cm.get_cmap('RdYlBu')
        # plot scatter:
        scat = plt.scatter(x=xvar,y=yvar,c=cvar,alpha=0.3, cmap=plt.cm.viridis_r ,edgecolor = 'None')  #,cmap = cm_scat)
        
        cb = plt.colorbar(scat)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+prop+') , '+units['datm_dlnd'])
    
    
        ax.set_xlim([0,200])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean Water in Bucket ('+units['water']+')')
        ax.set_ylabel('Annual Mean PRECIP ('+units['precip']+')')
        plt.title('Sensitivity ('+prop+') dependence on PRECIP and Ground Water')
        
#        # add on scatter of region locations
#        for reg in region_names:
#            
#            ind_bot = regions[reg]['ind_bot']
#            ind_top = regions[reg]['ind_top']
#            ind_left = regions[reg]['ind_west']
#            ind_right = regions[reg]['ind_east']
#            
#            col = regions[reg]['col']
#            
#            
#            reg_xgrid = ( xgrid[ind_bot:ind_top,ind_left:ind_right] )# *
#                             #no_glc[ind_bot:ind_top,ind_left:ind_right] )
#            reg_xvar = reg_xgrid.flatten()
#        
#            reg_ygrid = (ygrid[ind_bot:ind_top,ind_left:ind_right] )#*
#                             #no_glc[ind_bot:ind_top,ind_left:ind_right] )
#            reg_yvar = reg_ygrid.flatten()
#
#            plt.scatter(x=reg_xvar,y=reg_yvar,s=20,alpha=0.9, edgecolors=col,facecolors='none' , label = reg)  #,cmap = cm_scat)
        
        
        # get handles and plot legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,labels,bbox_to_anchor=(1.35,1),loc=2,borderaxespad=0.)
        
        plt.show()   
        
        filename = 'sensitivity_'+prop+'_precip_water'
        fig_png = figpath+'/sensitivity/scatter_plots/'+filename+'.png'

        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)

        plt.close()
    
        
#%%
"""
        Sensitivity to evaporative resistane against bucket fullness
            yvar: annual mean precip
            xvar: annual mean temperatures ( in baseline a2 hc0.1 rs100 run )
            color: 0.1 dT2m / d  sfc props
"""
    
units = {}

dsl0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']

ctrl_x = ds0['FSNS'].values
ann_x = np.nanmean(ctrl_x,0) * no_glc
xgrid = ann_x
xvar = xgrid.flatten()

units['FSNS'] = ds0['FSNS'].units

ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ctrl_precip = (  ds0['PRECC'].values + ds0['PRECL'].values + 
                  ds0['PRECSC'].values + ds0['PRECSL'].values )
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
ygrid = ann_precip
yvar = ygrid.flatten()

#yvar = yvar*ms2mmyr

units['precip'] = 'mm/yr'    #ds0['PRECC'].units


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        zgrid = np.array(slope[sea][prop]) * 10
        datm_dlnd = zgrid.flatten()
        
        cvar = datm_dlnd
        units['datm_dlnd'] = '[K] / [10 prop units]'
        
    
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        
        cm_scat = plt.cm.get_cmap('viridis')
        cm_scat = plt.cm.get_cmap('RdYlBu')
        # plot scatter:
        scat = plt.scatter(x=xvar,y=yvar,c=cvar,alpha=0.3, cmap=plt.cm.viridis_r ,edgecolor = 'None')  #,cmap = cm_scat)
        
        cb = plt.colorbar(scat)
    
        #cb = plt.colorbar(scat)
        cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+prop+') , '+units['datm_dlnd'])
    
    
        ax.set_xlim([0,200])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('Annual Mean Absorbed Solar Radiation ('+units['FSNS']+')')
        ax.set_ylabel('Annual Mean PRECIP ('+units['precip']+')')
        plt.title('Sensitivity ('+prop+') dependence on PRECIP and FSNS')
        
#        # add on scatter of region locations
#        for reg in region_names:
#            
#            ind_bot = regions[reg]['ind_bot']
#            ind_top = regions[reg]['ind_top']
#            ind_left = regions[reg]['ind_west']
#            ind_right = regions[reg]['ind_east']
#            
#            col = regions[reg]['col']
#            
#            
#            reg_xgrid = ( xgrid[ind_bot:ind_top,ind_left:ind_right] )# *
#                             #no_glc[ind_bot:ind_top,ind_left:ind_right] )
#            reg_xvar = reg_xgrid.flatten()
#        
#            reg_ygrid = (ygrid[ind_bot:ind_top,ind_left:ind_right] )#*
#                             #no_glc[ind_bot:ind_top,ind_left:ind_right] )
#            reg_yvar = reg_ygrid.flatten()
#
#            plt.scatter(x=reg_xvar,y=reg_yvar,s=20,alpha=0.9, edgecolors=col,facecolors='none' , label = reg)  #,cmap = cm_scat)
        
        
        # get handles and plot legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,labels,bbox_to_anchor=(1.35,1),loc=2,borderaxespad=0.)
        
        plt.show()   
        
        filename = 'sensitivity_'+prop+'_precip_fsns'
        fig_png = figpath+'/sensitivity/scatter_plots/'+filename+'.png'

        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)

        plt.close()
    
                
        
#%%  
        
        
#%%   
"""
    Spatial scatter (esp for roughness) of sensitivity dT2m/dhc vs r^2 -> where / how many points just don't fit? 
"""

#%%
"""
        Sensitivity to evaporative resistane against bucket fullness
            yvar: annual mean precip
            xvar: annual mean water ( in baseline a2 hc0.1 rs100 run )
            color: dT2m / d rs
"""
    
units = {}

dsl0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']

ctrl_water = dsl0['MML_water'].values
ann_water = np.nanmean(ctrl_water,0) * no_glc
xgrid = ann_water
xvar = xgrid.flatten()

units['water'] = dsl0['MML_water'].units

ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']

ctrl_precip = (  ds0['PRECC'].values + ds0['PRECL'].values + 
                  ds0['PRECSC'].values + ds0['PRECSL'].values )
ann_precip = np.nanmean(ctrl_precip,0) * no_glc * ms2mmyr
ygrid = ann_precip
yvar = ygrid.flatten()

#yvar = yvar*ms2mmyr

units['precip'] = 'mm/yr'    #ds0['PRECC'].units


for prop in sfc_props:
    
    for sea in ['ANN']:
    
        zgrid = np.array(slope[sea][prop]) * 10
        datm_dlnd = zgrid.flatten()
        
        cvar = datm_dlnd
        units['datm_dlnd'] = '[K] / [10 prop units]'
        
        ygrid = np.array(slope[sea][prop]) * no_glc
        yvar = ygrid.flatten()
        
        xgrid = np.array(r_value[sea][prop]) * no_glc
        xvar = xgrid.flatten()
        
    
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        
        cm_scat = plt.cm.get_cmap('viridis')
        cm_scat = plt.cm.get_cmap('RdYlBu')
        # plot scatter:
        scat = plt.scatter(x=xvar,y=yvar,alpha=0.3,edgecolor = 'None')  #,cmap = cm_scat)
        
        #cb = plt.colorbar(scat)
    
        #cb = plt.colorbar(scat)
        #cb.set_label('Atmospheric Sensitivity d (T2m) / d ('+prop+') , '+units['datm_dlnd'])
    
    
        #ax.set_xlim([0,200])
        #ax.set_ylim([-0.005,])
        
        ax.set_xlabel('r^2 (linearity of fit)')
        ax.set_ylabel('Sensitivity dT2m/dlnd for ' + prop )
        plt.title('Sensitivity ('+prop+') vs linearity')
        
#        # add on scatter of region locations
#        for reg in region_names:
#            
#            ind_bot = regions[reg]['ind_bot']
#            ind_top = regions[reg]['ind_top']
#            ind_left = regions[reg]['ind_west']
#            ind_right = regions[reg]['ind_east']
#            
#            col = regions[reg]['col']
#            
#            
#            reg_xgrid = ( xgrid[ind_bot:ind_top,ind_left:ind_right] )# *
#                             #no_glc[ind_bot:ind_top,ind_left:ind_right] )
#            reg_xvar = reg_xgrid.flatten()
#        
#            reg_ygrid = (ygrid[ind_bot:ind_top,ind_left:ind_right] )#*
#                             #no_glc[ind_bot:ind_top,ind_left:ind_right] )
#            reg_yvar = reg_ygrid.flatten()
#
#            plt.scatter(x=reg_xvar,y=reg_yvar,s=20,alpha=0.9, edgecolors=col,facecolors='none' , label = reg)  #,cmap = cm_scat)
        
        
        # get handles and plot legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,labels,bbox_to_anchor=(1.35,1),loc=2,borderaxespad=0.)
        
        plt.show()   
        
        filename = 'sensitivity_'+prop+'_vs_r2'
        fig_png = figpath+'/sensitivity/scatter_plots/'+filename+'.png'

        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)

        plt.close()
    


#%%      
        
        
#%%       
        
        
#%%
"""
        Sensitivity to evaporative resistane against bucket fullness
            yvar: annual mean precip
            xvar: annual mean water ( in baseline a2 hc0.1 rs100 run )
            color: dT2m / d rs
"""
    
#units = {}
#
#dsl0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
#
#ctrl_water = dsl0['MML_water'].values
#ann_water = np.nanmean(ctrl_water,0) * no_glc
#xvar = ann_water.flatten()
#
#units['water'] = dsl0['MML_water'].units
#
#ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
#
#ctrl_trefht = (  ds0['TREFHT'].values  )
#ann_trefht = np.nanmean(ctrl_trefht,0) * no_glc
#yvar = ann_trefht.flatten()
#
#yvar = yvar
#
#units['trefht'] = ds0['TREFHT'].units
#
#
#for prop in ['rs']:
#    
#    for sea in ['ANN']:
#    
#        datm_dlnd = np.array(slope[sea][prop]).flatten()
#        
#        cvar = datm_dlnd*10
#        units['datm_dlnd'] = '[K] / [10 s/m]'
#        
#    
#        fig, ax = plt.subplots(1, 1, figsize=(6,6))
#        
#        cm_scat = plt.cm.get_cmap('viridis')
#        cm_scat = plt.cm.get_cmap('RdYlBu')
#        # plot scatter:
#        scat = plt.scatter(x=xvar,y=yvar,c=cvar,alpha=0.3, cmap=plt.cm.viridis_r )  #,cmap = cm_scat)
#        
#        cb = plt.colorbar(scat)
#    
#        #cb = plt.colorbar(scat)
#        cb.set_label('Atmospheric Sensitivity d (T2m) / d (rs) , '+units['datm_dlnd'])
#    
#    
#        ax.set_xlim([0,200])
#        #ax.set_ylim([-0.005,])
#        
#        ax.set_xlabel('Annual Mean Water in Bucket ('+units['water']+')')
#        ax.set_ylabel('Annual Mean T2m ('+units['trefht']+')')
#        plt.title('Yo')
#        
#        # Add no-fill circles with red outline on top of existing scatter
#        """
#            Australia box:
#                        ind_left = 48
#                        ind_right = 59
#                        ind_top = 37
#                        ind_bot = 31
#        """
#        
#        ind_left = 48
#        ind_right = 59
#        ind_top = 37
#        ind_bot = 31
#                        
#                        
#        AUS_ann_water = ( np.nanmean(ctrl_water,0)[ind_bot:ind_top,ind_left:ind_right] *
#                             no_glc[ind_bot:ind_top,ind_left:ind_right] )
#        AUS_xvar = AUS_ann_water.flatten()
#        
#        AUS_ann_fsnsc = (np.nanmean(ctrl_trefht,0)[ind_bot:ind_top,ind_left:ind_right] *
#                             no_glc[ind_bot:ind_top,ind_left:ind_right] )
#        AUS_yvar = AUS_ann_fsnsc.flatten()
#        
#        AUS_datm_dlnd = np.array(slope[sea][prop][ind_bot:ind_top,ind_left:ind_right]).flatten()
#        
#        AUS_cvar = AUS_datm_dlnd*10
#        
#        aus = plt.scatter(x=AUS_xvar,y=AUS_yvar,s=20,alpha=0.9, edgecolors='r',facecolors='none' )  #,cmap = cm_scat)
#        
#        
#        plt.show()
#    
##%%
#"""
#        Sensitivity to evaporative resistane against bucket fullness
#            yvar: annual mean precip
#            xvar: annual mean water ( in baseline a2 hc0.1 rs100 run )
#            color: dT2m / d rs
#"""
#    
#units = {}
#
#dsl0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
#
#ctrl_water = dsl0['MML_water'].values
#ann_water = np.nanmean(ctrl_water,0) * no_glc
#xvar = ann_water.flatten()
#
#units['water'] = dsl0['MML_water'].units
#
#ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
#
#ctrl_trefht = (  ds0['TREFHT'].values  )
#ann_trefht = np.nanmean(ctrl_trefht,0) * no_glc
#yvar = ann_trefht.flatten()
#
#yvar = yvar
#
#units['FSNS'] = ds0['FSNS'].units
#
#
#for prop in ['rs']:
#    
#    for sea in ['ANN']:
#    
#        datm_dlnd = np.array(slope[sea][prop]).flatten()
#        
#        cvar = datm_dlnd*10
#        units['datm_dlnd'] = '[K] / [10 s/m]'
#        
#    
#        fig, ax = plt.subplots(1, 1, figsize=(6,6))
#        
#        cm_scat = plt.cm.get_cmap('viridis')
#        cm_scat = plt.cm.get_cmap('RdYlBu')
#        # plot scatter:
#        scat = plt.scatter(x=xvar,y=yvar,s=20,c=cvar,alpha=0.3, cmap=plt.cm.viridis_r )  #,cmap = cm_scat)
#        
#        cb = plt.colorbar(scat)
#    
#        #cb = plt.colorbar(scat)
#        cb.set_label('Atmospheric Sensitivity d (T2m) / d (rs) , '+units['datm_dlnd'])
#    
#    
#        ax.set_xlim([0,200])
#        #ax.set_ylim([-0.005,])
#        
#        ax.set_xlabel('Annual Mean Water in Bucket ('+units['water']+')')
#        ax.set_ylabel('Annual Mean FSNS ('+units['FSNS']+')')
#        plt.title('Yo')
#        
#        
#        # Add no-fill circles with red outline on top of existing scatter
#        """
#            Australia box:
#                        ind_left = 48
#                        ind_right = 59
#                        ind_top = 37
#                        ind_bot = 31
#        """
#        
#        ind_left = 48
#        ind_right = 59
#        ind_top = 37
#        ind_bot = 31
#                        
#                        
#        AUS_ann_water = ( np.nanmean(ctrl_water,0)[ind_bot:ind_top,ind_left:ind_right] *
#                             no_glc[ind_bot:ind_top,ind_left:ind_right] )
#        AUS_xvar = AUS_ann_water.flatten()
#        
#        AUS_ann_fsnsc = (np.nanmean(ctrl_trefht,0)[ind_bot:ind_top,ind_left:ind_right] *
#                             no_glc[ind_bot:ind_top,ind_left:ind_right] )
#        AUS_yvar = AUS_ann_fsnsc.flatten()
#        
#        AUS_datm_dlnd = np.array(slope[sea][prop][ind_bot:ind_top,ind_left:ind_right]).flatten()
#        
#        AUS_cvar = AUS_datm_dlnd*10
#        
#        aus = plt.scatter(x=AUS_xvar,y=AUS_yvar,s=20,alpha=0.9, edgecolors='r',facecolors='none' )  #,cmap = cm_scat)
#        
#        
#        plt.show()        
#                
#        
#        
#        
#        
#        
#        
#        #%%
#"""
#        Sensitivity to evaporative resistane against precip (should look like fullness)
#            yvar: dT2m / d rs
#            xvar: annual mean precip ( in baseline a2 hc0.1 rs100 run )
#"""
#    
#units = {}
#
#ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
#
#ctrl_precip = (  ds0['PRECC'].values + ds0['PRECL'].values + 
#                  ds0['PRECSC'].values + ds0['PRECSL'].values )
#ann_precip = np.nanmean(ctrl_precip,0)
#xvar = ann_precip.flatten()
#
#ms2mmyr = 1000*60*60*24*365
#
#xvar = ms2mmyr*xvar
#
#units['precip'] = 'mm/year' #ds0['PRECC'].units
#
#
#for prop in ['rs']:
#    
#    for sea in ['ANN']:
#    
#        datm_dlnd = np.array(slope[sea][prop]).flatten()
#        
#        yvar = datm_dlnd*10
#        units['datm_dlnd'] = '[K] / [10 s/m]'
#    
#        fig, ax = plt.subplots(1, 1, figsize=(4,4))
#        
#        # plot scatter:
#        p1 = ax.scatter(xvar,yvar,s=10,color='blue',alpha=0.1)
#        
#        ax.set_xlim([min(xvar),max(xvar)])
#        #ax.set_ylim([-0.005,])
#        
#        ax.set_xlabel('Annual Mean Precip ('+units['precip']+')')
#        ax.set_ylabel('Sensitivity d(T2m)/d(r_s) ('+units['datm_dlnd']+')')
#        plt.title('Yo')
#    
#        plt.show()
#        
##%%
#"""
#        Sensitivity to evaporative resistane against bucket fullness
#            yvar: annual mean precip
#            xvar: annual mean water ( in baseline a2 hc0.1 rs100 run )
#            color: dT2m / d rs
#"""
#    
#units = {}
#
#dsl0 = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
#
#ctrl_water = dsl0['MML_water'].values
#ann_water = np.nanmean(ctrl_water,0)
#xvar = ann_water.flatten()
#
#units['water'] = dsl0['MML_water'].units
#
#ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
#
#ctrl_precip = (  ds0['PRECC'].values + ds0['PRECL'].values + 
#                  ds0['PRECSC'].values + ds0['PRECSL'].values )
#ann_precip = np.nanmean(ctrl_precip,0)
#yvar = ann_precip.flatten()
#
#yvar = yvar*ms2mmyr
#
#units['precip'] = 'mm/yr'    #ds0['PRECC'].units
#
#
#for prop in sfc_props:
#    
#    for sea in ['ANN']:
#    
#        datm_dlnd = np.array(slope[sea][prop]).flatten()
#        
#        cvar = datm_dlnd*10
#        units['datm_dlnd'] = '[K] / [10 s/m]'
#        
#    
#        fig, ax = plt.subplots(1, 1, figsize=(6,6))
#        
#        cm_scat = plt.cm.get_cmap('viridis')
#        cm_scat = plt.cm.get_cmap('RdYlBu')
#        # plot scatter:
#        scat = plt.scatter(x=xvar,y=yvar,c=cvar,alpha=0.3, cmap=plt.cm.viridis_r )  #,cmap = cm_scat)
#        
#        cb = plt.colorbar(scat)
#    
#        #cb = plt.colorbar(scat)
#        cb.set_label('Atmospheric Sensitivity d (T2m) / d (rs) , '+units['datm_dlnd'])
#    
#    
#        ax.set_xlim([0,200])
#        #ax.set_ylim([-0.005,])
#        
#        ax.set_xlabel('Annual Mean Water in Bucket ('+units['water']+')')
#        ax.set_ylabel('Annual Mean PRECIP ('+units['precip']+')')
#        plt.title('Yo')
#        
#        fig_png = figpath+'/sensitivity/scatter_plots/'+filename+'.png'
#        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
#                        edgecolor='w',orientation='portrait', 
#                        frameon=None)
#
#        
#        
#        plt.show()
#    
    
        
#%%
    
    
        
#%%
    
    
    