#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:43:35 2017

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

from NE_flux import NE_flux

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

#print(np.shape(LN))
#print(np.shape(landmask))


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

# Energy responses
energy = {}


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

energy_vars = ['FSNT','FLNT','FSNTC','FLNTC']

i=0

for sea in seasons: 
    atm_resp[sea] = {}
    energy[sea] = {}


for prop in sfc_props:
    pert[prop] = sfc_prop_ranges[i]
    
    energy['ANN'][prop] = {}
    
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
        
        for sea in ['ANN']:
            for evar in energy_vars:
                energy[sea][prop][evar] = np.array([np.array(ds1.mean('time')[evar].values[:,:]),
                                           np.array(ds2.mean('time')[evar].values[:,:]),
                                           np.array(ds3.mean('time')[evar].values[:,:])])
            energy[sea][prop]['imbal'] = energy[sea][prop]['FSNT'] - energy[sea][prop]['FLNT']
            energy[sea][prop]['imbalC'] = energy[sea][prop]['FSNTC'] - energy[sea][prop]['FLNTC']
            
            
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
    
        
        for sea in ['ANN']:
            for evar in energy_vars:
                energy[sea][prop][evar] = np.array([np.array(ds1.mean('time')[evar].values[:,:]),
                                       np.array(ds2.mean('time')[evar].values[:,:]),
                                       np.array(ds3.mean('time')[evar].values[:,:]),
                                       np.array(ds4.mean('time')[evar].values[:,:]),
                                       np.array(ds5.mean('time')[evar].values[:,:]),
                                       np.array(ds6.mean('time')[evar].values[:,:]),
                                       ])
            energy[sea][prop]['imbal'] = energy[sea][prop]['FSNT'] - energy[sea][prop]['FLNT']
            energy[sea][prop]['imbalC'] = energy[sea][prop]['FSNTC'] - energy[sea][prop]['FLNTC']
            
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

#%% Plot differences in contours of northward energy flux (both on zonal mean line plots and conoutred lon maps)

flux = {}
flux_zonal = {}

i = 0

for prop in sfc_props:
    flux[prop] = np.zeros(np.shape(energy['ANN'][prop]['imbal']))
    flux_zonal[prop] = np.zeros(np.shape(energy['ANN'][prop]['imbal'][:,:,0]))
    
    # loop over however many model runs we have
    pert[prop] = sfc_prop_ranges[i]
    

    
    if np.isnan(pert[prop][3]):
        n = 3  
    
    else:
        n = 5
        
    for j in range(n):
            
            FSNT = energy['ANN'][prop]['FSNT'][j,:,:]
            FLNT = energy['ANN'][prop]['FLNT'][j,:,:]
            
            flx, flx_z = NE_flux(lat,lon,FSNT,FLNT,area_grid=area_f19,zonal_mean=True,stats=False)
            
            flux[prop][j,:,:] = flx
            flux_zonal[prop][j,:] = flx_z

    
    
    i = i+1 


#%% Take slope of TOA energy imbalance, call that flux_ptl for now...


#%% Plot differences in northward E transport between extremes of each property
    # as a line plot

fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0.1,0.1,0.9,0.9])

# alb
prop = 'alb'
d_flux_zonal = flux_zonal[prop][-1,:] - flux_zonal[prop][0,:]

plt.plot(lat,d_flux_zonal,'blue',label=prop)
        
        
# hc
prop = 'hc'
# some kind of issue here where entry 5 is zeros???? 
d_flux_zonal = flux_zonal[prop][4,:] - flux_zonal[prop][0,:]

plt.plot(lat,d_flux_zonal,'forestgreen',label=prop)
        

# rs
prop = 'rs'
d_flux_zonal = flux_zonal[prop][-1,:] - flux_zonal[prop][0,:]

plt.plot(lat,d_flux_zonal,'orange',label=prop)
        
          
    
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)    

ax.set_xlim([-90,90])
ax.set_xticks([-90,-60,-30,0,30,60,90])
xlim = ax.get_xlim()
xline = [xlim[0], xlim[1]]
ylim = [0,0]
plt.plot([xlim[0],xlim[1]], [0,0] ,linestyle='dashed',color='gray')

plt.title('$\Delta$ Northward Energy Transport (PW) \n between biggest and smallest ' + prop)    
    
 
 
#%% Plot differences in northward E transport between extremes of each property
    # as a contour map
    
###########################################
# Albedo contours of NE Transport
    
prop = 'alb'
    
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
mp.drawcoastlines()
mp.drawmapboundary(fill_color='1.')  # make map background white
parallels = np.arange(-90.,90,20.)
mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
meridians = np.arange(0.,360.,20.)
mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
cmap_abs = plt.cm.viridis
clim_abs = [0.5,1]    
    
mapdata = flux[prop][-1,:,:] - flux[prop][0,:,:]

grad = np.gradient(mapdata)

abs_grad = np.sqrt(grad[0]**2 + grad[1]**2)

#mapdata = np.random(96,144])

x,y = np.meshgrid(lon,lat)

contour_levs = np.linspace(-0.015,0.015,15)

cs = mp.contour(x,y,mapdata,cmap=plt.cm.inferno,latlon=True,levels = contour_levs)
#cs = mp.pcolor(x,y,abs_grad,cmap=plt.cm.inferno,latlon=True)
    

###########################################
# Albedo gradient of delta NE Transport
    
prop = 'alb'
    
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])

mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
mp.drawcoastlines()
mp.drawmapboundary(fill_color='1.')  # make map background white
parallels = np.arange(-90.,90,20.)
mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
meridians = np.arange(0.,360.,20.)
mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
cmap_abs = plt.cm.viridis
clim_abs = [0.5,1]    
    
mapdata = flux[prop][-1,:,:] - flux[prop][0,:,:]

grad = np.gradient(mapdata)

abs_grad = np.sqrt(grad[0]**2 + grad[1]**2)

#mapdata = np.random(96,144])

x,y = np.meshgrid(lon,lat)

contour_levs = np.linspace(-0.015,0.015,15)

#cs = mp.contour(x,y,abs_grad,cmap=plt.cm.inferno,latlon=True,levels = contour_levs)
grad_mp = mp.pcolor(x,y,abs_grad,cmap=plt.cm.inferno,latlon=True)  
 



 
###########################################
# Albedo contours of NE Transport
    
prop = 'alb'
    
fig, axes = plt.subplots(1, 2, figsize=(10,6))

ax0 = axes.flatten()[0]
plt.sca(ax0)

mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
mp.drawcoastlines()
mp.drawmapboundary(fill_color='1.')  # make map background white
parallels = np.arange(-90.,90,20.)
mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
meridians = np.arange(0.,360.,20.)
mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
cmap_abs = plt.cm.viridis
clim_abs = [0.5,1]    
    
mapdata = flux[prop][0,:,:]

grad = np.gradient(mapdata)

abs_grad = np.sqrt(grad[0]**2 + grad[1]**2)

#mapdata = np.random(96,144])

x,y = np.meshgrid(lon,lat)

contour_levs = np.linspace(-0.06,0.06,15)

cs = mp.contour(x,y,mapdata,cmap=plt.cm.inferno,latlon=True,levels = contour_levs)
#cs = mp.pcolor(x,y,abs_grad,cmap=plt.cm.inferno,latlon=True)

plt.title('alb, lowest; NE Transport')
    


ax1 = axes.flatten()[1]
plt.sca(ax1)

mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
mp.drawcoastlines()
mp.drawmapboundary(fill_color='1.')  # make map background white
parallels = np.arange(-90.,90,20.)
mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
meridians = np.arange(0.,360.,20.)
mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
cmap_abs = plt.cm.viridis
clim_abs = [0.5,1]    
    
mapdata = flux[prop][-1,:,:]

grad = np.gradient(mapdata)

abs_grad = np.sqrt(grad[0]**2 + grad[1]**2)

#mapdata = np.random(96,144])

x,y = np.meshgrid(lon,lat)

contour_levs = np.linspace(-0.06,0.06,15)

cs = mp.contour(x,y,mapdata,cmap=plt.cm.inferno,latlon=True,levels = contour_levs)
#cs = mp.pcolor(x,y,abs_grad,cmap=plt.cm.inferno,latlon=True)

plt.title('alb, highest; NE Transport')
  

#%% Try just plotting toa energy imbalance with contours? But that shouldn't have a
# a *minimum* at the equator... 


###########################################
# Albedo contours of NE Transport
    
prop = 'alb'
    
fig, axes = plt.subplots(1, 2, figsize=(10,6))

ax0 = axes.flatten()[0]
plt.sca(ax0)

mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
mp.drawcoastlines()
mp.drawmapboundary(fill_color='1.')  # make map background white
parallels = np.arange(-90.,90,20.)
mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
meridians = np.arange(0.,360.,20.)
mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
cmap_abs = plt.cm.viridis
clim_abs = [0.5,1]    
    
mapdata = energy['ANN'][prop]['imbal'][0,:,:]

grad = np.gradient(mapdata)

abs_grad = np.sqrt(grad[0]**2 + grad[1]**2)

#mapdata = np.random(96,144])

x,y = np.meshgrid(lon,lat)

contour_levs = np.linspace(-125,125,15)

cs = mp.contour(x,y,mapdata,cmap=plt.cm.RdBu_r,latlon=True,levels = contour_levs)
#cs = mp.pcolor(x,y,abs_grad,cmap=plt.cm.inferno,latlon=True)

plt.title('alb, lowest; toa imbal')
    


ax1 = axes.flatten()[1]
plt.sca(ax1)

mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
mp.drawcoastlines()
mp.drawmapboundary(fill_color='1.')  # make map background white
parallels = np.arange(-90.,90,20.)
mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
meridians = np.arange(0.,360.,20.)
mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
cmap_abs = plt.cm.viridis
clim_abs = [0.5,1]    
    
mapdata = energy['ANN'][prop]['imbal'][2,:,:]

grad = np.gradient(mapdata)

abs_grad = np.sqrt(grad[0]**2 + grad[1]**2)

#mapdata = np.random(96,144])

x,y = np.meshgrid(lon,lat)

#contour_levs = np.linspace(-0.06,0.06,15)

cs = mp.contour(x,y,mapdata,cmap=plt.cm.RdBu_r,latlon=True,levels = contour_levs)
cb = plt.colorbar()
#cs = mp.pcolor(x,y,abs_grad,cmap=plt.cm.inferno,latlon=True)

plt.title('alb, highest; toa imbal')
  

#%% Plot points on top of roughness sensitivity map r^2 values
#
#[X,Y] = np.meshgrid(lon,lat)
#
#for prop in sfc_props:
#
#    toa_imbal = np.array(energy['ANN'][prop]['imbal'][-1,:,:] - energy['ANN'][prop]['imbal'][0,:,:])
#    grad_toa_imbal = np.gradient(toa_imbal)
#           
#    
#    fig = plt.figure(figsize=(8,8))
#    ax = fig.add_axes([0.1,0.1,0.8,0.8])
#    mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
#    mp.drawcoastlines()
#    mp.drawmapboundary(fill_color='1.')  # make map background white
#    parallels = np.arange(-90.,90,20.)
#    mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
#    meridians = np.arange(0.,360.,20.)
#    mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
#    ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
#    cmap_abs = plt.cm.viridis
#    clim_abs = [0.5,1]
#    
#    units = 'unitless'
#    uproj, vproj, xx, yy = mp.transform_vector(grad_toa_imbal[0],grad_toa_imbal[1],
#                                               X,Y,nx=np.size(lon),ny=np.size(lat),returnxy=True,masked=True)
#    mp.quiver(xx,yy,uproj,vproj)
#    
#    ax=plt.gca()
#    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#                    item.set_fontsize(12)
#    
#    ttl = 'Gradient in TOA energy imbalance for '+prop
#    plt.title(ttl,fontsize=12)
#        
#    
#    
#    # annotate with date/time
#    ax = plt.gca()
#    ax.text(-0.05,-0.25, time.strftime("%x")+'\n log_hc r$^2$' ,fontsize='10',
#                     ha = 'left',va = 'center',
#                     transform = ax.transAxes)
#    
#    plt.show()
#    #filename = 'log_hc_r2_with_locations'
#    #fig_png = figpath+'/sensitivity/point_maps/'+filename+'.png'
#    
#    #fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
#    #                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
#    #                    pad_inches=0.1,frameon=None)
#    
#    plt.close()        

       
   