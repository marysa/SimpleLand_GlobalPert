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

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import brewer2mpl as cbrew

# OS interaction
import os
import sys

from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)

from mml_mapping_fun import mml_map
from custom_python_mml_cmap import make_colormap, mml_cmap

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
ext_dir = '/home/disk/eos18/mlague/simple_land/output/global_pert/'

# Coupled simulations:
sims = ['global_a2_cv2_hc1_rs100',
       'global_a1_cv2_hc1_rs100','global_a3_cv2_hc1_rs100',
       'global_a2_cv2_hc0.5_rs100','global_a2_cv2_hc2_rs100',
       'global_a2_cv2_hc1_rs30','global_a2_cv2_hc1_rs200']

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

ds = ds_clm['global_a2_cv2_hc1_rs100']
lat = ds['lat'].values
lon = ds['lon'].values
landmask = ds['landmask'].values

LN,LT = np.meshgrid(lon,lat)

#print(np.shape(LN))
#print(np.shape(landmask))


#%%

# Define data sets and subtitles
ds0 = ds_cam['global_a2_cv2_hc0.5_rs100']

# perturbatinos
ds1 = ds_cam['global_a2_cv2_hc1_rs100']
ds2 = ds_cam['global_a2_cv2_hc2_rs100']

# land files
dsl0 = ds_clm['global_a2_cv2_hc0.5_rs100']
dsl1 = ds_clm['global_a2_cv2_hc1_rs100']
dsl2 = ds_clm['global_a2_cv2_hc2_rs100']


#%%  Get masks:

ds = ds_clm['global_a2_cv2_hc1_rs100']
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

#------------------------------------
# fill in data sets

# albedo:
ds_low['alb'] = ds_cam['global_a1_cv2_hc1_rs100']
ds_med['alb'] = ds_cam['global_a2_cv2_hc1_rs100']
ds_high['alb'] = ds_cam['global_a3_cv2_hc1_rs100']

dsl_low['alb'] = ds_clm['global_a1_cv2_hc1_rs100']
dsl_med['alb'] = ds_clm['global_a2_cv2_hc1_rs100']
dsl_high['alb'] = ds_clm['global_a3_cv2_hc1_rs100']

# roughness:
ds_low['hc'] = ds_cam['global_a2_cv2_hc0.5_rs100']
ds_med['hc'] = ds_cam['global_a2_cv2_hc1_rs100']
ds_high['hc'] = ds_cam['global_a2_cv2_hc2_rs100']

dsl_low['hc'] = ds_clm['global_a2_cv2_hc0.5_rs100']
dsl_med['hc'] = ds_clm['global_a2_cv2_hc1_rs100']
dsl_high['hc'] = ds_clm['global_a2_cv2_hc2_rs100']

# evaporative resistance:
ds_low['rs'] = ds_cam['global_a2_cv2_hc1_rs30']
ds_med['rs'] = ds_cam['global_a2_cv2_hc1_rs100']
ds_high['rs'] = ds_cam['global_a2_cv2_hc1_rs200']

dsl_low['rs'] = ds_clm['global_a2_cv2_hc1_rs30']
dsl_med['rs'] = ds_clm['global_a2_cv2_hc1_rs100']
dsl_high['rs'] = ds_clm['global_a2_cv2_hc1_rs200']


# atmospheric variable to evaluate:
atm_var= 'TREFHT'
units[atm_var] = ds1[atm_var].units
   
sfc_props = ['alb','hc','rs']
sfc_prop_ranges = np.array([ [0.1, 0.2, 0.3],
                  [0.5, 1., 2.],
                  [30., 100., 200.]])
print(np.shape(sfc_prop_ranges))

print(sfc_prop_ranges)

i=0

seasons = ['ANN','DJF','MAM','JJA','SON']

for sea in seasons: 
    atm_resp[sea] = {}


for prop in sfc_props:
    pert[prop] = sfc_prop_ranges[i,:]
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



    
    

print(prop)
print(pert)
print(atm_resp)
print(pert['alb'])



#%%

#%% 
"""
    Plot the experiment with the maximum (and minimum) temperature.
    Ie, albedo should be "1" everywhere. If hc is parabolic, it should be 2, while
    if it is more linear, it should be 1 or 3. Plot minimum, aslo.
    
"""
myvar = 'TREFHT'
clim_exp = [-1,3]
#cm = plt.get_cmap('jet',5)   # get 3 discrete colours. I think.
cm =  mpl.colors.ListedColormap(['cornflowerblue','forestgreen','mediumblue'])
   
# Loop over properties:
for prop in sfc_props: 
    

    # Loop over seasons:
    for sea in seasons:
        # values:
        # atm_resp['ANN'][prop]
        
         # what a mess... haven't figured out element-wise np.where statements, sadly...
        temp = np.array(atm_resp[sea][prop])
        
        max_exp = np.argmax(temp,axis=0)
        min_exp = np.argmin(temp,axis=0)
    
        #prop = 'alb'
        #myvar = 'TREFHT'
        ds0 = ds_cam['global_a2_cv2_hc1_rs100']
        mask_name = 'nomask'
        
        
        ttl_main = prop #'Albedo'
        filename = 'max_min_exp_'+prop+'_'+mask_name+'_'+sea
        
        
        cmap_abs = plt.cm.viridis
        cmap_diff = plt.cm.RdBu_r
        
        fig, axes = plt.subplots(1, 2, figsize=(12,6))
        
        ax0 = axes.flatten()[0]
        plt.sca(ax0)
        ttl = 'experiment with MAX temperature'
        units = 'unitless, low val to high'
        #clim_diff = [-.01,.01]
        mapdata = max_exp
        mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_exp,colmap=cm, cb_ttl='units: '+units )   #plt.cm.BuPu_r
        ax=ax0
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)
        
        ax1 = axes.flatten()[1]
        plt.sca(ax1)
        ttl = 'experiment with MIN temperature'
        units = 'unitless, low val to high'
       # clim_diff = [-25,25]
        #clim_abs = clim_diff
        mapdata = min_exp
        mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_exp,colmap=cm, cb_ttl='units: '+units )   #plt.cm.BuPu_r
        ax=ax1
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)
        
        # Annotate with season, variable, date
        ax0.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax0.transAxes)
        
        fig_name = figpath+'/sensitivity/'+filename+'.eps'
        fig.savefig(fig_name,dpi=1200,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)
        
        fig_name = figpath+'/sensitivity/'+filename+'.png'
        fig.savefig(fig_name,dpi=1200,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)
        
        plt.close()
        
        
    # end seasonal loop.
    
# end property loop
    




#%%



#%%



#%%




