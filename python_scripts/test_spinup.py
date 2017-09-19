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


from matplotlib.ticker import FormatStrFormatter

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

#from planar import BoundingBox

# Avoid having to restart the kernle if I modify my mapping scripts (or anything else)
import imp
#imp.reload(mml_map)
#imp.reload(mml_map_NA)
#imp.reload(mml_neon_box)
import matplotlib.colors as mcolors

#%%
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

# Time Series vars:
cam_vars = ['TS','TREFHT']
clm_vars = ['MML_alb','MML_snow','MML_ts']

# load the file paths and # Open the coupled data sets in xarray
cam_files = {}
clm_files = {}
ds_cam = {}
ds_clm = {}

for run in sims:
    #print ( ext_dir + run + '/means/' + run + '.cam.h0.05-end_year_avg.nc' )
    
    # library of cam/clm files per variable
    cam_files[run] = {}
    clm_files[run] = {}
    ds_cam[run] = {}
    ds_clm[run] = {}
    
    for avar in cam_vars:
        cam_files[run][avar] = ext_dir + run + '/TimeSeries/' + run + '.cam.h0.ts_full.0-50.'+avar+'.nc'
        ds_cam[run][avar] = xr.open_dataset(cam_files[run][avar])
    
    for lvar in clm_vars:
        clm_files[run][lvar] = ext_dir + run + '/TimeSeries/' + run + '.clm2.h0.ts_full.0-50.'+lvar+'.nc'
        ds_clm[run][lvar] = xr.open_dataset(clm_files[run][lvar])
    
    

# open a cam area file produced in matlab using an EarthEllipsoid from a cam5 f19 lat/lon data set
area_f19_mat = sio.loadmat('/home/disk/eos18/mlague/simple_land/scripts/python/analysis//f19_area.mat')
area_f19 = area_f19_mat['AreaGrid']

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



LN,LT = np.meshgrid(lon,lat)

#print(np.shape(LN))
#print(np.shape(landmask))

#%%

for run in sims:
    
    for var in cam_vars:
        
        ds = ds_cam[run][var]
        data = np.array(ds[var])
        units = ds[var].units
        
        # Reshape and mean data so we have annual values (vs monthly)
        
        # only use first 49 years
        data = data[0:588,:,:]
        nyear = 588/12
        data_new = np.reshape(data,[12,nyear,96,144])
        data_new = np.squeeze(np.mean(data_new,0))
        
        # average globally
        data_global = np.sum(np.sum(data_new * area_f19,1),1) / np.sum(area_f19)
        
        years = np.array(range(0,np.int(nyear)))
        
        
        # Plot trend

        fig, axes = plt.subplots(1, 1, figsize=(5,5))
        ax = plt.gca()
        
        plt.plot(years,data_global)
        
        # format axis to write temperatures like "273.1" vs "2.731e2"
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(var + ' [' + units + ']' )
        ax.set_xlabel('Years')
        
        plt.title('Global average spinup of ' + var + ' [' + units + '] \n' + run)
        
        ax.text(0.,-0.1,time.strftime("%x"), fontsize='10',
                         ha = 'left',va = 'center',
                         transform = ax.transAxes)
        
        ylim = ax.get_ylim()
        
        plt.plot([20,20],[ylim[0],ylim[1]],'g:')
         
         
        plt.show()
        plt.close()
        
        filename = run + '_' + var +'_spinup'
        fig_png = figpath+'/'+filename+'.png'
        print(fig_png)
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches='tight', 
                            pad_inches=0.1,frameon=None)
        
    for var in clm_vars:
        
        ds = ds_clm[run][var]
        data = np.array(ds[var])
        units = ds[var].units
        
        # Reshape and mean data so we have annual values (vs monthly)
        
        # only use first 49 years
        data = data[0:588,:,:]
        nyear = 588/12
        data_new = np.reshape(data,[12,nyear,96,144])
        data_new = np.squeeze(np.mean(data_new,0))
        
        if var == 'MML_snow' :
            data_new = data_new*inv_glc_mask
        
        
        # average globally
        data_global = np.nansum(np.nansum(data_new * area_f19 * landmask * landfrac,1),1) / np.nansum(area_f19 * landmask * landfrac)
        
        years = np.array(range(0,np.int(nyear)))
        
        
        # Plot trend

        fig, axes = plt.subplots(1, 1, figsize=(5,5))
        ax = plt.gca()
        
        plt.plot(years,data_global)
        
        # format axis to write temperatures like "273.1" vs "2.731e2"
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(var + ' [' + units + ']' )
        ax.set_xlabel('Years')
        
        plt.title('Global average spinup of ' + var + ' [' + units + '] \n' + run)
        
        ax.text(0.,-0.1,time.strftime("%x"), fontsize='10',
                         ha = 'left',va = 'center',
                         transform = ax.transAxes)
        
        ylim = ax.get_ylim()
        
        plt.plot([20,20],[ylim[0],ylim[1]],'g:')
         
         
        plt.show()
        plt.close()
        
        filename = run + '_' + var +'_spinup'
        fig_png = figpath+'/'+filename+'.png'
        print(fig_png)
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches='tight', 
                            pad_inches=0.1,frameon=None)
        

        
#%%
"""
    Alternatively, plot all the lines on one figure, one fig for each var. 
    Give each run a different colour. Have fun with colours ;)
        
        
    Order of sims:
    sims = ['global_a2_cv2_hc0.1_rs100_cheyenne',
       'global_a1_cv2_hc0.1_rs100_cheyenne','global_a3_cv2_hc0.1_rs100_cheyenne',
       'global_a2_cv2_hc0.01_rs100_cheyenne','global_a2_cv2_hc0.05_rs100_cheyenne',
       'global_a2_cv2_hc0.5_rs100_cheyenne',
       'global_a2_cv2_hc1.0_rs100_cheyenne','global_a2_cv2_hc2.0_rs100_cheyenne',
       'global_a2_cv2_hc0.1_rs30_cheyenne','global_a2_cv2_hc0.1_rs200_cheyenne']

"""

# list of colournames for runs. 10 sims.
run_cols = ['dimgray',
            'firebrick','orange',
            'cyan','darkturquoise','lightskyblue','dodgerblue','slateblue',
            'chartreuse','forestgreen']

# ATM plots

for var in cam_vars:
    
    fig, axes = plt.subplots(1, 1, figsize=(5,5))
    ax = plt.gca()
    
    i=0
    
    for run in sims:
        
        col = run_cols[i]
        ds = ds_cam[run][var]
        
        data = np.array(ds[var])
        units = ds[var].units
        
        # Reshape and mean data so we have annual values (vs monthly)
        
        # only use first 49 years
        data = data[0:588,:,:]
        nyear = 588/12
        data_new = np.reshape(data,[12,nyear,96,144])
        data_new = np.squeeze(np.mean(data_new,0))
        
        # average globally
        data_global = np.sum(np.sum(data_new * area_f19,1),1) / np.sum(area_f19)
        
        years = np.array(range(0,np.int(nyear)))
        
        
        # plot trend
        lbl = run
        
        plt.plot(years,data_global,color=col,label=lbl)

        
        i = i+1
    
    

    # get handles, plot legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,labels,bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
    
    
    # format axis to write temperatures like "273.1" vs "2.731e2"
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(var + ' [' + units + ']' )
    ax.set_xlabel('Years')
        
    plt.title('Global average spinup of ' + var + ' [' + units + ']')
        
    ax.text(0.,-0.1,time.strftime("%x"), fontsize='10',
                         ha = 'left',va = 'center',
                         transform = ax.transAxes)
        
    ylim = ax.get_ylim()
        
    plt.plot([20,20],[ylim[0],ylim[1]],'g:')
         
         
    plt.show()
    plt.close()
        
    filename = 'allruns_' + var +'_spinup'
    fig_png = figpath+'/'+filename+'.png'
    print(fig_png)
    fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches='tight', 
                            pad_inches=0.1,frameon=None)
    


# LND plots


for var in clm_vars:
    
    fig, axes = plt.subplots(1, 1, figsize=(5,5))
    ax = plt.gca()
    
    i=0
    
    for run in sims:
        
        col = run_cols[i]
        ds = ds_clm[run][var]
        
        data = np.array(ds[var])
        units = ds[var].units
        
        # Reshape and mean data so we have annual values (vs monthly)
        
        # only use first 49 years
        data = data[0:588,:,:]
        nyear = 588/12
        data_new = np.reshape(data,[12,nyear,96,144])
        data_new = np.squeeze(np.mean(data_new,0))
        
        if var == 'MML_snow':
            data_new = data_new*inv_glc_mask
        
        # average globally
        data_global = np.nansum(np.nansum(data_new * area_f19 * landfrac * landmask,1),1) / np.nansum(area_f19*landfrac*landmask)
        
        years = np.array(range(0,np.int(nyear)))
        
        
        # plot trend
        lbl = run
        
        plt.plot(years,data_global,color=col,label=lbl)

        
        i = i+1
    
    

    # get handles, plot legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,labels,bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
    
    
    # format axis to write temperatures like "273.1" vs "2.731e2"
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(var + ' [' + units + ']' )
    ax.set_xlabel('Years')
        
    plt.title('Global average spinup of ' + var + ' [' + units + ']')
        
    ax.text(0.,-0.1,time.strftime("%x"), fontsize='10',
                         ha = 'left',va = 'center',
                         transform = ax.transAxes)
        
    ylim = ax.get_ylim()
        
    plt.plot([20,20],[ylim[0],ylim[1]],'g:')
         
         
    plt.show()
    plt.close()
        
    filename = 'allruns_' + var +'_spinup'
    fig_png = figpath+'/'+filename+'.png'
    print(fig_png)
    fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches='tight', 
                            pad_inches=0.1,frameon=None)
    




#%%


#%%




#%%


