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
#ds_low['hc_big3'] = ds_cam['global_a2_cv2_hc0.5_rs100']
#ds_med['hc_big3'] = ds_cam['global_a2_cv2_hc1_rs100']
#ds_high['hc_big3'] = ds_cam['global_a2_cv2_hc2_rs100']
#
#dsl_low['hc_big3'] = ds_clm['global_a2_cv2_hc0.5_rs100']
#dsl_med['hc_big3'] = ds_clm['global_a2_cv2_hc1_rs100']
#dsl_high['hc_big3'] = ds_clm['global_a2_cv2_hc2_rs100']
#
#
## log rel'n roughness:
#ds_low['log_hc_big3'] = ds_cam['global_a2_cv2_hc0.5_rs100']
#ds_med['log_hc_big3'] = ds_cam['global_a2_cv2_hc1_rs100']
#ds_high['log_hc_big3'] = ds_cam['global_a2_cv2_hc2_rs100']
#
#dsl_low['log_hc_big3'] = ds_clm['global_a2_cv2_hc0.5_rs100']
#dsl_med['log_hc_big3'] = ds_clm['global_a2_cv2_hc1_rs100']
#dsl_high['log_hc_big3'] = ds_clm['global_a2_cv2_hc2_rs100']

# roughness:
ds_low['hc_big3'] = ds_cam['global_a2_cv2_hc0.5_rs100_cheyenne']
ds_med['hc_big3'] = ds_cam['global_a2_cv2_hc1.0_rs100_cheyenne']
ds_high['hc_big3'] = ds_cam['global_a2_cv2_hc2.0_rs100_cheyenne']


dsl_low['hc_big3'] = ds_clm['global_a2_cv2_hc0.5_rs100_cheyenne']
dsl_med['hc_big3'] = ds_clm['global_a2_cv2_hc1.0_rs100_cheyenne']
dsl_high['hc_big3'] = ds_clm['global_a2_cv2_hc2.0_rs100_cheyenne']


# log rel'n roughness:
ds_low['log_hc_big3'] = ds_cam['global_a2_cv2_hc0.5_rs100_cheyenne']
ds_med['log_hc_big3'] = ds_cam['global_a2_cv2_hc1.0_rs100_cheyenne']
ds_high['log_hc_big3'] = ds_cam['global_a2_cv2_hc2.0_rs100_cheyenne']

dsl_low['log_hc_big3'] = ds_clm['global_a2_cv2_hc0.5_rs100_cheyenne']
dsl_med['log_hc_big3'] = ds_clm['global_a2_cv2_hc1.0_rs100_cheyenne']
dsl_high['log_hc_big3'] = ds_clm['global_a2_cv2_hc2.0_rs100_cheyenne']

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
   
sfc_props = ['hc_big3','log_hc_big3']
#sfc_prop_ranges = np.array([ [0.1, 0.2, 0.3],
#                  [0.5, 1., 2.],
#                  [30., 100., 200.],
#                  [np.log(0.5), np.log(1.), np.log(2.)]])
sfc_prop_ranges = np.array([ 
                            [0.5, 1., 2.],
                            [np.log(0.5), np.log(1.), np.log(2.)]])
print(np.shape(sfc_prop_ranges))

print(sfc_prop_ranges)

seasons = ['ANN','DJF','MAM','JJA','SON']

i=0

for sea in seasons: 
    atm_resp[sea] = {}


for prop in sfc_props:
    pert[prop] = sfc_prop_ranges[i]
    
#    if np.isnan(pert[prop][3]):
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
    
#    else:
#        print(prop)
#        ds1 = ds_low1[prop] #0.01
#        ds2 = ds_low2[prop]  #0.05
#        ds3 = ds_med1[prop]  #0.1
#        ds4 = ds_med2[prop]    #0.5
#        ds5 = ds_high1[prop]    #1
#        ds6 = ds_high2[prop]    #2
#        
#        # annual mean response
#        atm_resp_ann[prop] = [ds1.mean('time')[atm_var].values[:,:],
#            ds2.mean('time')[atm_var].values[:,:],
#            ds3.mean('time')[atm_var].values[:,:],
#            ds4.mean('time')[atm_var].values[:,:],
#            ds5.mean('time')[atm_var].values[:,:],
#            ds6.mean('time')[atm_var].values[:,:],
#            ]
#    
#        # seasonal responses:
#        # (first, make 12 month response, then average over djf, jja, etc)
#        #print(np.shape(ds1[atm_var].values))
#        resp_mths = np.array([ds1[atm_var].values[:,:,:],
#                ds2[atm_var].values[:,:,:],
#                ds3[atm_var].values[:,:,:],
#                ds4[atm_var].values[:,:,:],
#                ds5[atm_var].values[:,:,:],
#                ds6[atm_var].values[:,:,:],
#                ])
    
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
#            if np.isnan(xvals[4]):  # they were forced to all be size 6 b/c of roughness. If those >3 are nan, set k to 3.
#                k = 3
#                xvals = xvals[0:3]
                
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
            
            

#%% Sensitivity Plots
do_sens_plots=1 
if do_sens_plots==1:
        
    myvar = 'TREFHT'
    # Loop over properties:
    for prop in sfc_props: 
        
        # set appropriate colour limits
        if prop =='alb':
            clim_dlnd = [-0.01, 0.01]
            clim_datm = [-25,25]
            units='unitless'
        elif prop =='hc_big3':
            clim_dlnd = [-0.5,0.5]
            clim_datm = [-0.5,0.5]
            units='m'
        elif prop=='rs' :
            clim_dlnd = [-15.,15.]
            clim_datm = [-.025,.025]
            units='s/m'
        elif prop =='log_hc_big3':
            clim_dlnd = [-0.5,0.5]
            clim_datm = [-0.25,0.25]
            units='m'
        
        # Loop over seasons:
        #seasons = ['ANN']
        for sea in seasons:
         #   #%% ALBEDO - Unmasked
            
        
            #prop = 'alb'
            #myvar = 'TREFHT'
            ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
            mask_name = 'nomask'
            
            #sea = 'ANN'
            
            mapdata_slope = slope[sea][prop]
            mapdata_inv = slope[sea][prop]**(-1)
            mapdata_r2 = r_value[sea][prop]**2
            
            
            ttl_main = prop #'Albedo'
            filename = 'sens_slopes_'+prop+'_'+mask_name+'_'+sea
            
            
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
            mapdata = mapdata_inv*0.1
            #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units )   #plt.cm.BuPu_r
            mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
            ax=ax0
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)
            #cbar.set_ticklabels(np.linspace(-0.01,0.01,9))
            ax1 = axes.flatten()[1]
            plt.sca(ax1)
            ttl = '$\delta$ T2m per unit change in '+prop
            units = 'K'
           # clim_diff = [-25,25]
            #clim_abs = clim_diff
            mapdata = mapdata_slope
            #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units )   #plt.cm.BuPu_r
            mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_datm,ext='both',disc=True )
            ax=ax1
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)
            
            ax2 = axes.flatten()[2]
            plt.sca(ax2)
            ttl = 'r^2'
            units = 'r^2'
            clim_abs = [0.5,1]
            mapdata = mapdata_r2
            #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units )   #plt.cm.BuPu_r
            mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_r2,ext='min')
            ax=ax2
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)
            
            fig.subplots_adjust(top=1.15)
            fig.suptitle(ttl_main, fontsize=20)   
            
            # Annotate with season, variable, date
            ax0.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                     ha = 'left',va = 'center',
                     transform = ax0.transAxes)
            
            fig_name = figpath+'/sensitivity/'+filename+'.eps'
            fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches='tight', 
                        pad_inches=0.1,frameon=None)
            
            fig_name = figpath+'/sensitivity/'+filename+'.png'
            fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches='tight', 
                        pad_inches=0.1,frameon=None)
     
            # Save the sub-plots as individual panels
            
            # (a) dlnd/datm
            extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
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
            fig.savefig(fig_eps,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                        frameon=None)
            
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
            fig.savefig(fig_eps,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                        frameon=None)
    
           
            plt.close()
            
            
        #    #%% ALBEDO - Land Mask
        
        
            #prop = 'alb'
            #myvar = 'TREFHT'
            ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
            mask_name = 'lndmask'
            #sea = 'ANN'
            
            mapdata_slope = slope[sea][prop]
            mapdata_inv = slope[sea][prop]**(-1)
            mapdata_r2 = r_value[sea][prop]**2
            
            
            ttl_main = prop #'Albedo'
            filename = 'sens_slopes_'+prop+'_'+mask_name+'_'+sea
            
            
            cmap_abs = plt.cm.get_cmap('viridis',11)#plt.cm.viridis()
            cmap_diff = plt.cm.RdBu_r
            
            fig, axes = plt.subplots(1, 3, figsize=(18,6))
            
            ax0 = axes.flatten()[0]
            plt.sca(ax0)
            ttl = '$\delta$ '+prop+' per 0.1K change in T2m'
            #units = 'unitless'
            #clim_diff = [-.01,.01]
            mapdata = mapdata_inv*0.1*bareground_mask
            mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
            #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik,ext='both',disc=True )   #plt.cm.BuPu_r
            mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
            ax=ax0
           # mml_map(LN,LT,mapdata,ds,myvar,proj,title=None,clim=None,colmap=None,cb_ttl=None,disc=None,ncol=None,nticks=None,ext=None):
       
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)
            
            ax1 = axes.flatten()[1]
            plt.sca(ax1)
            ttl = '$\delta$ T2m per unit change in '+prop
            units = 'K'
            #clim_diff = [-25,25]
            #clim_abs = clim_diff
            mapdata = mapdata_slope*bareground_mask
            mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
            #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units )   #plt.cm.BuPu_r
            mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_datm,ext='both',disc=True )
            ax=ax1
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)
            
            ax2 = axes.flatten()[2]
            plt.sca(ax2)
            ttl = 'r^2'
            units = 'r^2'
            clim_abs = [0.5,1]
            mapdata = mapdata_r2*bareground_mask
            mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
            #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units )   #plt.cm.BuPu_r
            mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_r2,ext='min')
            ax=ax2
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)
            
            fig.subplots_adjust(top=1.15)
            fig.suptitle(ttl_main, fontsize=20)    
            
            # Annotate with season, variable, date
            ax0.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                     ha = 'left',va = 'center',
                     transform = ax0.transAxes)
            
            
            fig_name = figpath+'/sensitivity/'+filename+'.eps'
            fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches='tight', 
                        pad_inches=0.1,frameon=None)
            
            fig_name = figpath+'/sensitivity/'+filename+'.png'
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
            fig.savefig(fig_eps,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                        frameon=None)
            
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
            fig.savefig(fig_eps,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                        frameon=None)
            
            plt.close()
            
            
        #    #%% ALBEDO - ocn mask
            
            
           # prop = 'alb'
            #myvar = 'TREFHT'
            ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
            mask_name = 'ocnmask'
            #sea = 'ANN'
            
            mapdata_slope = slope[sea][prop]
            mapdata_inv = slope[sea][prop]**(-1)
            mapdata_r2 = r_value[sea][prop]**2
            
            
            ttl_main = prop #'Albedo'
            filename = 'sens_slopes_'+prop+'_'+mask_name+'_'+sea
            
            
            cmap_abs = plt.cm.viridis
            cmap_diff = plt.cm.RdBu_r
            
            fig, axes = plt.subplots(1, 3, figsize=(18,6))
            
            ax0 = axes.flatten()[0]
            plt.sca(ax0)
            ttl = '$\delta$ '+prop+' per 0.1K change in T2m'
            #units = 'unitless'
            #clim_diff = [-.01,.01]
            mapdata = mapdata_inv*0.1*ocn_glc_mask
            mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
            #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units )   #plt.cm.BuPu_r
            mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
            ax=ax0
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)
            
            ax1 = axes.flatten()[1]
            plt.sca(ax1)
            ttl = '$\delta$ T2m per unit change in '+prop
            units = 'K'
            #clim_diff = [-25,25]
            #clim_abs = clim_diff
            mapdata = mapdata_slope*ocn_glc_mask
            mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
            #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units )   #plt.cm.BuPu_r
            mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_datm,ext='both',disc=True )
            ax=ax1
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)
            
            ax2 = axes.flatten()[2]
            plt.sca(ax2)
            ttl = 'r^2'
            units = 'r^2'
            clim_abs = [0.5,1]
            mapdata = mapdata_r2*ocn_glc_mask
            mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
            #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units )   #plt.cm.BuPu_r
            mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_r2,ext='min')
            ax=ax2
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)
            
            fig.subplots_adjust(top=1.15)
            fig.suptitle(ttl_main, fontsize=20)    
            
            # Annotate with season, variable, date
            ax0.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
                     ha = 'left',va = 'center',
                     transform = ax0.transAxes)
            
            fig_name = figpath+'/sensitivity/'+filename+'.eps'
            fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches='tight', 
                        pad_inches=0.1,frameon=None)
            
            fig_name = figpath+'/sensitivity/'+filename+'.png'
            fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches='tight', 
                        pad_inches=0.1,frameon=None)
    
            # Save the sub-plots as individual panels
            
            # (a) dlnd/datm
            extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
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
            fig.savefig(fig_eps,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                        frameon=None)
            
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
            fig.savefig(fig_eps,dpi=600,transparent=True,facecolor='w',
                        edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                        frameon=None)
            
            plt.close()
            
            
        # end seasonal loop.
        
    # end property loop
        
#%% ALBEDO - SEASONAL

#%% Make some point-local line-graphs of response; draw linear fit and see if log fit is more appropriate

#%% Select Point: South America

location_names = ['NorthAmazon','BC','USA','Eurasia','EqAfrica','Mongolia','Australia','Russia','SouthAfrica','SouthAmerica','NorthAmerica']

locations = {}
for nm in location_names:
    locations[nm] = {}

locations['SouthAfrica']['name'] = 'SouthAfrica'
locations['SouthAfrica']['loc_x'] = 10
locations['SouthAfrica']['loc_y'] = 35
locations['SouthAfrica']['col'] = 'red'

locations['SouthAmerica']['name'] = 'SouthAmerica'
locations['SouthAmerica']['loc_x'] = 125
locations['SouthAmerica']['loc_y'] = 40
locations['SouthAmerica']['col'] = 'deeppink'

locations['NorthAmazon']['name'] = 'NorthAmazon'
locations['NorthAmazon']['loc_x'] = 118
locations['NorthAmazon']['loc_y'] = 49
locations['NorthAmazon']['col'] = 'violet'

locations['USA']['name'] = 'USA'
locations['USA']['loc_x'] = 108
locations['USA']['loc_y'] = 67
locations['USA']['col'] = 'indianred'

locations['NorthAmerica']['name'] = 'NorthAmerica'
locations['NorthAmerica']['loc_x'] = 100
locations['NorthAmerica']['loc_y'] = 70
locations['NorthAmerica']['col'] = 'lightcyan'

locations['BC']['name'] = 'BC'
locations['BC']['loc_x'] = 95
locations['BC']['loc_y'] = 80
locations['BC']['col'] = 'deepskyblue'

locations['Russia']['name'] = 'Russia'
locations['Russia']['loc_x'] = 38
locations['Russia']['loc_y'] = 84
locations['Russia']['col'] = 'mistyrose'

locations['Eurasia']['name'] = 'Eurasia'
locations['Eurasia']['loc_x'] = 20
locations['Eurasia']['loc_y'] = 74
locations['Eurasia']['col'] = 'azure'

locations['Australia']['name'] = 'Australia'
locations['Australia']['loc_x'] = 52
locations['Australia']['loc_y'] = 34
locations['Australia']['col'] = 'orange'

locations['EqAfrica']['name'] = 'EqAfrica'
locations['EqAfrica']['loc_x'] = 9
locations['EqAfrica']['loc_y'] = 50
locations['EqAfrica']['col'] = 'slateblue'

locations['Mongolia']['name'] = 'Mongolia'
locations['Mongolia']['loc_x'] = 46
locations['Mongolia']['loc_y'] = 73
locations['Mongolia']['col'] = 'dodgerblue'

#%% Plot locations
make_maps=0
if make_maps==1:
    
    for nm in location_names:
        location = nm
        idx_x = locations[nm]['loc_x']
        idx_y = locations[nm]['loc_y']
        
        # First order of business: Find the point! 
        
        ds1 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
        lat = ds1.lat.values
        lon = ds1.lon.values
        lev = ds1.lev.values
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
        mp.drawcoastlines()
        mp.drawmapboundary(fill_color='1.')  # make map background white
        parallels = np.arange(-90.,90,20.)
        mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
        meridians = np.arange(0.,360.,20.)
        mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
        
        # North America
        #idx_x = 100
        #idx_y = 70
        
        # South America
        #location = 'SouthAmerica'
        #idx_x = 120
        #idx_y = 40
        
        x, y = mp(idx_x,idx_y)
        print(ds1.lat.values[idx_y])
        print(ds1.lon.values[idx_x])
        print(x)
        print(y)
        
        lon_temp = ds1.lon.values[idx_x]
        lat_temp = ds1.lat.values[idx_y]
        x, y = mp(lon_temp,lat_temp)
        mp.plot(x,y,'D-', markersize=8, linewidth=4, color='k', markerfacecolor='m')
        ttl = 'Location for Linear Point Analysis: '+np.str(lat_temp)+'N, '+np.str(lon_temp)+'E'
        print(ttl)
        plt.title(ttl,fontsize=12)
        
        # annotate with date/time
        ax = plt.gca()
        ax.text(-0.05,-0.05, time.strftime("%x")+'\n' + location,fontsize='10',
                         ha = 'left',va = 'center',
                         transform = ax.transAxes)
        
        plt.show()
        filename = location
        fig_png = figpath+'/sensitivity/point_maps/'+filename+'.png'
        
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches='tight', 
                            pad_inches=0.1,frameon=None)
        
        plt.close()
        
#%% Plot points on top of roughness sensitivity map r^2 values
        
fig = plt.figure(figsize=(8,8))
#ax = fig.add_axes([0.1,0.1,0.8,0.8])
#mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
#mp.drawcoastlines()
#mp.drawmapboundary(fill_color='1.')  # make map background white
#parallels = np.arange(-90.,90,20.)
#mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
#meridians = np.arange(0.,360.,20.)
#mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
ds0 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
cmap_abs = plt.cm.viridis
clim_abs = [0.5,1]

units = 'unitless'

ttl = 'loc_hc r^2 values with selected points for analysis'

mapdata = r_value['ANN']['log_hc_big3']
myvar = 'TREFHT'

mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units,ext='min')
ax=plt.gca()
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)


for nm in location_names:
    location = nm
    idx_x = locations[nm]['loc_x']
    idx_y = locations[nm]['loc_y']
    facecol = locations[nm]['col']
        
    x, y = mp(idx_x,idx_y)

    lon_temp = ds1.lon.values[idx_x]
    lat_temp = ds1.lat.values[idx_y]
    x, y = mp(lon_temp,lat_temp)
    lbl = nm
    mp.plot(x,y,'D', markersize=6, linewidth=4, label=lbl,markeredgecolor='gray',
            markerfacecolor=facecol)#color='k', markerfacecolor='m')
    
plt.legend(bbox_to_anchor=(1.3,1),numpoints=1)
    
ttl = 'Location for Linear Point Analysis: '+np.str(lat_temp)+'N, '+np.str(lon_temp)+'E'
print(ttl)
plt.title(ttl,fontsize=12)

# annotate with date/time
ax = plt.gca()
ax.text(-0.05,-0.25, time.strftime("%x")+'\n log_hc r$^2$' ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)

plt.show()
filename = 'log_hc_r2_with_locations'
fig_png = figpath+'/sensitivity/point_maps/'+filename+'.png'

fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)

plt.close()        

        
        
#%% Loop over locations
if do_line_plots==1:
    
    for nm in location_names:
        location = nm
        idx_x = locations[nm]['loc_x']
        idx_y = locations[nm]['loc_y']
        
        
        
        ##%% Plot the regression points at this location (SA)
        
        myvar = 'TREFHT'
        
        lat = ds1.lat.values
        lev = ds1.lev.values
        
        #del seasons
        #seasons = ['ANN']
        # Loop over seasons:
        for sea in seasons:
            
            # Do all var subplots together. Save individually also.
            i=0
            
            fig, axes = plt.subplots(1, 4, figsize=(16,4))
             
            ylim_alb=np.zeros(2)
            
            # Loop over properties:
            for prop in sfc_props: 
            
                # set appropriate colour limits
                if prop =='alb':
                    clim_dlnd = [-0.01, 0.01]
                    clim_datm = [-25,25]
                    units='unitless'
                    #ylim = []
                elif prop =='hc_big3':
                    clim_dlnd = [-1.,1.]
                    clim_datm = [-0.5,0.5]
                    units='m'
                elif prop=='rs' :
                    clim_dlnd = [-30.,30.]
                    clim_datm = [-.025,.025]
                    units='s/m'
                elif prop =='log_hc_big3':
                    clim_dlnd = [-1.,1.]
                    clim_datm = [-0.25,0.25]
                    units='m'
            
                # Start the plotting!
        
                ax = axes.flatten()[i]
                plt.sca(ax)
                
                if prop == 'log_hc_big3':
                    # log-plot xaxis (will look linear) for log_hc
                    xvals = pert['hc_big3']  #use regular roughness for x
                    if np.isnan(xvals[4]):
                        xvals = xvals[0:3]
                        
                    yvals = np.array(atm_resp[sea][prop])[:,idx_y,idx_x]
                    lbl='datm/dlnd'
                    pdat, = plt.plot(xvals,yvals,'D',label=lbl)   
                    
                    # make x axis logarithmic
                    ax.semilogx()
                    
                    # plot linear fit as y = a + b*ln(x)
                    #xlogs = pert['log_hc_big3']
                    xs = ax.get_xlim()
                    x = np.linspace(xs[0],xs[1],num=20)
                    
                    lin_fit_y = intercept[sea][prop][idx_y,idx_x] + np.log(x)*slope[sea][prop][idx_y,idx_x]
                    
                    lbl = ('Linear fit: ' + '%.2f' % slope[sea][prop][idx_y,idx_x] + ' K/ '
                           + prop + '(' + units +') \n r^2 = ' + np.str(r_value[sea][prop][idx_y,idx_x]) )
                    pfit, = plt.plot(x,lin_fit_y,'r:',linewidth=2.0,label=lbl)
                
                elif prop=='hc_big3':
                    xvals = pert['hc_big3']  #use regular roughness for x
                    if np.isnan(xvals[4]):
                        xvals = xvals[0:3]
                    
                    # plot linear fit to hc, AND log fit
                    
                    # data:
                    yvals = np.array(atm_resp[sea][prop])[:,idx_y,idx_x]
                    lbl='datm/dlnd'
                    pdat, = plt.plot(xvals,yvals,'D',label=lbl)   
                    
                    # Linear:
                    xs = ax.get_xlim()
                    x = np.linspace(xs[0],xs[1],num=20)
                    lin_fit_y = intercept[sea][prop][idx_y,idx_x] + x*slope[sea][prop][idx_y,idx_x]
                    
                    lbl = ('Linear fit: ' + '%.2f' % slope[sea][prop][idx_y,idx_x] + ' K/ '
                           + prop + '(' + units +') \n r^2 = ' + np.str(r_value[sea][prop][idx_y,idx_x]) )
                    pfit_lin, = plt.plot(x,lin_fit_y,'b:',linewidth=2.0,label=lbl)
                    
                    # Log:
                    xs = ax.get_xlim()
                    x = np.linspace(xs[0],xs[1],num=20)
                    
                    lin_fit_y = intercept[sea]['log_hc_big3'][idx_y,idx_x] + np.log(x)*slope[sea]['log_hc_big3'][idx_y,idx_x]
                    
                    lbl = ('Log-linear fit: ' + '%.2f' % slope[sea]['log_hc_big3'][idx_y,idx_x] + ' K/ '
                           + 'log_hc_big3' + '(' + units +') \n r^2 = ' + np.str(r_value[sea]['log_hc_big3'][idx_y,idx_x]) )
                    pfit_log, = plt.plot(x,lin_fit_y,'r:',linewidth=2.0,label=lbl)
                else:
                    # regular linear plot
                    xvals = pert[prop]
                    if np.isnan(xvals[4]):
                        xvals = xvals[0:3]
                    
                    yvals = np.array(atm_resp[sea][prop])[:,idx_y,idx_x]
                    lbl='datm/dlnd'
                    pdat, = plt.plot(xvals,yvals,'D',label=lbl)
                    
                    
                    xs = ax.get_xlim()
                    x = np.linspace(xs[0],xs[1],num=20)
                    lin_fit_y = intercept[sea][prop][idx_y,idx_x] + x*slope[sea][prop][idx_y,idx_x]
                    
                    lbl = ('Linear fit: ' + '%.2f' % slope[sea][prop][idx_y,idx_x] + ' K/ '
                           + prop + '(' + units +') \n r^2 = ' + np.str(r_value[sea][prop][idx_y,idx_x]) )
                    pfit, = plt.plot(x,lin_fit_y,'r:',linewidth=2.0,label=lbl)
                
                
                ylim_fixed = 0
                
                if ylim_fixed == 1:
                    y_tag = 'fixy'    
                    # if we're on albedo, save ylim out to use for the rest, and buffer
                    if prop=='alb':
                        # get albedo ylim (assuming it'll be the biggest)
                        #ax0 = axes.flatten()[0]
                        ylim = ax.get_ylim()
                        yrange = ylim[1]-ylim[0]
                        # buffer by 5% above and below
                        ylim_alb[0]=ylim[0]-0.05*yrange
                        ylim_alb[1] = ylim[1] + 0.05*yrange
                        #ylim_alb[1] = ylim[1]+1
                        ax.set_ylim(bottom=ylim_alb[0],top=ylim_alb[1])
                    else:
                        ax.set_ylim(bottom=ylim_alb[0],top=ylim_alb[1])
                else:
                    y_tag = 'vary'   
                    # buffer by 5% of range, unless range is small, then force to about .5 K
                    ylim = ax.get_ylim()
                    yrange = ylim[1]-ylim[0]
                    
                    if yrange > 0.5:
                        # buffer by 5% above and below
                        ylim_alb[0]=ylim[0]-0.05*yrange
                        ylim_alb[1] = ylim[1] + 0.05*yrange
                        
                    else:   
                        # these ones have too tiny a range, so buffer them manually
                        ylim_alb[0] = ylim[0] - 0.25
                        ylim_alb[1] = ylim[1] + 0.25
                    
                    ax.set_ylim(bottom=ylim_alb[0],top=ylim_alb[1])    
                    
                ttl = '$\Delta$ ' + myvar + ' / $\Delta$ ' + prop
                plt.title(ttl)
                
                # format axis to write temperatures like "273.1" vs "2.731e2"
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                
                # legend:
               
                if prop=='hc_big3':
                    plt.legend(handles=[pdat, pfit_lin,pfit_log],fontsize=10,bbox_to_anchor=(0.,-0.1),
                           loc=2,ncol=1,borderaxespad=0.)
                
                else:
                    plt.legend(handles=[pdat, pfit],fontsize=10,bbox_to_anchor=(0.,-0.1),
                               loc=2,ncol=1,borderaxespad=0.)                
                
                # plt.legend(handles=[pdat, pfit],fontsize=10,bbox_to_anchor=(0.,-0.1),
               #            loc=2,ncol=1,borderaxespad=0.)
                
                i = i+1
             
            
            # Annotate with date/season and save
            ax0 = axes.flatten()[0]
            plt.sca(ax0)
            ax0.text(0.,-0.4,time.strftime("%x")+'\n' + sea + ', '+location + ' ' + y_tag, fontsize='10',
                         ha = 'left',va = 'center',
                         transform = ax0.transAxes)
            
            # Full-figure save:
            filename = location + '_response_curves_' + sea +'_'+y_tag
            fig_png = figpath+'/sensitivity/point_maps/'+filename+'.png'
            print(fig_png)
            fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches='tight', 
                            pad_inches=0.1,frameon=None)
            
            
            # Save (and annotate) individual plots 
        
            
            # b: hc
            for j in range(np.shape(sfc_props)[0]):
        
                ax1 = axes.flatten()[j]
                plt.sca(ax1)
                ax1.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', ' + location + ', ' + sfc_props[j]+ ' ' + y_tag ,fontsize='10',
                             ha = 'left',va = 'center',
                             transform = ax1.transAxes)
                extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                filename = location+'_response_curve_' + sea + '_' + sfc_props[j] +  '_' + y_tag
                fig_png = figpath+'/sensitivity/point_maps/'+filename+'.png'
                vals = extent.extents
                new_extent = extent.from_extents(vals[0]-0.5,vals[1]-1.5,vals[2]+0.25,vals[3]+0.45)
                fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                                edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                                frameon=None)    
            
            plt.close()
            
    
#%% Loop over locations
  
       



