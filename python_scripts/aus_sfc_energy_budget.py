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
#sims = ['global_a2_cv2_hc0.1_rs100',
#       'global_a1_cv2_hc0.1_rs100','global_a3_cv2_hc0.1_rs100',
#       'global_a2_cv2_hc0.5_rs100','global_a2_cv2_hc2_rs100',
#       'global_a2_cv2_hc0.1_rs30','global_a2_cv2_hc0.1_rs200']
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
#ds1 = ds_cam['global_a2_cv2_hc0.1_rs100']
#ds2 = ds_cam['global_a2_cv2_hc2_rs100']
#
## land files
#dsl0 = ds_clm['global_a2_cv2_hc0.5_rs100']
#dsl1 = ds_clm['global_a2_cv2_hc0.1_rs100']
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
#%%
"""
    Goal:
        
        Draw a box over Australia.
        
        Break down the surface energy budget for each experiment over that box.
        
        Do this seasonally - expect to see a difference between MAM/SON, as
        the sensitivity is different over AUS in those seasons.
        
        Also check stability of atm over surface, and plot (on a map) that. 
        
        (Its sensitivity to ROUGHNESS that we're particularly interested in here)
        
"""


#%%
"""
    Find an Australia box
"""
ds1 = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
lat = ds1.lat.values
lon = ds1.lon.values
lev = ds1.lev.values

fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0) # can't make it start anywhere other than 180???
mp.drawcoastlines()
mp.drawmapboundary(fill_color='1.')  # make map background white
parallels = np.arange(-90.,90,20.)
mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
meridians = np.arange(0.,360.,20.)
mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])

ind_left = 48
ind_right = 59
ind_top = 37
ind_bot = 31

x1,y1 = mp(lon[ind_left],lat[ind_bot])
x2,y2 = mp(lon[ind_right],lat[ind_bot])
x3,y3 = mp(lon[ind_right],lat[ind_top])
x4,y4 = mp(lon[ind_left],lat[ind_top])
poly = plt.Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor=None,edgecolor='green',linewidth=3,fill=False)
plt.gca().add_patch(poly)


#x, y = mp(idx_x,idx_y)
#print(ds1.lat.values[idx_y])
#print(ds1.lon.values[idx_x])
#print(x)
#print(y)

#lon_temp = ds1.lon.values[idx_x]
#lat_temp = ds1.lat.values[idx_y]
#x, y = mp(lon_temp,lat_temp)
#mp.plot(x,y,'D-', markersize=8, linewidth=4, color='k', markerfacecolor='m')
ttl = ( 'AussieBox: \n %.1f' % lon[ind_left] + ' to %.1f' % lon[ind_right] + ' lon, \n %.1f ' 
        % lat[ind_bot]  + ' to %.1f' % lat[ind_top] + ' lat' )
print(ttl)
plt.title(ttl,fontsize=10)

# annotate with date/time
ax = plt.gca()
ax.text(-0.3,-0.6, time.strftime("%x")+'\n',fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)

plt.show()
filename = 'AssuieBoxMap'
fig_png = figpath+'/sensitivity/point_maps/'+filename+'.png'

fig.savefig(fig_png,dpi=1200,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)


plt.close()


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


# evaporative resistance:
ds_low['rs'] = ds_cam['global_a2_cv2_hc0.1_rs30_cheyenne']
ds_med['rs'] = ds_cam['global_a2_cv2_hc0.1_rs100_cheyenne']
ds_high['rs'] = ds_cam['global_a2_cv2_hc0.1_rs200_cheyenne']

dsl_low['rs'] = ds_clm['global_a2_cv2_hc0.1_rs30_cheyenne']
dsl_med['rs'] = ds_clm['global_a2_cv2_hc0.1_rs100_cheyenne']
dsl_high['rs'] = ds_clm['global_a2_cv2_hc0.1_rs200_cheyenne']


# atmospheric variable to evaluate:
atm_var= 'TREFHT'
units[atm_var] = ds_low['alb'][atm_var].units
   
sfc_props = ['alb','rs','hc','log_hc']
#sfc_prop_ranges = np.array([ [0.1, 0.2, 0.3],
#                  [0.5, 1., 2.],
#                  [30., 100., 200.],
#                  [np.log(0.5), np.log(1.), np.log(2.)]])
sfc_prop_ranges = np.array([ [0.1, 0.2, 0.3,np.nan,np.nan],
                            [30., 100., 200.,np.nan,np.nan],
                            [0.01,0.05,0.1,0.5, 1., 2.],
                            [np.log(0.01),np.log(0.05),np.log(0.1),np.log(0.5), np.log(1.), np.log(2.)]])

#print(np.shape(sfc_prop_ranges))

#print(sfc_prop_ranges)

i=0

seasons = ['ANN','DJF','MAM','JJA','SON']

# Make empty AUS response library
australia = {}

fluxes = ['SHFLX','LHFLX','FSNS','FLNS','FSNSC','FLNSC']

for flux in fluxes:
    australia[flux] = {}

    for sea in seasons: 
        australia[flux][sea] = {}

for sea in seasons: 
    atm_resp[sea] = {}
    

for prop in sfc_props:
    pert[prop] = sfc_prop_ranges[i]

    

    if np.isnan(pert[prop][3]):
        #print(prop)
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
        #print(prop)
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
    
    
    
    
    """ 
    Australia response: average surface fluxes in AUS box (defined above)
        ind_left = 48
        ind_right = 59
        ind_top = 37
        ind_bot = 31
    """
    # make library 
    for flux in fluxes:
        ds0 = ds_low['alb']
        australia[flux]['units'] = ds0[flux].units

        
        # Split differently for alb,rs than hc, log_hc
        if np.isnan(pert[prop][3]):
            #print(prop)
            ds1 = ds_low[prop]
            ds2 = ds_med[prop]
            ds3 = ds_high[prop]
        
            # annual mean response
            data_all = np.array([ds1[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds2[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds3[flux].values[:,ind_bot:ind_top,ind_left:ind_right] ])
            
            aus_area = np.nansum( np.nansum( area_f19[ind_bot:ind_top,ind_left:ind_right] 
                            * landmask[ind_bot:ind_top,ind_left:ind_right] ) )
            
            data_all_aus = (np.nansum(np.nansum( data_all * area_f19[ind_bot:ind_top,ind_left:ind_right] *
                                               landmask[ind_bot:ind_top,ind_left:ind_right],3 ),2)
                                    /aus_area )
            
            data_ann = np.mean(data_all_aus[:,:],1)
            data_djf = np.mean(data_all_aus[:,[11,0,1]],1)
            data_mam = np.mean(data_all_aus[:,[2,3,4]],1)
            data_jja = np.mean(data_all_aus[:,[5,6,7]],1)
            data_son = np.mean(data_all_aus[:,[8,9,10]],1)
            
            australia[flux]['ANN'][prop] = data_ann
            australia[flux]['DJF'][prop] = data_djf
            australia[flux]['MAM'][prop] = data_mam
            australia[flux]['JJA'][prop] = data_jja
            australia[flux]['SON'][prop] = data_son
            
        else:
            ds1 = ds_low1[prop] #0.01
            ds2 = ds_low2[prop]  #0.05
            ds3 = ds_med1[prop]  #0.1
            ds4 = ds_med2[prop]    #0.5
            ds5 = ds_high1[prop]    #1
            ds6 = ds_high2[prop]    #2
      
        
            # annual mean response
            data_all = np.array([ds1[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds2[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds3[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds4[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds5[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds6[flux].values[:,ind_bot:ind_top,ind_left:ind_right] ])
            
            aus_area = np.nansum( np.nansum( area_f19[ind_bot:ind_top,ind_left:ind_right] 
                            * landmask[ind_bot:ind_top,ind_left:ind_right] ) )
            
            data_all_aus = (np.nansum(np.nansum( data_all * area_f19[ind_bot:ind_top,ind_left:ind_right] *
                                               landmask[ind_bot:ind_top,ind_left:ind_right],3 ),2)
                                    /aus_area )
            
            data_ann = np.mean(data_all_aus[:,:],1)
            data_djf = np.mean(data_all_aus[:,[11,0,1]],1)
            data_mam = np.mean(data_all_aus[:,[2,3,4]],1)
            data_jja = np.mean(data_all_aus[:,[5,6,7]],1)
            data_son = np.mean(data_all_aus[:,[8,9,10]],1)
            
            australia[flux]['ANN'][prop] = data_ann
            australia[flux]['DJF'][prop] = data_djf
            australia[flux]['MAM'][prop] = data_mam
            australia[flux]['JJA'][prop] = data_jja
            australia[flux]['SON'][prop] = data_son
            
            
    
    #print(np.shape(atm_resp_djf[prop]))
    i=i+1






#%%
"""
    calculate the surface energy budget terms over that box for each simulation, 
    in each season. Store in a library?
"""

#%%
"""
    Plot (bar graphs) of the sfc energy budget, seasonally
    
    MAM lhflx 12345  shflx 12345 
    SON lhflx 12345  shflx 12345 
    
    ... might be hard to compare differneces. Try 
    lh 1mam 1son 2mam 2son etc ... green for mam, blue for son, for example
"""

for prop in sfc_props:
    
    for sea in seasons:
        
        """
            First, do a bar plot with ALL the fluxes. Then break up? This could get chunky.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8,4))
        
        # number of experiments
        ngroups = np.shape(australia['SHFLX']['ANN'][prop])[0]
        nflux = np.shape(fluxes)[0]
        
        index = np.arange(ngroups)
        bar_width=0.2
        opacity=0.8
        
        # fluxdata per experiment:
        i=0
        # loop over fluxes, make a vector for each
        flux_array = np.zeros([nflux,ngroups])
        
        shflx_data = australia['SHFLX'][sea][prop]
        lhflx_data = australia['LHFLX'][sea][prop]
        fsns_data  = australia['FSNS'][sea][prop]
        flns_data  = australia['FLNS'][sea][prop]
        fsnsc_data = australia['FSNSC'][sea][prop]
        flnsc_data = australia['FLNSC'][sea][prop]
        
        
        i=0
        for flux in fluxes:
            
            for exp in range(ngroups):
                
                flux_array[i,:] = australia[flux][sea][prop]
            
            i = i+1
        
        
        shflx_bars = plt.bar(index,shflx_data, bar_width,
                                     alpha=opacity,
                                     color = 'orangered',
                                     label = 'SHFLX')
        
        lhflx_bars = plt.bar(index + bar_width,lhflx_data, bar_width,
                                     alpha=opacity,
                                     color = 'dodgerblue',
                                     label = 'LHFLX')
        
        flns_bars = plt.bar(index + 2*bar_width,flns_data, bar_width,
                                     alpha=opacity,
                                     color = 'forestgreen',
                                     label = 'FLNS')
        
#        fsns_bars = plt.bar(index + 3*bar_width,fsns_data, bar_width,
#                                     alpha=opacity,
#                                     color = 'gold',
#                                     label = 'FSNS')
        
        
        
#        fsnsc_bars = plt.bar(index + 4*bar_width,fsnsc_data, bar_width,
#                                     alpha=opacity,
#                                     color = 'goldenrod',
#                                     label = 'FSNSC')
#        
#        flnsc_bars = plt.bar(index + 5*bar_width,flnsc_data, bar_width,
#                                     alpha=opacity,
#                                     color = 'mediumpurple',
#                                     label = 'FLNSC')
        
        
        # set x-ticks:
        if prop=='alb':
            prop_units = 'unitless'
            exp_tick_labels = np.array([str(x) for x in pert[prop]])[0:3]
        elif prop=='rs':
            prop_units = 's/m'
            exp_tick_labels = np.array([str(x) for x in pert[prop]])[0:3]
        elif prop=='hc':
            prop_units = 'm'
            exp_tick_labels =  np.array([str(x) for x in pert[prop]])
        elif prop=='log_hc':
            prop_units = 'log(m)'
            exp_tick_labels = np.array(['log('+str(x)+')' for x in pert['hc']])
        
        plt.xlabel('Perturbation Intensity ['+prop_units+']')
        plt.ylabel('W/m2')
        plt.title('Surface Energy Budget Breakdown ('+prop+', '+sea+')')
        
        
        
        
        
        plt.xticks(index + 3*bar_width ,exp_tick_labels)
        plt.legend(fontsize=10,bbox_to_anchor=(0.,-0.1),
                               loc=2,ncol=1,borderaxespad=0.) 
        plt.tight_layout()
        plt.show()
        
        
        # Full-figure save:
        filename = 'AUS_sfcEbudget_barchart_' + prop + '_' + sea
        fig_png = figpath+'/aus_sfcEbudget/'+filename+'.png'
        print(fig_png)
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                            edgecolor='w',orientation='portrait',bbox_inches='tight', 
                            pad_inches=0.1,frameon=None)
        
#        for exp in range(ngroups):
#            
#            flux_bars[exp] = plt.bar(index,flux_array[:,exp],bar_width,
#                                     alpha=opacity,
#                                     color = col)


#%%
"""
    plot map of inversions over AUS for each season
    
    T2m - Tsfc, T10-T2m, T10-Tsfc
    
    crop map to Aus (see EAGER work for that)
    
"""
for prop in sfc_props:
    pert[prop] = sfc_prop_ranges[i]

    

    if np.isnan(pert[prop][3]):
        #print(prop)
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
        #print(prop)
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
    
    
    
    
    """ 
    Australia response: average surface fluxes in AUS box (defined above)
        ind_left = 48
        ind_right = 59
        ind_top = 37
        ind_bot = 31
    """
    # make library 
    #for flux in fluxes:
        ds0 = ds_low['alb']
        australia['stability']['units'] = ds0['TS'].units

        
        # Split differently for alb,rs than hc, log_hc
        if np.isnan(pert[prop][3]):
            #print(prop)
            ds1 = ds_low[prop]
            ds2 = ds_med[prop]
            ds3 = ds_high[prop]
        
            # annual mean response
            data_all = np.array([ds1['stability].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds2[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds3[flux].values[:,ind_bot:ind_top,ind_left:ind_right] ])
            
            aus_area = np.nansum( np.nansum( area_f19[ind_bot:ind_top,ind_left:ind_right] 
                            * landmask[ind_bot:ind_top,ind_left:ind_right] ) )
            
            data_all_aus = (np.nansum(np.nansum( data_all * area_f19[ind_bot:ind_top,ind_left:ind_right] *
                                               landmask[ind_bot:ind_top,ind_left:ind_right],3 ),2)
                                    /aus_area )
            
            data_ann = np.mean(data_all_aus[:,:],1)
            data_djf = np.mean(data_all_aus[:,[11,0,1]],1)
            data_mam = np.mean(data_all_aus[:,[2,3,4]],1)
            data_jja = np.mean(data_all_aus[:,[5,6,7]],1)
            data_son = np.mean(data_all_aus[:,[8,9,10]],1)
            
            australia['stability']['ANN'][prop] = data_ann
            australia['stability']['DJF'][prop] = data_djf
            australia['stability']['MAM'][prop] = data_mam
            australia['stability']['JJA'][prop] = data_jja
            australia['stability']['SON'][prop] = data_son
            
        else:
            ds1 = ds_low1[prop] #0.01
            ds2 = ds_low2[prop]  #0.05
            ds3 = ds_med1[prop]  #0.1
            ds4 = ds_med2[prop]    #0.5
            ds5 = ds_high1[prop]    #1
            ds6 = ds_high2[prop]    #2
      
        
            # annual mean response
            data_all = np.array([ds1[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds2[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds3[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds4[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds5[flux].values[:,ind_bot:ind_top,ind_left:ind_right],
                                 ds6[flux].values[:,ind_bot:ind_top,ind_left:ind_right] ])
            
            aus_area = np.nansum( np.nansum( area_f19[ind_bot:ind_top,ind_left:ind_right] 
                            * landmask[ind_bot:ind_top,ind_left:ind_right] ) )
            
            data_all_aus = (np.nansum(np.nansum( data_all * area_f19[ind_bot:ind_top,ind_left:ind_right] *
                                               landmask[ind_bot:ind_top,ind_left:ind_right],3 ),2)
                                    /aus_area )
            
            data_ann = np.mean(data_all_aus[:,:],1)
            data_djf = np.mean(data_all_aus[:,[11,0,1]],1)
            data_mam = np.mean(data_all_aus[:,[2,3,4]],1)
            data_jja = np.mean(data_all_aus[:,[5,6,7]],1)
            data_son = np.mean(data_all_aus[:,[8,9,10]],1)
            
            australia['stability']['ANN'][prop] = data_ann
            australia['stability']['DJF'][prop] = data_djf
            australia['stability']['MAM'][prop] = data_mam
            australia['stability']['JJA'][prop] = data_jja
            australia['stability']['SON'][prop] = data_son
            
            
    
    #print(np.shape(atm_resp_djf[prop]))
    i=i+1





#%%
"""
    
"""

#%%
"""
    
"""

#%%
"""
    
"""

#%%
"""
    
"""



