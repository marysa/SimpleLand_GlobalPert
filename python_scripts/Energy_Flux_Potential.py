#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:40:48 2017

@author: mlague
"""

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

#%% 
        
def get_MSE(z,T,q):
    
    # cam fileds: Z3 [m], T [K], Q [kg/kg]
    
    # Calculates Moist Static Energy as
    # MSE = c_p * T + g * z + L_v * q
    #       (internal) + (pot'l) + (latent heat due to wv)
    #
    # Where
    #
    # u and v are wind vector fields
    # cp = heat capacity of dry air
    # z is the height
    # T is the absoulte temperature in K (not potential temperature)
    # g is 9.81m/s2 (gravity)
    # q is the water vapour (check units)  ----> [kg/kg]
    # L_v is the latent heat of vapourization
    
    #  Joule = kg m2/s2
    
    g = 9.81    # m/s2
    Lv = 2.257e6   # J/kg   (source: springer link and engineering toolbox)
    cp = 1006      # J/kg/K
    
    
    internal = cp * T     # [ J/kg/K ] * [ K ] = [ J/kg ] = [ kg m2 / s2] / [ kg ] = [ m2/s2 ]
    potential = g * z     # [ m/s2 ] * [ m ] = [ m2/s2 ]
    latent = Lv * q       # [ J/kg ] * [kg/kg] = [J/kg] = [ m2/s2 ]
    
    MSE  = internal + potential + latent
    
    

    return MSE  

def get_EFP(u,v,MSE,lat,lon,lev,ilev, area):
    
    # calculate the energy flux potential as described by equation 1 of Boose 2016
    #
    # Returns:
    #
    # EFP, the Energy Flux Potential X
    # gradEFP, the gradient of the Energy Flux Potential, or the divergent part of 
    #       atmospheric energy transport, 
    # divEFP, the laplacian of Energy Flux Potential, which seasonally averaged should
    #       equal the net energy input to the atmopsheric column
    
    # grad^2 X = grad . int_0^ps ( <u,v>*MSE/g ) dp
    
    # Inputs:
    #   u   zonal wind (x direction, EW)        [m/s]
    #   v   meridional wind (y direction, NS)   [m/s]
    #   MSE moist static energy                 [m2/s2]
    #   lat  degrees latitude
    #   lon  degrees longitude
    #   area area of gridcell (for integrating)  [m2]
    #   lev pressure level                      [hpa] = 1000*[kg/m/s2]
    #   ilev boundaries of pressure levels
    
    # Need dlat and dlon in m, vs degrees, which requires Earth's radius in m
    r = 6378100     # [m]
    
    g = 9.81    # m/s2
    
    # delta lat & lon in degrees
    dlat_d = np.diff(lat)
    dlat_d = np.append(dlat_d,dlat_d[-1])
    
    dlon_d = np.diff(lon)
    dlon_d = np.append(dlon_d,dlon_d[-1])
    
    # convert to delta lat & lon in m
    dlat_m = r * (dlat_d*np.pi/180)
    
    #dlon_m = r * np.abs( np.cos(lon[1:]*np.pi/180.) - np.cos(lon[0:-1]*np.pi/180.) )
    dlon_m = r * np.abs( np.cos(lat[1:]*np.pi/180.) - np.cos(lat[0:-1]*np.pi/180.) )
    dlon_m = np.append(dlon_m,dlon_m[0])
    
    # default size: [12,30,96,144]
    
    integrand_u = np.array(u) * np.array(MSE) / g   # [] = [m/s] * [m2/s2] / [m/s2] = [m2/s]
    integrand_v = np.array(v) * np.array(MSE) / g   # [] = [m/s] * [m2/s2] / [m/s2] = [m2/s]
    
    # reshape so 30 dimension is last - not SURE I'm doing this right. A "permute" function would be nice.
    #integrand_u = np.reshape(integrand_u,[12,96,144,30])
    #integrand_v = np.reshape(integrand_v,[12,96,144,30])
    
    dp = np.array( np.diff(ilev)*1000 )     # [ hpa ]*1000 ti [pa]
    
    # integrate wrt pressure
    gradEFP_u = np.sum(integrand_u*dp[None,:,None,None],1)    # [] = [m2/s]*[pa]
    gradEFP_v = np.sum(integrand_v*dp[None,:,None,None],1)    # [] = [m2/s]*[pa] = [kg m /s3]
    
    gradEFP = {}
    gradEFP['u'] = gradEFP_u
    gradEFP['v'] = gradEFP_v
    
    # instead of rollaxis, do *dlat[:,None]
    
    # integrate in x (u) and y (v) to get EFP
    EFP={}
    EFP['u'] = ((gradEFP_u)*dlon_m[None,None,:])    # check summing dimensino - should be x dir, 144
    #gradEFP_v = np.rollaxis(gradEFP_v,1,start=3)*dlat_m[None,:,None]
    EFP_v = (gradEFP_v)*dlat_m[None,:,None]

    #EFP['v'] = np.rollaxis((gradEFP_v),2,1) 
    EFP['v'] = EFP_v
    print(np.shape(EFP['v']))
    
    # laplacian of EFP: d/dx2 + d/dy2 , already have dEFP/dz as gradEFP_u, so take d/dx of that
    print(np.shape(gradEFP_u))
    print(np.shape(dlon_m))
    print(np.shape(gradEFP_v))
    print(np.shape(dlat_m))
    #divEFP= gradEFP_u/dlon_m[None,None,:] + np.reshape(np.reshape(gradEFP_v,[12,144,96])/dlat_m,[12,96,144])
    divEFP = gradEFP_u/dlon_m[None,None,:] + gradEFP_v/dlat_m[None,:,None]
    
    return EFP, gradEFP, divEFP


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
        atm_resp_ann[prop] = np.array([np.array(ds1.mean('time')[atm_var].values[:,:]),
            np.array(ds2.mean('time')[atm_var].values[:,:]),
            np.array(ds3.mean('time')[atm_var].values[:,:])])
        
        # seasonal responses:
        # (first, make 12 month response, then average over djf, jja, etc)
        #print(np.shape(ds1[atm_var].values))
        resp_mths = np.array([np.array(ds1[atm_var].values[:,:,:]),
                np.array(ds2[atm_var].values[:,:,:]),
                np.array(ds3[atm_var].values[:,:,:])])
        
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

#%% Calculate the moist static energy for each model run

MSE = {}
ds_tree = {}

i = 0

sfc_props_mod = ['alb','rs','hc']

for prop in sfc_props_mod:
    MSE[prop] = {}
    ds_tree[prop] = {}

    for sea in seasons:
        MSE[prop][sea] = {}
        ds_tree[prop][sea] = {}
        
        
            
        if prop == 'hc':
            print(prop)
            ds1 = ds_low1[prop] #0.01
            ds2 = ds_low2[prop]  #0.05
            ds3 = ds_med1[prop]  #0.1
            ds4 = ds_med2[prop]    #0.5
            ds5 = ds_high1[prop]    #1
            ds6 = ds_high2[prop]    #
            
            ds_tree[prop][sea]['low'] = ds1
            ds_tree[prop][sea]['med'] = ds3
            ds_tree[prop][sea]['high'] = ds5
        else:
            print(prop)
            ds1 = ds_low[prop]
            ds2 = ds_med[prop]
            ds3 = ds_high[prop]
            
            ds_tree[prop][sea]['low'] = ds1
            ds_tree[prop][sea]['med'] = ds2
            ds_tree[prop][sea]['high'] = ds3
        
        i = i+1



#%%

# Test the MSE and EPF functions:

ds0 = ds_tree['alb']['ANN']['med']

u = np.array(ds0['U'].values)
v = np.array(ds0['V'].values)
T = np.array(ds0['T'].values)
z = np.array(ds0['Z3'].values)  # I think I probably should actuallly be using pressure coords like the model, translated into z
q = np.array(ds0['Q'].values)

lev = np.array(ds0['lev'].values)
ilev = np.array(ds0['ilev'].values)

MSE = np.array(get_MSE(z,T,q))

#EFP, gradEFP, divEFP = get_EFP(u,v,MSE,lat,lon,lev,ilev, area=area_f19)

MSE_ann = np.mean(MSE,0)

z_ann = np.mean(z,0)

dz = z_ann
dz[1:30,:,:] = z_ann[1:30,:,:] - z_ann[0:29,:,:]

MSE_ann_col_ish = np.sum(MSE_ann*dz,0)

plt.imshow(MSE_ann_col_ish)
plt.colorbar()
plt.show()
plt.close()

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
    
 

