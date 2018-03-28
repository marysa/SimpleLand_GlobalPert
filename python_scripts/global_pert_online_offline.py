#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:54:20 2017

@author: mlague

    New sensitivity slope analysis, for both online & offline simulations

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

# MML's functions:
from mml_mapping_fun import mml_map, discrete_cmap
from custom_python_mml_cmap import make_colormap, mml_cmap
from sensitivity_slope_fun import sensitivity_slope
#from sens_slope_fun2 import sensitivity_slope
from load_masks_coords_fun import get_masks, get_coords, get_seasons
from load_global_pert_data import make_variable_arrays, get_online, get_offline, get_online_ts, get_offline_ts
from box_avg_fun import avg_over_box, draw_box


# Avoid having to restart the kernle if I modify my mapping scripts (or anything else)
import imp
import matplotlib.colors as mcolors

#%%

"""
    Do the preliminary import of masks, area fns, etc
"""

# area grid, lat/lon/lev:
area_f19, lat, lon, LT, LN, lev = get_coords()

# dictionary of seasons:
seasons = get_seasons()

# masks:
landfrac, landmask, ocnmask, bareground_mask, glc_mask, inv_glc_mask = get_masks()

#%%

"""
    fetch & load the global_pert data
"""
ds_cam = {}
ds_clm = {} 
sims = {} 
ds0_cam = {} 
ds0_clm = {}

ds_cam_ts = {}
ds_clm_ts = {} 
sims_ts = {} 
ds0_cam_ts = {} 
ds0_clm_ts = {}

# MEANS

# online runs:
ds_cam['online'], ds_clm['online'], sims['online'], ds0_cam['online'], ds0_clm['online'] = get_online(alb=1, rs=1, hc=1)

# offline runs:
ds_clm['offline'], sims['offline'], ds0_clm['offline'] = get_offline(alb=1, rs=1, hc=0)


# TIME SERIES
varlist_lnd = ['MML_ts']
varlist_atm = ['TREFHT','SHFLX','LHFLX','FSNS','FSNSC','FLNS',
               'FLNSC','FSNT','FSNTC','FLNT','FLNTC',
               'CLDLOW','CLDMED','CLDHGH','CLDTOT']
# online runs:
ds_cam_ts['online'], ds_clm_ts['online'], sims_ts['online'], ds0_cam_ts['online'], ds0_clm_ts['online'] = get_online_ts(alb=1, rs=0, hc=0,
                                                                             varlist_atm=varlist_atm,varlist_lnd=varlist_lnd)

# offline runs:
ds_clm_ts['offline'], sims_ts['offline'], ds0_clm_ts['offline'] = get_offline_ts(alb=1, rs=0, hc=0,varlist_lnd=varlist_lnd)

#%%

"""
    Put data into numpy arrays 
"""

# ds_cam : list of cam xarray data sets, format ds_cam[prop][run]
# ds_clm : list of clm xarray data sets, format ds_clm[prop][run]
# sims : list of full simulation names; format sims['prop'] = ['sim1','sim2','sim3']
# slope_vars : dictionary: atm, then lnd - list of strings for variables to take the slope of. These must be standard output variables. Custom below.
# derived_vars : list of strings for custom derived variables to take the slope of, e.g. MSE. Have to hard code these in here.
# seasons : if present, loop over seasons. If not, just do annual mean.

# dictionaries where we'll store the arrays
atm_resp = {}
lnd_resp = {}

# Vairables to take slopes of (output vars)
slope_vars = {}
slope_vars['atm'] = ['TREFHT','U10','SHFLX','LHFLX','FSNS','FSNSC','FLNS',
          'FLNSC','FSDS','FSDSC','FLDS','PRECC','PRECL','PRECSC','PRECSL','FSNT','FSNTC','FLNT','FLNTC',
          'FLUT','FLDS','SWCF','LWCF','CLDLOW','CLDMED','CLDHGH','U','V','WSUB','VQ','Z3','VT','VU',]
slope_vars['lnd'] = ['MML_water','MML_ts','MML_shflx','MML_lhflx','MML_fsns',
          'MML_fsds','MML_fsr','MML_alb','MML_flns','RH','MML_lwdn','MML_lwup']   # note, RH is from atm - if running offline, shouldn't change?


# Variables to take slopes of that need to be derrived
derived_vars = ['PRECIP','MSE','MSEC','BOWEN','EVAPFRAC','sh_plus_lh']#,'ALBEDO','ALBEDOC']

## temporary small vars lists
#
#slope_vars['atm'] = ['TREFHT','SHFLX','LHFLX']
#slope_vars['lnd'] = ['MML_shflx','MML_lhflx']
## Variables to take slopes of that need to be derrived
#derived_vars = ['EVAPFRAC']#,'ALBEDO','ALBEDOC']


atm_resp['online'], lnd_resp['online'] = make_variable_arrays(ds_cam=ds_cam['online'], 
                                ds_clm=ds_clm['online'], sims=sims['online'], 
                                slope_vars=slope_vars, derived_vars=derived_vars,
                                seasons=seasons,on_or_off='online')
    
atm_resp['offline'], lnd_resp['offline'] = make_variable_arrays(ds_cam=None, 
                                ds_clm=ds_clm['offline'], sims=sims['offline'], 
                                slope_vars=slope_vars, derived_vars=derived_vars,
                                seasons=seasons,on_or_off='offline')

# Put slope vars atm back to their old value
slope_vars['atm'] = ['TREFHT','U10','SHFLX','LHFLX','FSNS','FSNSC','FLNS',
          'FLNSC','PRECC','PRECL','PRECSC','PRECSL','FSNT','FSNTC','FLNT','FLNTC',
          'FLUT','FLDS','SWCF','LWCF','CLDLOW','CLDMED','CLDHGH']
#%%
"""
    Calculate the slope for the slope_vars
"""

# haven't defined forcing values yet. Do so here.
forcing = {}
forcing['alb'] = np.array([0.3, 0.2, 0.1])
forcing['rs'] = np.array([30., 100., 200.])
#forcing['hc'] = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
forcing['hc'] = np.array([0.1, 0.5, 1.0, 2.0, 10.0, 20.0])

seas = ['ANN','DJF','MAM','JJA','SON']

# Empty dictionaries for slopes, intercepts, etc
slope = {}
slope_inv = {}
intercept = {}
r_value = {}
p_values = {}
std_dev = {}

datm_dlnd_scaled = {}
dlnd_datm_scaled = {}

# Online first:
onoff = 'online'

slope[onoff] = {}
slope_inv[onoff] = {}
intercept[onoff] = {}
r_value[onoff] = {}
p_values[onoff] = {}
std_dev[onoff] = {}

datm_dlnd_scaled[onoff] = {}
dlnd_datm_scaled[onoff] = {}

props = list(sims[onoff].keys())

# atm first:
slope[onoff]['atm'] = {}
slope_inv[onoff]['atm'] = {}
intercept[onoff]['atm'] = {}
r_value[onoff]['atm'] = {}
p_values[onoff]['atm'] = {}
std_dev[onoff]['atm'] = {}

datm_dlnd_scaled[onoff]['atm'] = {}
dlnd_datm_scaled[onoff]['atm'] = {}

# loop over atm slope vars and derived atm slope vars
all_atm_vars = slope_vars['atm'] + ['MSE','MSEC','PRECIP']
    
for var in all_atm_vars:
    
    slope[onoff]['atm'][var] = {} 
    slope_inv[onoff]['atm'][var] = {} 
    intercept[onoff]['atm'][var] = {}
    r_value[onoff]['atm'][var] = {}
    p_values[onoff]['atm'][var] = {}
    std_dev[onoff]['atm'][var] = {}
    
    datm_dlnd_scaled[onoff]['atm'][var] = {}
    dlnd_datm_scaled[onoff]['atm'][var] = {}
    
    for prop in props:
        
        slope[onoff]['atm'][var][prop] = {}
        slope_inv[onoff]['atm'][var][prop] = {}
        intercept[onoff]['atm'][var][prop] = {} 
        r_value[onoff]['atm'][var][prop] = {}
        p_values[onoff]['atm'][var][prop] = {}
        std_dev[onoff]['atm'][var][prop] = {}
        
        datm_dlnd_scaled[onoff]['atm'][var][prop] = {}
        dlnd_datm_scaled[onoff]['atm'][var][prop] = {}
            
        for sea in seas:
            
            forcing_data = forcing[prop]
            response_data = atm_resp[onoff][var][prop][sea]
            
            slp, slp_inv, intrcpt, rv, pv, sig = sensitivity_slope(forcing_data,response_data)
            
            slope[onoff]['atm'][var][prop][sea] = slp
            slope_inv[onoff]['atm'][var][prop][sea] = slp_inv
            intercept[onoff]['atm'][var][prop][sea] = intrcpt
            r_value[onoff]['atm'][var][prop][sea] = rv
            p_values[onoff]['atm'][var][prop][sea] = pv
            std_dev[onoff]['atm'][var][prop][sea] = sig
            
            dT_factor = 0.1
            dT_factor=1.
            if prop == 'alb':
                #scale_factor = 0.1  # want d atm / d 0.1 albedo
                scale_factor = -0.01
                datm_dlnd_scaled[onoff]['atm'][var][prop][sea] = slp*scale_factor
                # how much dlnd needed for 0.1 dleta T
                dlnd_datm_scaled[onoff]['atm'][var][prop][sea] = (slp**(-1))*(-1*dT_factor )
                
                dlnd_datm_scaled[onoff]['atm'][var][prop]['dT_factor'] = -dT_factor
                datm_dlnd_scaled[onoff]['atm'][var][prop]['scale_factor'] = scale_factor
            elif prop == 'rs':
                #scale_factor = 10   # datm / d 10 s/m
                scale_factor = 10.
                datm_dlnd_scaled[onoff]['atm'][var][prop][sea] = slp*scale_factor
                # how much dlnd needed for 0.1 dleta T
                dlnd_datm_scaled[onoff]['atm'][var][prop][sea] = (slp**(-1))*dT_factor 
                
                dlnd_datm_scaled[onoff]['atm'][var][prop]['dT_factor'] = dT_factor
                datm_dlnd_scaled[onoff]['atm'][var][prop]['scale_factor'] = scale_factor
            elif prop == 'hc':
                #scale_factor = 0.1   # datm / d 0.1 m
                scale_factor = 0.1
                datm_dlnd_scaled[onoff]['atm'][var][prop][sea] = slp*scale_factor
                # how much dlnd needed for 0.1 dleta T
                dlnd_datm_scaled[onoff]['atm'][var][prop][sea] = (slp**(-1))*dT_factor
                
                dlnd_datm_scaled[onoff]['atm'][var][prop]['dT_factor'] = dT_factor
                datm_dlnd_scaled[onoff]['atm'][var][prop]['scale_factor'] = scale_factor
                
                
# now lnd (still online):
slope[onoff]['lnd'] = {}
slope_inv[onoff]['lnd'] = {}
intercept[onoff]['lnd'] = {}
r_value[onoff]['lnd'] = {}
p_values[onoff]['lnd'] = {}
std_dev[onoff]['lnd'] = {}

dlnd_datm_scaled[onoff]['lnd'] = {}
datm_dlnd_scaled[onoff]['lnd'] = {}

# loop over atm slope vars and derived atm slope vars
all_lnd_vars = slope_vars['lnd'] + ['BOWEN','EVAPFRAC','sh_plus_lh']

for var in all_lnd_vars:
    
    slope[onoff]['lnd'][var] = {} 
    slope_inv[onoff]['lnd'][var] = {} 
    intercept[onoff]['lnd'][var] = {}
    r_value[onoff]['lnd'][var] = {}
    p_values[onoff]['lnd'][var] = {}
    std_dev[onoff]['lnd'][var] = {}
    
    dlnd_datm_scaled[onoff]['lnd'][var] = {}
    datm_dlnd_scaled[onoff]['lnd'][var] = {}
    
    for prop in props:
        
        slope[onoff]['lnd'][var][prop] = {}
        slope_inv[onoff]['lnd'][var][prop] = {}
        intercept[onoff]['lnd'][var][prop] = {} 
        r_value[onoff]['lnd'][var][prop] = {}
        p_values[onoff]['lnd'][var][prop] = {}
        std_dev[onoff]['lnd'][var][prop] = {}
        
        dlnd_datm_scaled[onoff]['lnd'][var][prop] = {}
        datm_dlnd_scaled[onoff]['lnd'][var][prop] = {}
        
            
        for sea in seas:
            
            forcing_data = forcing[prop]
            response_data = lnd_resp[onoff][var][prop][sea]
            response_data = np.where(np.isnan(response_data),0.0,response_data)
            #response_data = np.ma.masked_where(np.isnan(response_data),response_data)
            
            slp, slp_inv, intrcpt, rv, pv, sig = sensitivity_slope(forcing_data,response_data)
            
            slope[onoff]['lnd'][var][prop][sea] = slp
            slope_inv[onoff]['lnd'][var][prop][sea] = slp_inv
            intercept[onoff]['lnd'][var][prop][sea] = intrcpt
            r_value[onoff]['lnd'][var][prop][sea] = rv
            p_values[onoff]['lnd'][var][prop][sea] = pv
            std_dev[onoff]['lnd'][var][prop][sea] = sig


            #dT_factor = 0.1
            #dlnd_datm_scaled[onoff]['lnd'][var][prop][sea] = (slp**(-1))#*dT_factor
            #dlnd_datm_scaled[onoff]['lnd'][var][prop]['dT_factor'] = dT_factor
             #dT_factor = 0.1
            dT_factor=1.
            if prop == 'alb':
                #scale_factor = 0.1  # want d atm / d 0.1 albedo
                scale_factor = -0.01
                datm_dlnd_scaled[onoff]['lnd'][var][prop][sea] = slp*scale_factor
                # how much dlnd needed for 0.1 dleta T
                dlnd_datm_scaled[onoff]['lnd'][var][prop][sea] = (slp**(-1))*(-1*dT_factor )
                
                dlnd_datm_scaled[onoff]['lnd'][var][prop]['dT_factor'] = -dT_factor
                datm_dlnd_scaled[onoff]['lnd'][var][prop]['scale_factor'] = scale_factor
            elif prop == 'rs':
                #scale_factor = 10   # datm / d 10 s/m
                scale_factor = 10.
                datm_dlnd_scaled[onoff]['lnd'][var][prop][sea] = slp*scale_factor
                # how much dlnd needed for 0.1 dleta T
                dlnd_datm_scaled[onoff]['lnd'][var][prop][sea] = (slp**(-1))*dT_factor 
                
                dlnd_datm_scaled[onoff]['lnd'][var][prop]['dT_factor'] = dT_factor
                datm_dlnd_scaled[onoff]['lnd'][var][prop]['scale_factor'] = scale_factor
            elif prop == 'hc':
                #scale_factor = 0.1   # datm / d 0.1 m
                scale_factor = 0.1
                datm_dlnd_scaled[onoff]['lnd'][var][prop][sea] = slp*scale_factor
                # how much dlnd needed for 0.1 dleta T
                dlnd_datm_scaled[onoff]['lnd'][var][prop][sea] = (slp**(-1))*dT_factor
                
                dlnd_datm_scaled[onoff]['lnd'][var][prop]['dT_factor'] = dT_factor
                datm_dlnd_scaled[onoff]['lnd'][var][prop]['scale_factor'] = scale_factor    
# offline:

onoff = 'offline'

slope[onoff] = {}
slope_inv[onoff] = {}
intercept[onoff] = {}
r_value[onoff] = {}
p_values[onoff] = {}
std_dev[onoff] = {}

dlnd_datm_scaled[onoff] = {}
datm_dlnd_scaled[onoff] = {}


props = list(sims[onoff].keys())
            
# now lnd (still online):
slope[onoff]['lnd'] = {}
slope_inv[onoff]['lnd'] = {}
intercept[onoff]['lnd'] = {}
r_value[onoff]['lnd'] = {}
p_values[onoff]['lnd'] = {}
std_dev[onoff]['lnd'] = {}

dlnd_datm_scaled[onoff]['lnd'] = {}
datm_dlnd_scaled[onoff]['lnd'] = {}

for var in all_lnd_vars:
    
    slope[onoff]['lnd'][var] = {} 
    slope_inv[onoff]['lnd'][var] = {} 
    intercept[onoff]['lnd'][var] = {}
    r_value[onoff]['lnd'][var] = {}
    p_values[onoff]['lnd'][var] = {}
    std_dev[onoff]['lnd'][var] = {}
    
    dlnd_datm_scaled[onoff]['lnd'][var] = {}
    datm_dlnd_scaled[onoff]['lnd'][var] = {}
    
    for prop in props:
        
        slope[onoff]['lnd'][var][prop] = {}
        slope_inv[onoff]['lnd'][var][prop] = {}
        intercept[onoff]['lnd'][var][prop] = {} 
        r_value[onoff]['lnd'][var][prop] = {}
        p_values[onoff]['lnd'][var][prop] = {}
        std_dev[onoff]['lnd'][var][prop] = {}
            
        dlnd_datm_scaled[onoff]['lnd'][var][prop] = {}
        datm_dlnd_scaled[onoff]['lnd'][var][prop] = {}
        
        
        for sea in seas:
            
            forcing_data = forcing[prop]
            response_data = lnd_resp[onoff][var][prop][sea]
            response_data = np.where(np.isnan(response_data),0.0,response_data)
            #response_data = np.ma.masked_where(np.isnan(response_data),response_data)
            
            slp, slp_inv, intrcpt, rv, pv, sig = sensitivity_slope(forcing_data,response_data)
            
            slope[onoff]['lnd'][var][prop][sea] = slp
            slope_inv[onoff]['lnd'][var][prop][sea] = slp
            intercept[onoff]['lnd'][var][prop][sea] = intrcpt
            r_value[onoff]['lnd'][var][prop][sea] = rv
            p_values[onoff]['lnd'][var][prop][sea] = pv
            std_dev[onoff]['lnd'][var][prop][sea] = sig
                
            
            
            #dT_factor = 0.1
            dlnd_datm_scaled[onoff]['lnd'][var][prop][sea] = (slp**(-1))#*dT_factor 
            #dlnd_datm_scaled[onoff]['lnd'][var][prop]['dT_factor'] = dT_factor
            dT_factor=1.
            if prop == 'alb':
                #scale_factor = 0.1  # want d atm / d 0.1 albedo
                scale_factor = -0.1
                datm_dlnd_scaled[onoff]['lnd'][var][prop][sea] = slp*scale_factor
                # how much dlnd needed for 0.1 dleta T
                dlnd_datm_scaled[onoff]['lnd'][var][prop][sea] = (slp**(-1))*(-1*dT_factor )
                
                dlnd_datm_scaled[onoff]['lnd'][var][prop]['dT_factor'] = -dT_factor
                datm_dlnd_scaled[onoff]['lnd'][var][prop]['scale_factor'] = scale_factor
            elif prop == 'rs':
                #scale_factor = 10   # datm / d 10 s/m
                scale_factor = 10.
                datm_dlnd_scaled[onoff]['lnd'][var][prop][sea] = slp*scale_factor
                # how much dlnd needed for 0.1 dleta T
                dlnd_datm_scaled[onoff]['lnd'][var][prop][sea] = (slp**(-1))*dT_factor 
                
                dlnd_datm_scaled[onoff]['lnd'][var][prop]['dT_factor'] = dT_factor
                datm_dlnd_scaled[onoff]['lnd'][var][prop]['scale_factor'] = scale_factor
            elif prop == 'hc':
                #scale_factor = 0.1   # datm / d 0.1 m
                scale_factor = 0.1
                datm_dlnd_scaled[onoff]['lnd'][var][prop][sea] = slp*scale_factor
                # how much dlnd needed for 0.1 dleta T
                dlnd_datm_scaled[onoff]['lnd'][var][prop][sea] = (slp**(-1))*dT_factor
                
                dlnd_datm_scaled[onoff]['lnd'][var][prop]['dT_factor'] = dT_factor
                datm_dlnd_scaled[onoff]['lnd'][var][prop]['scale_factor'] = scale_factor 
                
#slope, intercept, r_value, p_value, std_dev =  sensitivity_slope(forcing,response)

#%% Write out slope dictionary because it takes so long to make
dict_path = '/home/disk/p/mlague/eos18/simple_land/scripts/python/analysis/global_pert/python_scripts/'

global_pert_data = {}
global_pert_data['slope'] = slope
global_pert_data['slope_inv'] = slope_inv
global_pert_data['intercept'] = intercept
global_pert_data['r_value'] = r_value
global_pert_data['p_values'] = p_values
global_pert_data['std_dev'] = std_dev
global_pert_data['datm_dlnd_scaled'] = datm_dlnd_scaled
global_pert_data['dlnd_datm_scaled'] = dlnd_datm_scaled
#global_pert_data['atm_resp'] = atm_resp
#global_pert_data['lnd_resp'] = lnd_resp

response_data = {}
response_data['atm'] = atm_resp
response_data['lnd'] = lnd_resp

pickle_file = dict_path + 'glbl_pert_dict_file_test_20180228_pickle'

with open(pickle_file, 'wb') as f:
    pickle.dump(global_pert_data, f)

pickle_file = dict_path + 'response_dict_file_test_20180228_pickle'

with open(pickle_file, 'wb') as f:
    pickle.dump(response_data, f)
#target = open(dict_path + 'slope_file_test_oct30.txt','a')
#target.write(str(slope))
#
#target = open(dict_path + 'glbl_pert_dict_file_test_oct30.txt','a')
#target.write(str(global_pert_data))



#def writing(self):
#        self.whip[p.name] = p.age, p.address, p.phone
#        target = open('deed.txt', 'a')
#        target.write(str(self.whip))
#        print self.whip
#
#    def reading(self):
#        self.whip = open('deed.txt', 'r').read()
#        name = raw_input("> ")
#        if name in self.whip:
#            print self.whip[name]
    
#%% Temporary analyze vector field changes in U x WSUB ... move to analysis script after re-running 
prop = 'alb'
sea = 'ANN'

var = 'U'
U = atm_resp['online'][var][prop][sea]

var = 'V'
V = atm_resp['online'][var][prop][sea]

var = 'WSUB'
WSUB = atm_resp['online'][var][prop][sea]

dU = U[0,:,:,:] - U[2,:,:,:]
dV = V[0,:,:,:] - V[2,:,:,:] 
dWSUB = WSUB[0,:,:,:] - WSUB[2,:,:,:] 

#%%
## Plot vector field of U, W along transect of equatorish and SA/EPac
lon_inds = range(85,132)
print(lon[lon_inds])
lat_ind = 45
print(lat[lat_ind])

dU_EPac = dU[:,lat_ind,lon_inds]
dW_EPac = dWSUB[:,lat_ind,lon_inds]

X, Z = np.meshgrid(lon[lon_inds],lev)

#plt.figure()
fig,ax = plt.subplots(1, 1, figsize=(6,6))

plt.title('U-W vectors, delta a1-a3,every other arrow, 100*w')

Q = plt.quiver(np.flipud(X[::2,::2]),np.flipud(Z[::2,::2]),
                         np.flipud(dU_EPac[::2,::2]),100*np.flipud(dW_EPac[::2,::2]),
                         units='width',pivot='mid')
qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$2 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.show()
#plt.close()

#%% Mean state
## Plot vector field of U, W along transect of equatorish and SA/EPac
lon_inds = range(85,132)
print(lon[lon_inds])
lat_ind = 45
print(lat[lat_ind])

U_mean = U[1,:,lat_ind,lon_inds]
W_mean = WSUB[1,:,lat_ind,lon_inds]

X, Z = np.meshgrid(lon[lon_inds],lev)

#plt.figure()
fig,ax = plt.subplots(1, 1, figsize=(6,6))

plt.title('U-W vectors, delta a1-a3,every other arrow, 100*w')

Q = plt.quiver(np.flipud(X[::,::]),np.flipud(Z[::,::]),
                         np.flipud(U_mean[::,::]),100*np.flipud(W_mean[::,::]),
                         units='width',pivot='mid')
qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$2 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.show()
#plt.close()


