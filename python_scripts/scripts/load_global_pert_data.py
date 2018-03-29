#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:43:05 2017

@author: mlague

Load the online & offline runs into a numpy array

"""

# In[]:

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

from mml_mapping_fun import mml_map, discrete_cmap
from custom_python_mml_cmap import make_colormap, mml_cmap


# Avoid having to restart the kernle if I modify my mapping scripts (or anything else)
import imp
import matplotlib.colors as mcolors



#%%
def make_variable_arrays(ds_cam=None, ds_clm=None, sims=None, slope_vars=None, derived_vars=None,seasons=None,on_or_off=None):
    #-----------------------------
    # Inputs:
    #
    # ds_cam : list of cam xarray data sets, format ds_cam[prop][run]
    # ds_clm : list of clm xarray data sets, format ds_clm[prop][run]
    # sims : list of full simulation names; format sims['prop'] = ['sim1','sim2','sim3']
    # slope_vars : dictionary: atm, then lnd - list of strings for variables to take the slope of. These must be standard output variables. Custom below.
    # derived_vars : list of strings for custom derived variables to take the slope of, e.g. MSE. Have to hard code these in here.
    # seasons : if present, loop over seasons. If not, just do annual mean.
    #
    # Returns:
    #
    # atm_resp: atmospheric response, as atm_resp[var][prop][sea]
    # lnd_resp: atmospheric response, as lnd_resp[var][prop][sea]
    #
    #
    # Turn xarray datasets into np.array's with the low, medium, high experiments 
    # from sims
    #
    #-----------------------------
    
    sfc_props = list(sims.keys())
    #sfc_props = list(sims['online'].keys())
#    if on_or_off == 'online':
#        #sim_names = list(sims['online'].keys())
#        sim_names = list(sims.keys())
#    elif on_or_off =='offline':
#        #sim_names = list(sims['offline'].keys())
#        sim_names = list(sims.keys())
    
    atm_vars = slope_vars['atm']
    lnd_vars = slope_vars['lnd']
    
    atm_resp = {}
    lnd_resp = {}
    
    if ds_cam:
        for var in atm_vars:
            
            atm_resp[var] = {}
            
            for prop in sfc_props:
               
               atm_resp[var][prop] = {}
                
               sim_names = list(sims[prop])
#               if on_or_off == 'online':
#                   sim_names = list(sims['online'][prop])
#               elif on_or_off =='offline':
#                   sim_names = list(sims['offline'][prop])
               
               k = np.shape(sim_names)[0]
               #print('MML hey - k = ___')
               #print(k)
               
               # should make this flexible; for now hard coding in
               if k == 3:
                   #print('in k = 3')
                   # 3 experiments x 12 months x 96 lats x 144 lons
                   atm_resp[var][prop]['12months'] = np.array( [ ds_cam[prop][sim_names[0]][var].values[:], 
                                                                 ds_cam[prop][sim_names[1]][var].values[:], 
                                                                 ds_cam[prop][sim_names[2]][var].values[:] ] ) 
                   atm_resp[var][prop]['units'] = ds_cam[prop][sim_names[0]][var].units
                   
               elif k == 6:
                   #print('in k = 6')
                   atm_resp[var][prop]['12months'] = np.array( [ ds_cam[prop][sim_names[0]][var].values[:], 
                                                                 ds_cam[prop][sim_names[1]][var].values[:],
                                                                 ds_cam[prop][sim_names[2]][var].values[:],
                                                                 ds_cam[prop][sim_names[3]][var].values[:],
                                                                 ds_cam[prop][sim_names[4]][var].values[:],
                                                                 ds_cam[prop][sim_names[5]][var].values[:]] )
                   atm_resp[var][prop]['units'] = ds_cam[prop][sim_names[0]][var].units
    
                   
               for sea in list(seasons['names']):
                   sea_ind = seasons['indices'][sea]
                   
                   #temp = atm_resp[var][prop]['12months'].values[:]
                   
                   atm_resp[var][prop][sea] = np.mean(atm_resp[var][prop]['12months'][:,sea_ind,:,:],1)
                   #atm_resp[var][prop][sea] = np.mean(temp[:,sea_ind,:,:],1)
           
           
           
        
        
    if ds_clm:
        for var in lnd_vars:
            lnd_resp[var] = {}
            
            for prop in sfc_props:
               
               lnd_resp[var][prop] = {}
               
               sim_names = list(sims[prop])
#               if on_or_off == 'online':
#                   sim_names = list(sims['online'][prop])
#               elif on_or_off =='offline':
#                   sim_names = list(sims['offline'][prop])
               
               k = np.shape(sim_names)[0]
               
               # should make this flexible; for now hard coding in
               if k == 3:
                   #print('in k = 3, lnd')
                   # 3 experiments x 12 months x 96 lats x 144 lons
                   lnd_resp[var][prop]['12months'] = np.array( [ ds_clm[prop][sim_names[0]][var].values[:], 
                                                                 ds_clm[prop][sim_names[1]][var].values[:], 
                                                                 ds_clm[prop][sim_names[2]][var].values[:] ] ) 
                   lnd_resp[var][prop]['units'] = ds_clm[prop][sim_names[0]][var].units
        
               elif k == 6:
                   #print('in k = 6, lnd')
                   lnd_resp[var][prop]['12months'] = np.array( [ ds_clm[prop][sim_names[0]][var].values[:], 
                                                                 ds_clm[prop][sim_names[1]][var].values[:],
                                                                 ds_clm[prop][sim_names[2]][var].values[:],
                                                                 ds_clm[prop][sim_names[3]][var].values[:],
                                                                 ds_clm[prop][sim_names[4]][var].values[:],
                                                                 ds_clm[prop][sim_names[5]][var].values[:]] ) 
                   lnd_resp[var][prop]['units'] = ds_clm[prop][sim_names[0]][var].units
                   
               for sea in list(seasons['names']):
                   sea_ind = seasons['indices'][sea]
                   
                   lnd_resp[var][prop][sea] = np.mean(lnd_resp[var][prop]['12months'][:,sea_ind,:,:],1)
        
    
    # if derived vars are requested, calculate them:
    if derived_vars:
        
        # initially assume neither 
        is_atm = 0
        is_lnd = 0
        
        # Loop over derived vars:
        for dvar in derived_vars:
            
            # these come both from atm and land... if I were
            # being rigorous, I would check that I have ds_cam and ds_clm for them...
            
            # initially assume neither 
            is_atm = 0
            is_lnd = 0
            
            # Calculate the field on the 12 month array
            if ds_cam:
                if dvar=='MSE':
                    # Moist Static Energy; averaged over long periods, can just do this from TOA - SFC energy sources to column.
                    is_atm = 1
                    is_lnd = 0
                    atm_resp[dvar] = {}
                    
                    for prop in sfc_props:
                        atm_resp[dvar][prop] = {}
                        #print('MSE')
                        
                        # MSE calculation:
                        atm_resp[dvar][prop]['12months'] =  np.array( (atm_resp['FSNT'][prop]['12months'] - atm_resp['FLNT'][prop]['12months']) - 
                                            ( (atm_resp['FLNS'][prop]['12months'] ) + atm_resp['SHFLX'][prop]['12months'] + atm_resp['LHFLX'][prop]['12months']) )
                        atm_resp[dvar][prop]['units'] = 'W/m2'
                        
                        for sea in list(seasons['names']):
                           # print(sea)
                            sea_ind = seasons['indices'][sea]
                            atm_resp[dvar][prop][sea] = np.mean(atm_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
                        
                elif dvar=='MSEC':
                    # Moist Static Energy from clearsky fluxes; averaged over long periods, can just do this from TOA - SFC energy sources to column.
                    is_atm = 1
                    is_lnd = 0
                    
                    atm_resp[dvar] = {}
                    
                    for prop in sfc_props:
                        atm_resp[dvar][prop] = {}
                        
                        # MSE calculation:
                        atm_resp[dvar][prop]['12months'] =  np.array( (atm_resp['FSNTC'][prop]['12months'] - atm_resp['FLNTC'][prop]['12months']) - 
                                            ( (atm_resp['FLNSC'][prop]['12months'] ) + atm_resp['SHFLX'][prop]['12months'] + atm_resp['LHFLX'][prop]['12months']) )
                        atm_resp[dvar][prop]['units'] = 'W/m2'
                        
                        for sea in list(seasons['names']):
                            sea_ind = seasons['indices'][sea]
                            atm_resp[dvar][prop][sea] = np.mean(atm_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
                            
                elif dvar=='PRECIP':
                    # Moist Static Energy from clearsky fluxes; averaged over long periods, can just do this from TOA - SFC energy sources to column.
                    is_atm = 1
                    is_lnd = 0
                    
                    ms2mmday = 60*60*14*1000
                    
                    atm_resp[dvar] = {}
                    
                    for prop in sfc_props:
                        atm_resp[dvar][prop] = {}
                        
                        
                        # MSE calculation:
                        atm_resp[dvar][prop]['12months'] =  np.array( atm_resp['PRECC'][prop]['12months'] + 
                                                                    atm_resp['PRECL'][prop]['12months'] +
                                                                    atm_resp['PRECSC'][prop]['12months'] +
                                                                    atm_resp['PRECSL'][prop]['12months'] 
                                                                    ) * ms2mmday
                        atm_resp[dvar][prop]['units'] = 'mm/day'
                        
                        for sea in list(seasons['names']):
                            sea_ind = seasons['indices'][sea]
                            atm_resp[dvar][prop][sea] = np.mean(atm_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
                            
                elif dvar=='SW_CLOUD':
                    # FSNS - FSNSC = energy not reaching surface because of clouds
                    is_atm = 1
                    is_lnd = 0
                    
                    atm_resp[dvar] = {}
                    
                    for prop in sfc_props:
                        atm_resp[dvar][prop] = {}
                        
                        # MSE calculation:
                        atm_resp[dvar][prop]['12months'] =  np.array( atm_resp['FSNS'][prop]['12months'] - 
                                                                    atm_resp['FSNSC'][prop]['12months'] 
                                                                    )
                        atm_resp[dvar][prop]['units'] = 'W/m2'
                        
                        for sea in list(seasons['names']):
                            sea_ind = seasons['indices'][sea]
                            atm_resp[dvar][prop][sea] = np.mean(atm_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
                
                # make seasonal means
                #print('hi?')
#                for sea in list(seasons['names']):
#                    sea_ind = seasons['indices'][sea]
#                    #print('hey...')
#                    if is_atm == 1:
#                        atm_resp[dvar][prop][sea] = np.mean(atm_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
            
            if ds_clm:
                if dvar=='BOWEN':
                    # Bowen ratio = SH / LH
                    is_atm = 0
                    is_lnd = 1
                    
                    lnd_resp[dvar] = {}
                    
                    for prop in sfc_props:
                        lnd_resp[dvar][prop] = {}
                        
                        
                        # MSE calculation:
                        lnd_resp[dvar][prop]['12months'] =  np.array( lnd_resp['MML_shflx'][prop]['12months'] /
                                                                    lnd_resp['MML_lhflx'][prop]['12months'] 
                                                                    )
                        lnd_resp[dvar][prop]['units'] = 'mm/day'
                        
                        for sea in list(seasons['names']):
                            sea_ind = seasons['indices'][sea]
                            lnd_resp[dvar][prop][sea] = np.mean(lnd_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
                    
                elif dvar=='EVAPFRAC':
                    # Evaporative fraction = LH / (SH + LH)
                    is_atm = 0
                    is_lnd = 1
                    
                    lnd_resp[dvar] = {}
                    
                    for prop in sfc_props:
                        lnd_resp[dvar][prop] = {}
                        
                        
                        #  calculation:
                        lnd_resp[dvar][prop]['12months'] =  np.array( lnd_resp['MML_lhflx'][prop]['12months'] /
                                                                    (lnd_resp['MML_lhflx'][prop]['12months'] + lnd_resp['MML_shflx'][prop]['12months'])
                                                                    )
                        lnd_resp[dvar][prop]['units'] = 'mm/day'
                        
                        for sea in list(seasons['names']):
                            sea_ind = seasons['indices'][sea]
                            lnd_resp[dvar][prop][sea] = np.mean(lnd_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
                    
                elif dvar=='ALBEDO':
                    # albedo at surface (will use lnd vars...)
                    is_atm = 0
                    is_lnd = 1
                    
                    # temporary no calculation:
                    for prop in sfc_props:
                        lnd_resp[dvar][prop] = {}
                        
                        
                        #  calculation:
                        lnd_resp[dvar][prop]['12months'] =  0.*np.array( lnd_resp['MML_lhflx'][prop]['12months'] )
                        lnd_resp[dvar][prop]['units'] = 'NOT YET CALCULATED'
                        
                        for sea in list(seasons['names']):
                            sea_ind = seasons['indices'][sea]
                            lnd_resp[dvar][prop][sea] = np.mean(lnd_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
                    
                elif dvar=='ALBEDOC':
                    # clearsky albedo at surface (will use lnd vars...)
                    is_atm = 0
                    is_lnd = 1
                    
                    # temporary no calculation:
                    for prop in sfc_props:
                        lnd_resp[dvar][prop] = {}
                        
                        
                        #  calculation:
                        lnd_resp[dvar][prop]['12months'] =  0.*np.array( lnd_resp['MML_lhflx'][prop]['12months'] )
                        lnd_resp[dvar][prop]['units'] = 'NOT YET CALCULATED'
                        
                        for sea in list(seasons['names']):
                            sea_ind = seasons['indices'][sea]
                            lnd_resp[dvar][prop][sea] = np.mean(lnd_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
                elif dvar =='sh_plus_lh':
                    # MML_shflx + MML_lhflx, total turbulent heat flux
                    is_atm = 0
                    is_lnd = 1
                    
                    lnd_resp[dvar] = {}
                    
                    for prop in sfc_props:
                        lnd_resp[dvar][prop] = {}
                        
                        
                        #  calculation:
                        lnd_resp[dvar][prop]['12months'] =  np.array( lnd_resp['MML_lhflx'][prop]['12months'] +
                                                                        lnd_resp['MML_shflx'][prop]['12months'])
                                                                    
                        lnd_resp[dvar][prop]['units'] = 'W/m2'
                        
                        for sea in list(seasons['names']):
                            sea_ind = seasons['indices'][sea]
                            lnd_resp[dvar][prop][sea] = np.mean(lnd_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
                            
                            
                # make seasonal means
                #print('hi?')
#                for sea in list(seasons['names']):
#                    sea_ind = seasons['indices'][sea]
#                    #print('hey...')
#                    if is_atm == 1:
#                        atm_resp[dvar][prop][sea] = np.mean(atm_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
#                
#                    elif is_lnd == 1:
#                        lnd_resp[dvar][prop][sea] = np.mean(lnd_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)   
                    
                
            
                    
            # make seasonal means
            #print('hi?')
#            for sea in list(seasons['names']):
#                    sea_ind = seasons['indices'][sea]
#                   # print('hey...')
#                    if is_atm == 1:
#                        print(sea + ' ' + dvar)
#                        atm_resp[dvar][prop][sea] = np.mean(atm_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
#                
#                    elif is_lnd == 1:
#                        lnd_resp[dvar][prop][sea] = np.mean(lnd_resp[dvar][prop]['12months'][:,sea_ind,:,:],1)
                        
    
    return atm_resp, lnd_resp


#%%

def get_online(alb=None, rs=None, hc=None,varlist_atm=None,varlist_lnd=None):
    #-----------------------------
    # Inputs:
    #
    # alb : if alb==True, load albedo runs
    # rs : if rs==True, load evaporative resistance runs
    # hc : if hc==True, load roughenss runs
    #
    # Returns:
    #
    # ds_cam: dictionary of xarray netcdfs for cam output as [prop][run]
    # ds_clm: dictionary of xarray netcdfs for clm output as [prop][run]
    # sims: dictionary of lookup names for runs
    # ds0_cam
    # ds0_clm
    #
    # old plan of return:
    #
    # alb_atm : response of atmosphere to albedo runs (in 3 x 12 x lat x lon array)
    # alb_lnd : response of land to albedo runs (in 3 x 12 x lat x lon array)
    #
    # rs_atm : response of atmosphere to evap rs runs (in 3 x 12 x lat x lon array)
    # rs_lnd : response of land to evap rs runs (in 3 x 12 x lat x lon array)
    #
    # hc_atm : response of atmosphere to roughness runs (in 3 x 12 x lat x lon array)
    # hc_lnd : response of land to roughness runs (in 3 x 12 x lat x lon array)
    #
    # ds0_cam : middle-of-the-road xarray dataset for cam
    # ds0_clm : middle-of-the-road xarray dataset for clm
    #
    #-----------------------------
    
    #-----------------------------
    # Define filepaths
    #-----------------------------
    
    ext_dir = '/home/disk/eos18/mlague/simple_land/output/global_pert_cheyenne/'
    
    
    sims = {}
    sims['alb'] = [ 'global_a3_cv2_hc0.1_rs100_cheyenne',
                    'global_a2_cv2_hc0.1_rs100_cheyenne',
                    'global_a1_cv2_hc0.1_rs100_cheyenne' ]
    sims['rs'] =  [ 'global_a2_cv2_hc0.1_rs30_cheyenne',
                    'global_a2_cv2_hc0.1_rs100_cheyenne',
                    'global_a2_cv2_hc0.1_rs200_cheyenne' ]
#    sims['hc'] =  [ 'global_a2_cv2_hc0.01_rs100_cheyenne',
#                    'global_a2_cv2_hc0.05_rs100_cheyenne',
#                    'global_a2_cv2_hc0.1_rs100_cheyenne',
#                    'global_a2_cv2_hc0.5_rs100_cheyenne',
#                    'global_a2_cv2_hc1.0_rs100_cheyenne',
#                    'global_a2_cv2_hc2.0_rs100_cheyenne' ]
    sims['hc'] =  [ 'global_a2_cv2_hc0.1_rs100_cheyenne',
                    'global_a2_cv2_hc0.5_rs100_cheyenne',
                    'global_a2_cv2_hc1.0_rs100_cheyenne',
                    'global_a2_cv2_hc2.0_rs100_cheyenne',
                    'global_a2_cv2_hc10.0_rs100_cheyenne',
                    'global_a2_cv2_hc20.0_rs100_cheyenne' ]
 
    #print(sims)
    
    run0 = 'global_a2_cv2_hc0.1_rs100_cheyenne'
    ds0_cam = xr.open_dataset( ext_dir + run0 + '/means/' + run0 + '.cam.h0.20-50_year_avg.nc')
    ds0_clm = xr.open_dataset(ext_dir + run0 + '/means/' + run0 + '.clm2.h0.20-50_year_avg.nc')
    
    ds_cam = {}
    ds_clm = {}
    
    
    if alb:
        prop = 'alb'
        
        ds_cam[prop] = {}
        ds_clm[prop] = {}
        
        for run in sims[prop]:
                ds_cam[prop][run] = xr.open_dataset(ext_dir + run + '/means/' + run + '.cam.h0.20-50_year_avg.nc')
                ds_clm[prop][run] = xr.open_dataset(ext_dir + run + '/means/' + run + '.clm2.h0.20-50_year_avg.nc')
        
        
    if rs:
        prop = 'rs'
        
        ds_cam[prop] = {}
        ds_clm[prop] = {}
        
        for run in sims[prop]:
                ds_cam[prop][run] = xr.open_dataset(ext_dir + run + '/means/' + run + '.cam.h0.20-50_year_avg.nc')
                ds_clm[prop][run] = xr.open_dataset(ext_dir + run + '/means/' + run + '.clm2.h0.20-50_year_avg.nc')
    
    if hc:
        prop = 'hc'
        
        ds_cam[prop] = {}
        ds_clm[prop] = {}
        
        for run in sims[prop]:
                ds_cam[prop][run] = xr.open_dataset(ext_dir + run + '/means/' + run + '.cam.h0.20-50_year_avg.nc')
                ds_clm[prop][run] = xr.open_dataset(ext_dir + run + '/means/' + run + '.clm2.h0.20-50_year_avg.nc')
  
    #print('HEY!!!!')
    #print(ds_cam)
    
    return ds_cam, ds_clm, sims, ds0_cam, ds0_clm
#%%

def get_online_ts(alb=None, rs=None, hc=None,varlist_atm=None,varlist_lnd=None):
    #-----------------------------
    # Inputs:
    #
    # alb : if alb==True, load albedo runs
    # rs : if rs==True, load evaporative resistance runs
    # hc : if hc==True, load roughenss runs
    #
    # Returns:
    #
    # ds_cam: dictionary of xarray netcdfs for cam output as [prop][run]
    # ds_clm: dictionary of xarray netcdfs for clm output as [prop][run]
    # sims: dictionary of lookup names for runs
    # ds0_cam
    # ds0_clm
    #
    # old plan of return:
    #
    # alb_atm : response of atmosphere to albedo runs (in 3 x 12 x lat x lon array)
    # alb_lnd : response of land to albedo runs (in 3 x 12 x lat x lon array)
    #
    # rs_atm : response of atmosphere to evap rs runs (in 3 x 12 x lat x lon array)
    # rs_lnd : response of land to evap rs runs (in 3 x 12 x lat x lon array)
    #
    # hc_atm : response of atmosphere to roughness runs (in 3 x 12 x lat x lon array)
    # hc_lnd : response of land to roughness runs (in 3 x 12 x lat x lon array)
    #
    # ds0_cam : middle-of-the-road xarray dataset for cam
    # ds0_clm : middle-of-the-road xarray dataset for clm
    #
    #-----------------------------
    
    #-----------------------------
    # Define filepaths
    #-----------------------------
    
    ext_dir = '/home/disk/eos18/mlague/simple_land/output/global_pert_cheyenne/'
    
    
    sims = {}
    sims['alb'] = [ 'global_a3_cv2_hc0.1_rs100_cheyenne',
                    'global_a2_cv2_hc0.1_rs100_cheyenne',
                    'global_a1_cv2_hc0.1_rs100_cheyenne' ]
    sims['rs'] =  [ 'global_a2_cv2_hc0.1_rs30_cheyenne',
                    'global_a2_cv2_hc0.1_rs100_cheyenne',
                    'global_a2_cv2_hc0.1_rs200_cheyenne' ]
#    sims['hc'] =  [ 'global_a2_cv2_hc0.01_rs100_cheyenne',
#                    'global_a2_cv2_hc0.05_rs100_cheyenne',
#                    'global_a2_cv2_hc0.1_rs100_cheyenne',
#                    'global_a2_cv2_hc0.5_rs100_cheyenne',
#                    'global_a2_cv2_hc1.0_rs100_cheyenne',
#                    'global_a2_cv2_hc2.0_rs100_cheyenne' ]
    sims['hc'] =  [ 'global_a2_cv2_hc0.1_rs100_cheyenne',
                    'global_a2_cv2_hc0.5_rs100_cheyenne',
                    'global_a2_cv2_hc1.0_rs100_cheyenne',
                    'global_a2_cv2_hc2.0_rs100_cheyenne',
                    'global_a2_cv2_hc10.0_rs100_cheyenne',
                    'global_a2_cv2_hc20.0_rs100_cheyenne' ]
 
    #print(sims)
    
    run0 = 'global_a2_cv2_hc0.1_rs100_cheyenne'
    ds0_cam = xr.open_dataset( ext_dir + run0 + '/means/' + run0 + '.cam.h0.20-50_year_avg.nc')
    ds0_clm = xr.open_dataset(ext_dir + run0 + '/means/' + run0 + '.clm2.h0.20-50_year_avg.nc')
    
    ds_cam_ts = {}
    ds_clm_ts = {}
    
    
    if alb:
        prop = 'alb'
        
        ds_cam_ts[prop] = {}
        ds_clm_ts[prop] = {}
        
        for run in sims[prop]:
            for var in varlist_atm:
                ds_cam_ts[prop][run] = {}
                print((ext_dir + run + '/TimeSeries/' + run + '.cam.h0.20-50.' + var + '.nc'))
                ds_cam_ts[prop][run][var] = xr.open_dataset(ext_dir + run + '/TimeSeries/' + run + '.cam.h0.ts.20-50.' + var + '.nc')
            for var in varlist_lnd:
                ds_clm_ts[prop][run] = {}
                ds_clm_ts[prop][run][var] = xr.open_dataset(ext_dir + run + '/TimeSeries/' + run + '.clm2.h0.ts.20-50.' + var + '.nc')
        
        
    if rs:
        prop = 'rs'
        
        ds_cam_ts[prop] = {}
        ds_clm_ts[prop] = {}
        
        for run in sims[prop]:
            for var in varlist_atm:
                ds_cam_ts[prop][run] = {}
                ds_cam_ts[prop][run][var] = xr.open_dataset(ext_dir + run + '/TimeSeries/' + run + '.cam.h0.ts.20-50.' + var + '.nc')
            for var in varlist_lnd:
                ds_clm_ts[prop][run] = {}
                ds_clm_ts[prop][run][var] = xr.open_dataset(ext_dir + run + '/TimeSeries/' + run + '.clm2.h0.ts.20-50.' + var + '.nc')
    
    if hc:
        prop = 'hc'
        
        ds_cam_ts[prop] = {}
        ds_clm_ts[prop] = {}
        
        for run in sims[prop]:
            for var in varlist_atm:
                ds_cam_ts[prop][run] = {}
                ds_cam_ts[prop][run][var] = xr.open_dataset(ext_dir + run + '/TimeSeries/' + run + '.cam.h0.ts.20-50.' + var + '.nc')
            for var in varlist_lnd:
                ds_clm_ts[prop][run] = {}
                ds_clm_ts[prop][run][var] = xr.open_dataset(ext_dir + run + '/TimeSeries/' + run + '.clm2.h0.ts.20-50.' + var + '.nc')
  
    #print('HEY!!!!')
    #print(ds_cam)
    
    return ds_cam_ts, ds_clm_ts, sims, ds0_cam, ds0_clm
  
#%%
def get_offline(alb=None, rs=None, hc=None):
    #-----------------------------
    # Inputs:
    #
    # alb : if alb==True, load albedo runs
    # rs : if rs==True, load evaporative resistance runs
    # hc : if hc==True, load roughenss runs
    #
    # Returns:
    #
    # ds_cam: dictionary of xarray netcdfs for cam output as [prop][run]
    # ds_clm: dictionary of xarray netcdfs for clm output as [prop][run]
    # sims: dictionary of lookup names for runs
    # ds0_cam
    # ds0_clm
    #
    # old plan of return:
    #
    # alb_atm : response of atmosphere to albedo runs (in 3 x 12 x lat x lon array)
    # alb_lnd : response of land to albedo runs (in 3 x 12 x lat x lon array)
    #
    # rs_atm : response of atmosphere to evap rs runs (in 3 x 12 x lat x lon array)
    # rs_lnd : response of land to evap rs runs (in 3 x 12 x lat x lon array)
    #
    # hc_atm : response of atmosphere to roughness runs (in 3 x 12 x lat x lon array)
    # hc_lnd : response of land to roughness runs (in 3 x 12 x lat x lon array)
    #
    # ds0_cam : middle-of-the-road xarray dataset for cam
    # ds0_clm : middle-of-the-road xarray dataset for clm
    #
    #-----------------------------
    
    #-----------------------------
    # Define filepaths
    #-----------------------------
    
    ext_dir = '/home/disk/eos18/mlague/simple_land/output/global_pert_offline_MML/'
    
    
    sims = {}
    sims['alb'] = [ 'global_a3_cv2_hc0.1_rs100_offline_b07',
                    'global_a2_cv2_hc0.1_rs100_offline_b07',
                    'global_a1_cv2_hc0.1_rs100_offline_b07' ]
    sims['rs'] = [ 'global_a2_cv2_hc0.1_rs30_offline_b07',
                    'global_a2_cv2_hc0.1_rs100_offline_b07',
                    'global_a2_cv2_hc0.1_rs200_offline_b07' ]
    #sims['rs'] =  [   ]
    #sims['hc'] =  [   ]
 
    
    run0 = 'global_a2_cv2_hc0.1_rs100_offline_b07'
    ds0_clm = xr.open_dataset(ext_dir + run0 + '/means/' + run0 + '.clm2.h0.20-50_year_avg.nc')
    
    ds_clm = {}
    
    if alb:
        prop = 'alb'
        
        ds_clm[prop] = {}
        
        for run in sims[prop]:
            ds_clm[prop][run] = xr.open_dataset(ext_dir + run + '/means/' + run + '.clm2.h0.20-50_year_avg.nc')
        
        
    if rs:
        prop = 'rs'
        
        ds_clm[prop] = {}
        
        for run in sims[prop]:
            ds_clm[prop][run] = xr.open_dataset(ext_dir + run + '/means/' + run + '.clm2.h0.20-50_year_avg.nc')
    
    if hc:
        prop = 'hc'
 
        ds_clm[prop] = {}
        
        for run in sims[prop]:
            ds_clm[prop][run] = xr.open_dataset(ext_dir + run + '/means/' + run + '.clm2.h0.20-50_year_avg.nc')
  
    
    
    return ds_clm, sims, ds0_clm


#%%
def get_offline_ts(alb=None, rs=None, hc=None,varlist_lnd=None):
    #-----------------------------
    # Inputs:
    #
    # alb : if alb==True, load albedo runs
    # rs : if rs==True, load evaporative resistance runs
    # hc : if hc==True, load roughenss runs
    #
    # Returns:
    #
    # ds_cam: dictionary of xarray netcdfs for cam output as [prop][run]
    # ds_clm: dictionary of xarray netcdfs for clm output as [prop][run]
    # sims: dictionary of lookup names for runs
    # ds0_cam
    # ds0_clm
    #
    # old plan of return:
    #
    # alb_atm : response of atmosphere to albedo runs (in 3 x 12 x lat x lon array)
    # alb_lnd : response of land to albedo runs (in 3 x 12 x lat x lon array)
    #
    # rs_atm : response of atmosphere to evap rs runs (in 3 x 12 x lat x lon array)
    # rs_lnd : response of land to evap rs runs (in 3 x 12 x lat x lon array)
    #
    # hc_atm : response of atmosphere to roughness runs (in 3 x 12 x lat x lon array)
    # hc_lnd : response of land to roughness runs (in 3 x 12 x lat x lon array)
    #
    # ds0_cam : middle-of-the-road xarray dataset for cam
    # ds0_clm : middle-of-the-road xarray dataset for clm
    #
    #-----------------------------
    
    #-----------------------------
    # Define filepaths
    #-----------------------------
    
    ext_dir = '/home/disk/eos18/mlague/simple_land/output/global_pert_offline_MML/'
    
    
    sims = {}
    sims['alb'] = [ 'global_a3_cv2_hc0.1_rs100_offline_b07',
                    'global_a2_cv2_hc0.1_rs100_offline_b07',
                    'global_a1_cv2_hc0.1_rs100_offline_b07' ]
    sims['rs'] = [ 'global_a2_cv2_hc0.1_rs30_offline_b07',
                    'global_a2_cv2_hc0.1_rs100_offline_b07',
                    'global_a2_cv2_hc0.1_rs200_offline_b07' ]
    #sims['rs'] =  [   ]
    #sims['hc'] =  [   ]
 
    
    run0 = 'global_a2_cv2_hc0.1_rs100_offline_b07'
    ds0_clm = xr.open_dataset(ext_dir + run0 + '/means/' + run0 + '.clm2.h0.20-50_year_avg.nc')
    
    ds_clm_ts = {}
    
    if alb:
        prop = 'alb'
        
        ds_clm_ts[prop] = {}
        
        for run in sims[prop]:
            for var in varlist_lnd:
                ds_clm_ts[prop][run] = {}
                ds_clm_ts[prop][run][var] = xr.open_dataset(ext_dir + run + '/TimeSeries/' + run + '.clm2.h0.ts.20-50.' + var + '.nc')
        
        
    if rs:
        prop = 'rs'
        
        ds_clm_ts[prop] = {}
        
        for run in sims[prop]:
            for var in varlist_lnd:
                ds_clm_ts[prop][run] = {}
                ds_clm_ts[prop][run][var] = xr.open_dataset(ext_dir + run + '/TimeSeries/' + run + '.clm2.h0.ts.20-50.' + var + '.nc')
    
    if hc:
        prop = 'hc'
 
        ds_clm_ts[prop] = {}
        
        for run in sims[prop]:
            for var in varlist_lnd:
                ds_clm_ts[prop][run] = {}
                ds_clm_ts[prop][run][var] = xr.open_dataset(ext_dir + run + '/TimeSeries/' + run + '.clm2.h0.ts.20-50.' + var + '.nc')
  
    
    
    return ds_clm_ts, sims, ds0_clm


