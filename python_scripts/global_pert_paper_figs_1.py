#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:04:42 2018

@author: mlague

Paper-specific (first pass at) figures for global perturbation write-up


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
from load_global_pert_data import make_variable_arrays, get_online, get_offline
from box_avg_fun import avg_over_box, draw_box


# Avoid having to restart the kernle if I modify my mapping scripts (or anything else)
import imp
import matplotlib.colors as mcolors


#%%


def mml_map_local(mapdata_raw=None,
                     units=None,prop=None,sea=None,mask=None,maskname=None, 
                     LN=None, LT=None,
                     cb=None,climits=None,save_subplot=None,
                     figpath=None,scale=None,filename=None,ttl=None):
    # need to have already opened a figure/axis
    #plt.sca(ax)
    
    from mpl_toolkits.basemap import Basemap, cm
    
    
    
    if cb:
        cb = cb
    else:
        cb = plt.cm.viridis
          
    if ttl:
        ttl_main = ttl
    else:
        ttl_main = 'insert title here'
    
    #mask_name = np.str(mask)
    
        
    #filename = 'global_datmdlnd'+var+'_'+prop+'_'+maskname+'_'+sea+'_'+onoff
                
    fig, axes = plt.subplots(1, 1, figsize=(5,4))
    
    
    #------------------------------------------
    # left plot: dlnd/datm
    #------------------------------------------
    
    ax0 = plt.gca()
    
    mapdata = mapdata_raw
    
    mapdata = mapdata*mask
    #if mask:
    #    mapdata = mapdata*mask
    
    mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
    
    
    plt.sca(ax0)
   
    
    cm = cb
    clim = climits
    
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=None,proj='moll',title=ttl_main,clim=clim,colmap=cm, cb_ttl='units: '+units,ext='both')
    
    # ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
    ax=ax0
    
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()   
     
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
#    else:


    
    plt.show()
    
    fig_name = figpath+'/'+filename+'.png'
    fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
    
    plt.close()
    
    #plt.suptitle('units?')
    #plt.show()
    
    
    #plt.show()
    return mp, cbar, cs , fig, axes

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

nomask = np.ones(np.shape(landmask))
#%%

dict_path = '/home/disk/p/mlague/eos18/simple_land/scripts/python/analysis/global_pert/python_scripts/'

#global_pert_data = {}
#global_pert_data['slope'] = slope
#global_pert_data['slope_inv'] = slope_inv
#global_pert_data['intercept'] = intercept
#global_pert_data['r_value'] = r_value
#global_pert_data['p_values'] = p_values
#global_pert_data['std_dev'] = std_dev
#global_pert_data['datm_dlnd_scaled'] = datm_dlnd_scaled
#global_pert_data['dlnd_datm_scaled'] = dlnd_datm_scaled

load_location = dict_path + 'glbl_pert_dict_file_test_20180228_pickle'

with open(load_location, 'rb') as f:
    slope_analysis_dict = pickle.load(f)

load_location = dict_path + 'response_dict_file_test_20180228_pickle'

with open(load_location, 'rb') as f:
    response_dict = pickle.load(f)

#%%
figpath = {}
figpath['main'] = '/home/disk/eos18/mlague/simple_land/scripts/python/analysis/global_pert/figures_paperdraft/'
figpath['sensitivity'] = figpath['main'] + 'sens_slopes/'
figpath['sens_sub'] = figpath['sensitivity'] + 'subfigs/'
figpath['theoretical_dT'] = figpath['main'] + 'theoretical_dT/'
    
# If those directories don't exist, make them
for tag in list(figpath.keys()):
    if not os.path.exists(figpath[tag] ):
        os.makedirs(figpath[tag] )
        print('made directory:' + figpath[tag])

    

#%%

"""
    Idealized temperature change 
    
    OFFLINE
    
    Do 
    - dFSNS/dalb , 
    - actual dTs/dalb, dLW/dalb, dSH, dLH
    - theoretical dTS required to balance dFSNS, 
    - difference in theoretical dTs - actual dTs 
    => places where there is a big difference should match with places where turbulent fluxes are large
    - actual d(SH+LH) 
    
"""

# Focusing on albedo:
prop = "alb"

onoff = "offline"
sea = "ANN"
scale = 0.1 # d(atm) / d (0.1 albedo)
sign = -1.0

maskname = "landmask"

#%%
########################## 
#
#   d(FSNS)/d(0.1 alb)
#
##########################
var = "MML_fsns"
clim = [0.,14.]
units = "$W/m^2$"
cmap = plt.cm.viridis

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
dFSNS = datm_actual.copy()
E_in = datm_actual.copy()
#E_in = (slope_analysis_dict['slope'][onoff]['lnd']['MML_fsds'][prop][sea] -
#        slope_analysis_dict['slope'][onoff]['lnd']['MML_fsr'][prop][sea] +
#        slope_analysis_dict['slope'][onoff]['lnd']['MML_lwdn'][prop][sea] - # = 0
#        slope_analysis_dict['slope'][onoff]['lnd']['MML_lwup'][prop][sea]) * scale  * sign
E_in = ( slope_analysis_dict['slope'][onoff]['lnd']['MML_fsns'][prop][sea]) * scale  * sign
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/dalb " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=E_in,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)
#%%
########################## 
#
#   d(FLNS + SH + LH)/d(0.1 alb) 
#
#   should balance d FSNS
#
##########################
var = "E_out"
clim = [0.,14.]
units = "$W/m^2$"
cmap = plt.cm.viridis

#---------
# Actual:
#---------
E_out = ( slope_analysis_dict['slope'][onoff]['lnd']['MML_lwup'][prop][sea] +
             slope_analysis_dict['slope'][onoff]['lnd']['MML_lhflx'][prop][sea] +
                slope_analysis_dict['slope'][onoff]['lnd']['MML_shflx'][prop][sea] ) * scale * sign

#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/dalb " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=E_out,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

########################## 
#
#   E_in - E_out (should = 0)
#
##########################
var = "E_in_m_E_out"
clim = [-0.1,0.1]
units = "$W/m^2$"
cmap = plt.cm.RdBu_r


ttl = "Annual mean d/dalb " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=(E_in - E_out),
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)


#%%
########################## 
#
#   d(LW)/d(0.1 alb)
#
##########################
var = "MML_lwup"
clim = [0.,8.]
units = "$W/m^2$"
cmap = plt.cm.viridis

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/dalb " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=datm_actual,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#%%
########################## 
#
#   d(LH +SH)/d(0.1 alb)
#
##########################
var = "turbulent_flux"
clim = [0.,8.]
units = "$W/m^2$"
cmap = plt.cm.viridis

#---------
# Actual:
#---------
datm_actual = (slope_analysis_dict['slope'][onoff]['lnd']['MML_shflx'][prop][sea] + 
               slope_analysis_dict['slope'][onoff]['lnd']['MML_lhflx'][prop][sea]  ) * scale * sign
mapdata = datm_actual*landmask
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/dalb " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=datm_actual,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#%%
########################## 
#
#   d(Ts)/d(0.1 alb)
#
##########################
var = "MML_ts"
clim = [0.,3.5]
units = "$K$"
cmap = plt.cm.jet

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/dalb " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=datm_actual,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#---------
# Theoretical:
#   (in absence of any change in LH, SH, should equal change in net absorbed SW)
#---------
dLW = dFSNS
T_avg = response_dict['lnd'][onoff][var][prop][sea][1,:,:]  # take middel a=0.2 run, thats what the [1,:,:] is for
## LW = sig T^4
## dLW = 4 sig T^3 * dT
sig = 5.67e-8

dT = dLW / (3 * sig * T_avg**3)
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/dalb " + var + ", theoretical"

filename = 'theoretical_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=dT,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#---------
# Delta theoretical - actual dTs -> pattern should look like SH+LH
#---------
delta = dT - datm_actual
clim=[0.,2.0]
cmap=plt.cm.viridis

ttl = "Annual mean d/dalb " + var + ", theoretical - actual"

filename = 'theoretical_m_actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=delta,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#%%
########################## 
#
#   d(LH)/d(0.1 alb)
#
##########################
var = "MML_lhflx"
clim = [0.,8]
units = "$K$"
cmap = plt.cm.viridis

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/dalb " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=datm_actual,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#%%
########################## 
#
#   d(LSH)/d(0.1 alb)
#
##########################
var = "MML_shflx"
clim = [0.,8]
units = "$K$"
cmap = plt.cm.viridis

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/dalb " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=datm_actual,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)




#########################################
#%%

# Focusing on evaporative resistnace:
prop = "rs"

onoff = "offline"
sea = "ANN"
scale = 50. # d(atm) / d (0.1 albedo)
sign = 1.0

maskname = "landmask"

#%%
########################## 
#
#   d(FSNS)/d(0.1 alb)
#
##########################
var = "MML_fsns"
clim = [-1.,1.]
units = "$W/m^2$"
cmap = plt.cm.RdBu_r

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
dFSNS = datm_actual.copy()
E_in = datm_actual.copy()
#E_in = (slope_analysis_dict['slope'][onoff]['lnd']['MML_fsds'][prop][sea] -
#        slope_analysis_dict['slope'][onoff]['lnd']['MML_fsr'][prop][sea] +
#        slope_analysis_dict['slope'][onoff]['lnd']['MML_lwdn'][prop][sea] - # = 0
#        slope_analysis_dict['slope'][onoff]['lnd']['MML_lwup'][prop][sea]) * scale  * sign
E_in = ( slope_analysis_dict['slope'][onoff]['lnd']['MML_fsns'][prop][sea]) * scale  * sign
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/drs " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=E_in,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#%%
########################## 
#
#   d(FLNS)/d(0.1 alb)
#
##########################
var = "MML_flns"
clim = [-1.,1.]
units = "$W/m^2$"
cmap = plt.cm.RdBu_r

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
dFSNS = datm_actual.copy()
E_in = datm_actual.copy()
#E_in = (slope_analysis_dict['slope'][onoff]['lnd']['MML_fsds'][prop][sea] -
#        slope_analysis_dict['slope'][onoff]['lnd']['MML_fsr'][prop][sea] +
#        slope_analysis_dict['slope'][onoff]['lnd']['MML_lwdn'][prop][sea] - # = 0
#        slope_analysis_dict['slope'][onoff]['lnd']['MML_lwup'][prop][sea]) * scale  * sign
E_in = ( slope_analysis_dict['slope'][onoff]['lnd']['MML_fsns'][prop][sea]) * scale  * sign
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/drs " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=E_in,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)
#%%
########################## 
#
#   d(FLNS + SH + LH)/d(0.1 alb) 
#
#   should balance d FSNS
#
##########################
var = "E_out"
clim = [0.,14.]
units = "$W/m^2$"
cmap = plt.cm.viridis

#---------
# Actual:
#---------
E_out = ( slope_analysis_dict['slope'][onoff]['lnd']['MML_lwup'][prop][sea] +
             slope_analysis_dict['slope'][onoff]['lnd']['MML_lhflx'][prop][sea] +
                slope_analysis_dict['slope'][onoff]['lnd']['MML_shflx'][prop][sea] ) * scale * sign

#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/drs " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=E_out,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

########################## 
#
#   E_in - E_out (should = 0)
#
##########################
var = "E_in_m_E_out"
clim = [-0.1,0.1]
units = "$W/m^2$"
cmap = plt.cm.RdBu_r


ttl = "Annual mean d/drs " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=(E_in - E_out),
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)


#%%
########################## 
#
#   d(LW)/d(0.1 alb)
#
##########################
var = "MML_lwup"
clim = [-6.,6.]
units = "$W/m^2$"
cmap = plt.cm.RdBu

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/drs " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=datm_actual,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#%%
########################## 
#
#   d(LH +SH)/d(0.1 alb)
#
##########################
var = "turbulent_flux"
clim = [-6.,6.]
units = "$W/m^2$"
cmap = plt.cm.RdBu

#---------
# Actual:
#---------
datm_actual = (slope_analysis_dict['slope'][onoff]['lnd']['MML_shflx'][prop][sea] + 
               slope_analysis_dict['slope'][onoff]['lnd']['MML_lhflx'][prop][sea]  ) * scale * sign
mapdata = datm_actual*landmask
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/drs " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=datm_actual,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#%%
########################## 
#
#   d(Ts)/d(0.1 alb)
#
##########################
var = "MML_ts"
clim = [0.,3.5]
units = "$K$"
cmap = plt.cm.jet

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/drs " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=datm_actual,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#---------
# Theoretical:
#   (in absence of any change in LH, SH, should equal change in net absorbed SW)
#---------
dLW = dFSNS
T_avg = response_dict['lnd'][onoff][var][prop][sea][1,:,:]  # take middel a=0.2 run, thats what the [1,:,:] is for
## LW = sig T^4
## dLW = 4 sig T^3 * dT
sig = 5.67e-8

dT = dLW / (3 * sig * T_avg**3)
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/drs " + var + ", theoretical"

filename = 'theoretical_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=dT,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#---------
# Delta theoretical - actual dTs -> pattern should look like SH+LH
#---------
delta = dT - datm_actual
clim=[0.,2.0]
cmap=plt.cm.viridis

ttl = "Annual mean d/drs " + var + ", theoretical - actual"

filename = 'theoretical_m_actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=delta,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#%%
########################## 
#
#   d(LH)/d(0.1 alb)
#
##########################
var = "MML_lhflx"
clim = [-6.,6.]
units = "$W/m^2$"
cmap = plt.cm.RdBu

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/drs " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=datm_actual,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#%%
########################## 
#
#   d(LSH)/d(0.1 alb)
#
##########################
var = "MML_shflx"
clim = [-6.,6.]
units = "$W/m^2$"
cmap = plt.cm.RdBu

#---------
# Actual:
#---------
datm_actual = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scale * sign
mapdata = datm_actual*landmask
#mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)

ttl = "Annual mean d/drs " + var + ", actual"

filename = 'actual_' + var + '_' + sea + '_' + prop + '_' + maskname

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=datm_actual,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['theoretical_dT'],scale=None,filename=filename,ttl=ttl)

#%%
"""
    Plot delta T2m between a few different roughness simulations
"""

# Start by plotting r2 of roughness regression

mapdata = slope_analysis_dict['r_value']['online']['atm']['TREFHT']['hc']['ANN'][:]

filename = 'hc_r2_nomask'
cmap = plt.cm.viridis
clim = [0,1]
ttl = "$r^2$ value of hc linear fit"

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=mapdata,
                                         units='$r^2$',prop=None,sea=None,mask=np.ones(np.shape(landmask)),maskname='none', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)


# 1.0 m - 20 m

# Start by plotting r2 of roughness regression

mapdata = slope_analysis_dict['slope']['online']['atm']['TREFHT']['hc']['ANN'][:]

filename = 'hc_slope'
cmap = plt.cm.RdBu_r
clim = [-0.1,0.1]
ttl = "dT/dhc (1m) slope value of hc linear fit"

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=mapdata,
                                         units='K / m',prop=None,sea=None,mask=np.ones(np.shape(landmask)),maskname='none', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)







# 0.1 m - 1.0 m (should have same sign as 1.0 - 20 if the linearity is holding)


#%%
"""

    South American Cloud Deck

"""

"""
    ---------------------
    
    Plots to make:
        - full map of global cloud change
        - zoomed in regional cloud change map
        - vertical cross section at maybe 5 S showing:
            - clouds,
            - subsidence
            - humitidy
            - uplift? vertical mostion? I guess thats subsidence... WHAT IS THIS CITATION
            
    
"""

"""
    note: for slope vars, do a test to see if slope is significantly different from zero. I think the slope calculation might do something like this already... CHECK!!!
"""
#%%
#----------------------------------------------------------------------------------------
# Clouds, global
#----------------------------------------------------------------------------------------

# Load time series to do t-test!!!! 




#%%
#----------------------------------------------------------------------------------------
# Clouds, regional
#----------------------------------------------------------------------------------------


#%%
#----------------------------------------------------------------------------------------
# mark cross section on cloud tot map
#----------------------------------------------------------------------------------------



#%%
#----------------------------------------------------------------------------------------
# Cross sections
#----------------------------------------------------------------------------------------

#---------
# Clouds
#---------




#---------
# subsidence
#---------




