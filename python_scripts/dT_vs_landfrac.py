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

# OS interaction
import os
import sys

#from IPython.display import display
#from IPython.display import HTML
#import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)

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
landfrac = ds['landfrac'].values
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


# log rel'n roughness:
ds_low['log_hc'] = ds_cam['global_a2_cv2_hc0.5_rs100']
ds_med['log_hc'] = ds_cam['global_a2_cv2_hc1_rs100']
ds_high['log_hc'] = ds_cam['global_a2_cv2_hc2_rs100']

dsl_low['log_hc'] = ds_clm['global_a2_cv2_hc0.5_rs100']
dsl_med['log_hc'] = ds_clm['global_a2_cv2_hc1_rs100']
dsl_high['log_hc'] = ds_clm['global_a2_cv2_hc2_rs100']


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
   
sfc_props = ['alb','hc','rs','log_hc']
sfc_prop_ranges = np.array([ [0.1, 0.2, 0.3],
                  [0.5, 1., 2.],
                  [30., 100., 200.],
                  [np.log(0.5), np.log(1.), np.log(2.)]])
print(np.shape(sfc_prop_ranges))

print(sfc_prop_ranges)

seasons = ['ANN','DJF','MAM','JJA','SON']

i=0

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
        print(np.max(xvals))
        
        # grab atmospheric response data for current property, make an np.array
        raw_data = np.array(atm_resp[sea][prop])
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


#%% Calculate landfrac data
"""
    I want to calculate what percent of land in each experiment is able to drive
    a given temperature response from the atmosphere, for a unit change in surface
    property.
    
    That is, I'm going to take the datm/dlnd slopes, and see what total fraction
    of the land surface can give a certain steepness of datm/dlnd (ie maybe 50% of it 
    can reach a slope of 0.2 K / whatever, but ony 10% gives the very sensitive 3 K / whatever)
"""
# Doing it manually here.

# total land area by gridcell
land_area = landfrac*area_f19
tot_land_area = np.nansum(land_area)    # m2

pct_land = {}
tot_land = {}
T_range = {}

mags = {}
untis = {}

# Test this out on just one set, eg alb and annual 
sea = 'ANN'


# ALBEDO
prop = 'alb'
s = slope[sea][prop]
dsfc = 0.1  #delta albedo

mags[prop]=dsfc
units[prop]=''

sp = dsfc*s

T_range[prop] = np.linspace(np.min(sp),np.max(sp),50)
tot_land[prop] = np.nan*np.ones(np.shape(T_range[prop]))
pct_land[prop] = np.nan*np.ones(np.shape(T_range[prop]))
    
for iter in range(np.size(T_range[prop])):
    T = T_range[prop][iter]
    less_than_T = np.where(sp<T,1,np.nan)
    tot_land[prop][iter] = np.nansum(less_than_T*land_area)
    pct_land[prop][iter] = 100*tot_land[prop][iter]/tot_land_area
    
# roughness
prop = 'hc'
s = slope[sea][prop]
dsfc = 0.5  #delta albedo

mags[prop]=dsfc
units[prop]='m'

sp = dsfc*s

T_range[prop] = np.linspace(np.min(sp),np.max(sp),50)
tot_land[prop] = np.nan*np.ones(np.shape(T_range[prop]))
pct_land[prop] = np.nan*np.ones(np.shape(T_range[prop]))
    
for iter in range(np.size(T_range[prop])):
    T = T_range[prop][iter]
    less_than_T = np.where(sp<T,1,np.nan)
    tot_land[prop][iter] = np.nansum(less_than_T*land_area)
    pct_land[prop][iter] = 100*tot_land[prop][iter]/tot_land_area
    
# log roughness (meaningful?)
prop = 'log_hc'
s = slope[sea][prop]
dsfc = 0.5  #delta albedo

mags[prop]=dsfc
units[prop]='m'

sp = dsfc*s

T_range[prop] = np.linspace(np.min(sp),np.max(sp),50)
tot_land[prop] = np.nan*np.ones(np.shape(T_range[prop]))
pct_land[prop] = np.nan*np.ones(np.shape(T_range[prop]))
    
for iter in range(np.size(T_range[prop])):
    T = T_range[prop][iter]
    less_than_T = np.where(sp<T,1,np.nan)
    tot_land[prop][iter] = np.nansum(less_than_T*land_area)
    pct_land[prop][iter] = 100*tot_land[prop][iter]/tot_land_area
    


# evaporative resistance
prop = 'rs'
s = slope[sea][prop]
dsfc = 50  #delta albedo

mags[prop]=dsfc
units[prop]='s/m'

sp = dsfc*s

T_range[prop] = np.linspace(np.min(sp),np.max(sp),50)
tot_land[prop] = np.nan*np.ones(np.shape(T_range[prop]))
pct_land[prop] = np.nan*np.ones(np.shape(T_range[prop]))
    
for iter in range(np.size(T_range[prop])):
    T = T_range[prop][iter]
    less_than_T = np.where(sp<T,1,np.nan)
    tot_land[prop][iter] = np.nansum(less_than_T*land_area)
    pct_land[prop][iter] = 100*tot_land[prop][iter]/tot_land_area
    


fig, ax = plt.subplots(1, 1, figsize=(4,4))

p = {}

for prop in sfc_props:
    #p[a]=
    lbl_str = prop+' ( $\Delta$ '+np.str(mags[prop])+' '+units[prop]+')'
    plt.plot(pct_land[prop],T_range[prop],label=lbl_str)


plt.title('% of land with a given temperature change \n to a $\Delta$ increase in surface property')
plt.xlabel('% of land')
plt.ylabel('2m Air Temperature [K]')

plt.plot([0,100],[0,0],color='k',linestyle=':')
#plt.legend([p1,p2],['aa','a'])
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles,bbox_to_anchor=(1.5,0.5))#,bbox_transform=plt.gcf().transFigure)

plt.ylim([-3,1])
#plt.xlim([0,20])
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)

bigger_than_10 = np.where(s<-10,land_area,np.nan)
# hey, this could make a cool video, before summing it all up. 

filename = 'pct_land_with_dT2m'

fig_name = figpath+'/sensitivity/'+filename+'.png'
fig.savefig(fig_name,dpi=1200,transparent=True,facecolor='w',
            edgecolor='w',orientation='portrait',bbox_inches='tight', 
            pad_inches=0.1,frameon=None)

fig_name = figpath+'/sensitivity/'+filename+'.eps'
fig.savefig(fig_name,dpi=1200,transparent=True,facecolor='w',
            edgecolor='w',orientation='portrait',bbox_inches='tight', 
            pad_inches=0.1,frameon=None)



#%% Calculate landfrac data
"""
    I want to calculate what percent of land in each experiment is able to drive
    a given temperature response from the atmosphere, for a unit change in surface
    property.
    
    That is, I'm going to take the datm/dlnd slopes, and see what total fraction
    of the land surface can give a certain steepness of datm/dlnd (ie maybe 50% of it 
    can reach a slope of 0.2 K / whatever, but ony 10% gives the very sensitive 3 K / whatever)
"""

# Test this out on just one set, eg alb and annual 
sea = 'ANN'
prop = 'alb'

# total land area by gridcell
land_area = landfrac*area_f19
s = slope[sea][prop]
print(np.shape(s))

pct_land = {}
tot_land = {}
T_range = {}

# Albedo cools, so its the big negative numbers taht are places that are most sensitive

tot_land_area = np.nansum(land_area)    # m2

albs_to_test = np.array([0.2,0.1,0.05,0.01])
albs_to_test = np.array([0.1])
albs = ["%.2f" % a for a in albs_to_test]


for a in range(np.size(albs_to_test)):
    alb_val = albs_to_test[a]
    alb = albs[a]
    #print(alb_val)
    #print(alb)
    sp = alb_val*s

    
    T_range[alb] = np.linspace(np.min(sp),np.max(sp),50)
    tot_land[alb] = np.nan*np.ones(np.shape(T_range[alb]))
    pct_land[alb] = np.nan*np.ones(np.shape(T_range[alb]))
    
    for iter in range(np.size(T_range[alb])):
        T = T_range[alb][iter]
        #print(T)
        less_than_T = np.where(sp<T,1,np.nan)
        #plt.imshow(less_than_T)
        tot_land[alb][iter] = np.nansum(less_than_T*land_area)
        #print(tot_land[alb][T])
        pct_land[alb][iter] = 100*tot_land[alb][iter]/tot_land_area
    


fig, ax = plt.subplots(1, 1, figsize=(4,4))

p = {}

for a in albs:
    #p[a]=
    plt.plot(pct_land[a],T_range[a],label=a)


plt.title('Temperature change per $\Delta$ increase in Albedo')
plt.xlabel('% of land')
plt.ylabel('2m Air Temperature [K]')
#plt.legend([p1,p2],['aa','a'])
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles,bbox_to_anchor=(1.4,0.4))#,bbox_transform=plt.gcf().transFigure)

plt.ylim([-6,0.1])
#plt.xlim([0,20])
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)

bigger_than_10 = np.where(s<-10,land_area,np.nan)
# hey, this could make a cool video, before summing it all up. 



#        fig, axes = plt.subplots(1, 3, figsize=(18,6))
#        
#        NCo = 21
#        NTik_dlnd = 5
#        NTik_datm = 9#np.floor(NCo/2)
#        NTik_r2 = 6
#        
#        ax0 = axes.flatten()[0]
#        plt.sca(ax0)
#        ttl = '$\delta$ '+prop+' per 0.1K change in T2m'
#        #units = 'unitless'
#        #clim_diff = [-.01,.01]
#        mapdata = mapdata_inv*0.1
#        #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units )   #plt.cm.BuPu_r
#        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
#        ax=ax0
#        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#            item.set_fontsize(12)
#        #cbar.set_ticklabels(np.linspace(-0.01,0.01,9))



#%% ALBEDO: ANNUAL MEAN
#
#
#myvar = 'TREFHT'
## Loop over properties:
#for prop in sfc_props: 
#    
#    # set appropriate colour limits
#    if prop =='alb':
#        clim_dlnd = [-0.01, 0.01]
#        clim_datm = [-25,25]
#        units='unitless'
#    elif prop =='hc':
#        clim_dlnd = [-2.,2.]
#        clim_datm = [-0.5,0.5]
#        units='m'
#    elif prop=='rs' :
#        clim_dlnd = [-30.,30.]
#        clim_datm = [-.025,.025]
#        units='s/m'
#    
#    # Loop over seasons:
#    for sea in seasons:
#     #   #%% ALBEDO - Unmasked
#        
#    
#        #prop = 'alb'
#        #myvar = 'TREFHT'
#        ds0 = ds_cam['global_a2_cv2_hc1_rs100']
#        mask_name = 'nomask'
#        
#        #sea = 'ANN'
#        
#        mapdata_slope = slope[sea][prop]
#        mapdata_inv = slope[sea][prop]**(-1)
#        mapdata_r2 = r_value[sea][prop]**2
#        
#        
#        ttl_main = prop #'Albedo'
#        filename = 'sens_slopes_'+prop+'_'+mask_name+'_'+sea
#        
#        
#        cmap_abs = plt.cm.viridis
#        cmap_diff = plt.cm.RdBu_r
#        
#        fig, axes = plt.subplots(1, 3, figsize=(18,6))
#        
#        NCo = 21
#        NTik_dlnd = 5
#        NTik_datm = 9#np.floor(NCo/2)
#        NTik_r2 = 6
#        
#        ax0 = axes.flatten()[0]
#        plt.sca(ax0)
#        ttl = '$\delta$ '+prop+' per 0.1K change in T2m'
#        #units = 'unitless'
#        #clim_diff = [-.01,.01]
#        mapdata = mapdata_inv*0.1
#        #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units )   #plt.cm.BuPu_r
#        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
#        ax=ax0
#        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#            item.set_fontsize(12)
#        #cbar.set_ticklabels(np.linspace(-0.01,0.01,9))
#        ax1 = axes.flatten()[1]
#        plt.sca(ax1)
#        ttl = '$\delta$ T2m per unit change in '+prop
#        units = 'K'
#       # clim_diff = [-25,25]
#        #clim_abs = clim_diff
#        mapdata = mapdata_slope
#        #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units )   #plt.cm.BuPu_r
#        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_datm,ext='both',disc=True )
#        ax=ax1
#        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#            item.set_fontsize(12)
#        
#        ax2 = axes.flatten()[2]
#        plt.sca(ax2)
#        ttl = 'r^2'
#        units = 'r^2'
#        clim_abs = [0.5,1]
#        mapdata = mapdata_r2
#        #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units )   #plt.cm.BuPu_r
#        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_r2,ext='min')
#        ax=ax2
#        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#            item.set_fontsize(12)
#        
#        fig.subplots_adjust(top=1.15)
#        fig.suptitle(ttl_main, fontsize=20)   
#        
#        # Annotate with season, variable, date
#        ax0.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
#                 ha = 'left',va = 'center',
#                 transform = ax0.transAxes)
#        
#        fig_name = figpath+'/sensitivity/'+filename+'.eps'
#        fig.savefig(fig_name,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
#                    pad_inches=0.1,frameon=None)
#        
#        fig_name = figpath+'/sensitivity/'+filename+'.png'
#        fig.savefig(fig_name,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
#                    pad_inches=0.1,frameon=None)
# 
#        # Save the sub-plots as individual panels
#        
#        # (a) dlnd/datm
#        extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        fig_png = figpath+'/sensitivity/subplots/'+filename+'_a.png'
#        fig_eps = figpath+'/sensitivity/subplots/'+filename+'_a.eps'
#        vals = extent.extents
#        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
#        fig.savefig(fig_png,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        fig.savefig(fig_eps,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        
#        # (b) datm/dlnd
#        # add datetime tag
#        # Annotate with season, variable, date
#        ax1.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
#                 ha = 'left',va = 'center',
#                 transform = ax1.transAxes)
#        extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        fig_png = figpath+'/sensitivity/subplots/'+filename+'_b.png'
#        fig_eps = figpath+'/sensitivity/subplots/'+filename+'_b.eps'
#        vals = extent.extents
#        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
#        fig.savefig(fig_png,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        fig.savefig(fig_eps,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        
#        # (c) r^2
#        # Annotate with season, variable, date
#        ax2.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
#                 ha = 'left',va = 'center',
#                 transform = ax2.transAxes)
#        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        fig_png = figpath+'/sensitivity/subplots/'+filename+'_c.png'
#        fig_eps = figpath+'/sensitivity/subplots/'+filename+'_c.eps'
#        vals = extent.extents
#        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
#        fig.savefig(fig_png,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        fig.savefig(fig_eps,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#
#       
#        plt.close()
#        
#        
#    #    #%% ALBEDO - Land Mask
#    
#    
#        #prop = 'alb'
#        #myvar = 'TREFHT'
#        ds0 = ds_cam['global_a2_cv2_hc1_rs100']
#        mask_name = 'lndmask'
#        #sea = 'ANN'
#        
#        mapdata_slope = slope[sea][prop]
#        mapdata_inv = slope[sea][prop]**(-1)
#        mapdata_r2 = r_value[sea][prop]**2
#        
#        
#        ttl_main = prop #'Albedo'
#        filename = 'sens_slopes_'+prop+'_'+mask_name+'_'+sea
#        
#        
#        cmap_abs = plt.cm.get_cmap('viridis',11)#plt.cm.viridis()
#        cmap_diff = plt.cm.RdBu_r
#        
#        fig, axes = plt.subplots(1, 3, figsize=(18,6))
#        
#        ax0 = axes.flatten()[0]
#        plt.sca(ax0)
#        ttl = '$\delta$ '+prop+' per 0.1K change in T2m'
#        #units = 'unitless'
#        #clim_diff = [-.01,.01]
#        mapdata = mapdata_inv*0.1*bareground_mask
#        mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
#        #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik,ext='both',disc=True )   #plt.cm.BuPu_r
#        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
#        ax=ax0
#       # mml_map(LN,LT,mapdata,ds,myvar,proj,title=None,clim=None,colmap=None,cb_ttl=None,disc=None,ncol=None,nticks=None,ext=None):
#   
#        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#            item.set_fontsize(12)
#        
#        ax1 = axes.flatten()[1]
#        plt.sca(ax1)
#        ttl = '$\delta$ T2m per unit change in '+prop
#        units = 'K'
#        #clim_diff = [-25,25]
#        #clim_abs = clim_diff
#        mapdata = mapdata_slope*bareground_mask
#        mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
#        #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units )   #plt.cm.BuPu_r
#        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_datm,ext='both',disc=True )
#        ax=ax1
#        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#            item.set_fontsize(12)
#        
#        ax2 = axes.flatten()[2]
#        plt.sca(ax2)
#        ttl = 'r^2'
#        units = 'r^2'
#        clim_abs = [0.5,1]
#        mapdata = mapdata_r2*bareground_mask
#        mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
#        #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units )   #plt.cm.BuPu_r
#        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_r2,ext='min')
#        ax=ax2
#        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#            item.set_fontsize(12)
#        
#        fig.subplots_adjust(top=1.15)
#        fig.suptitle(ttl_main, fontsize=20)    
#        
#        # Annotate with season, variable, date
#        ax0.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
#                 ha = 'left',va = 'center',
#                 transform = ax0.transAxes)
#        
#        
#        fig_name = figpath+'/sensitivity/'+filename+'.eps'
#        fig.savefig(fig_name,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
#                    pad_inches=0.1,frameon=None)
#        
#        fig_name = figpath+'/sensitivity/'+filename+'.png'
#        fig.savefig(fig_name,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
#                    pad_inches=0.1,frameon=None)
#        
#        
#        # Save the sub-plots as individual panels
#        
#        # (a) dlnd/datm
#        extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        bbx = extent.extents
#        bbx[0]=bbx[1]-0.25
#        bbx[1]=bbx[1]-0.5
#        bbx[2]=bbx[2]+0.25
#        bbx[3]=bbx[3]+0.2
#        fig_png = figpath+'/sensitivity/subplots/'+filename+'_a.png'
#        fig_eps = figpath+'/sensitivity/subplots/'+filename+'_a.eps'
#        vals = extent.extents
#        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
#        fig.savefig(fig_png,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        fig.savefig(fig_eps,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        
#        # (b) datm/dlnd
#        # add datetime tag
#        # Annotate with season, variable, date
#        ax1.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
#                 ha = 'left',va = 'center',
#                 transform = ax1.transAxes)
#        extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        fig_png = figpath+'/sensitivity/subplots/'+filename+'_b.png'
#        fig_eps = figpath+'/sensitivity/subplots/'+filename+'_b.eps'
#        vals = extent.extents
#        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
#        fig.savefig(fig_png,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        fig.savefig(fig_eps,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        
#        # (c) r^2
#        # Annotate with season, variable, date
#        ax2.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
#                 ha = 'left',va = 'center',
#                 transform = ax2.transAxes)
#        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        fig_png = figpath+'/sensitivity/subplots/'+filename+'_c.png'
#        fig_eps = figpath+'/sensitivity/subplots/'+filename+'_c.eps'
#        vals = extent.extents
#        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
#        fig.savefig(fig_png,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        fig.savefig(fig_eps,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        
#        plt.close()
#        
#        
#    #    #%% ALBEDO - ocn mask
#        
#        
#       # prop = 'alb'
#        #myvar = 'TREFHT'
#        ds0 = ds_cam['global_a2_cv2_hc1_rs100']
#        mask_name = 'ocnmask'
#        #sea = 'ANN'
#        
#        mapdata_slope = slope[sea][prop]
#        mapdata_inv = slope[sea][prop]**(-1)
#        mapdata_r2 = r_value[sea][prop]**2
#        
#        
#        ttl_main = prop #'Albedo'
#        filename = 'sens_slopes_'+prop+'_'+mask_name+'_'+sea
#        
#        
#        cmap_abs = plt.cm.viridis
#        cmap_diff = plt.cm.RdBu_r
#        
#        fig, axes = plt.subplots(1, 3, figsize=(18,6))
#        
#        ax0 = axes.flatten()[0]
#        plt.sca(ax0)
#        ttl = '$\delta$ '+prop+' per 0.1K change in T2m'
#        #units = 'unitless'
#        #clim_diff = [-.01,.01]
#        mapdata = mapdata_inv*0.1*ocn_glc_mask
#        mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
#        #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units )   #plt.cm.BuPu_r
#        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_dlnd,colmap=cm_dlnd, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
#        ax=ax0
#        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#            item.set_fontsize(12)
#        
#        ax1 = axes.flatten()[1]
#        plt.sca(ax1)
#        ttl = '$\delta$ T2m per unit change in '+prop
#        units = 'K'
#        #clim_diff = [-25,25]
#        #clim_abs = clim_diff
#        mapdata = mapdata_slope*ocn_glc_mask
#        mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
#        #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units )   #plt.cm.BuPu_r
#        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_datm,colmap=cm_datm, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_datm,ext='both',disc=True )
#        ax=ax1
#        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#            item.set_fontsize(12)
#        
#        ax2 = axes.flatten()[2]
#        plt.sca(ax2)
#        ttl = 'r^2'
#        units = 'r^2'
#        clim_abs = [0.5,1]
#        mapdata = mapdata_r2*ocn_glc_mask
#        mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
#        #mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units )   #plt.cm.BuPu_r
#        mp, cbar, cs = mml_map(LN,LT,mapdata,ds0,myvar,'moll',title=ttl,clim=clim_abs,colmap=cmap_abs, cb_ttl='units: '+units, ncol=NCo,nticks=NTik_r2,ext='min')
#        ax=ax2
#        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#            item.set_fontsize(12)
#        
#        fig.subplots_adjust(top=1.15)
#        fig.suptitle(ttl_main, fontsize=20)    
#        
#        # Annotate with season, variable, date
#        ax0.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
#                 ha = 'left',va = 'center',
#                 transform = ax0.transAxes)
#        
#        fig_name = figpath+'/sensitivity/'+filename+'.eps'
#        fig.savefig(fig_name,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
#                    pad_inches=0.1,frameon=None)
#        
#        fig_name = figpath+'/sensitivity/'+filename+'.png'
#        fig.savefig(fig_name,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
#                    pad_inches=0.1,frameon=None)
#
#        # Save the sub-plots as individual panels
#        
#        # (a) dlnd/datm
#        extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        fig_png = figpath+'/sensitivity/subplots/'+filename+'_a.png'
#        fig_eps = figpath+'/sensitivity/subplots/'+filename+'_a.eps'
#        vals = extent.extents
#        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
#        fig.savefig(fig_png,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        fig.savefig(fig_eps,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        
#        # (b) datm/dlnd
#        # add datetime tag
#        # Annotate with season, variable, date
#        ax1.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
#                 ha = 'left',va = 'center',
#                 transform = ax1.transAxes)
#        extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        fig_png = figpath+'/sensitivity/subplots/'+filename+'_b.png'
#        fig_eps = figpath+'/sensitivity/subplots/'+filename+'_b.eps'
#        vals = extent.extents
#        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
#        fig.savefig(fig_png,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        fig.savefig(fig_eps,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        
#        # (c) r^2
#        # Annotate with season, variable, date
#        ax2.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop ,fontsize='10',
#                 ha = 'left',va = 'center',
#                 transform = ax2.transAxes)
#        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        fig_png = figpath+'/sensitivity/subplots/'+filename+'_c.png'
#        fig_eps = figpath+'/sensitivity/subplots/'+filename+'_c.eps'
#        vals = extent.extents
#        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
#        fig.savefig(fig_png,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        fig.savefig(fig_eps,dpi=1200,transparent=True,facecolor='w',
#                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
#                    frameon=None)
#        
#        plt.close()
#        
#        
#    # end seasonal loop.
#    
## end property loop
#    
#%% ALBEDO - SEASONAL

#%% DJF



#%%



#%%



#%%




