#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:25:20 2017

@author: mlague

Load already computed slope data from txt file dictionary, and do some analysis!

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

load_location = dict_path + 'response_dict_file_test_20180201_pickle'

with open(load_location, 'rb') as f:
    response_dict = pickle.load(f)

#%%
figpath = {}
figpath['main'] = '/home/disk/eos18/mlague/simple_land/scripts/python/analysis/global_pert/figures_onoff/'
figpath['sensitivity'] = figpath['main'] + '/sens_slopes/'
figpath['sens_sub'] = figpath['sensitivity'] + '/subfigs/'
    


#%%
"""
    Make those 3x1 plots of dlnd/datm, datm/dlnd, r^2 for nomask and landmask cases
    (make a fn to do this... could be a local fn...)
"""


def sensitivity_map_3by1(datm=None,dlnd=None,r2=None,onoff=None,var=None,
                         units=None,prop=None,sea=None,mask=None,maskname=None, LN=None, LT=None,
                         cb=None,climits=None,save_subplot=None,figpath=None,scale=None):
    #----------------------
    # Inputs:
    #
    # xxxxxx slope_dictionary = dictionary of form dicitonary-> slope/slope_inv/r2 -> 
    # xxxxxx atm_or_lnd -> 'atm' or 'lnd' tell function if we're interested in atmospheric or land analysis
    # 
    #
    # datm = datm/dlnd data (slope)
    # dlnd = dlnd/datm data (slope_inv)
    # r2 = r2 data for slope
    # onoff = 'online' or 'offline'
    # var = variable to look up
    # units = dictionary units[datm] = K/0.1 alb, dicitonary units[dlnd] = dalb/0.1K
    # sea = track season for annotating figure & filename
    # mask = mask for land/ocn/none
    # LN, LT = feed in meshgrid lat lon
    # cm = dictionary cm['dlnd'], cm['datm'], cm['r2'] -> colormap for each subplot
    # clim = dictionary clim['dlnd'], clim['datm'], clim['r2'] -> colormap for each subplot
    # save_subplot = flag to save or not to save the subfigures individually
    # figpath = pass in figure path for this script (will make a subplot folder there)
    #
    #
    # Returns:
    #
    # fig: figure handle
    # axes: axis handles for each subplot
    # 
    #
    #
    #----------------------
    
    if cb:
        cb_dlnd = cb['dlnd']
        cb_datm = cb['datm']
        cb_r2 = cb['r2']
    else:
        cb_dlnd = plt.cm.plasma
        cb_datm = plt.cm.viridis
        cb_r2 = plt.cm.cool
          
    if prop:
        ttl_main = prop
    else:
        ttl_main = 'dlnd/datm, datm/dlnd, r2'
    
    mask_name = np.str(mask)
    
        
    filename = '3x1_sensitivity_'+var+'_'+prop+'_'+maskname+'_'+sea+'_'+onoff
                
    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    
    
    #------------------------------------------
    # left plot: dlnd/datm
    #------------------------------------------
    
    ax0 = axes.flatten()[0]
    plt.sca(ax0)
    ttl = '$\delta$ '+prop+' per change in '+var
    
    mapdata = dlnd
    
    mapdata = mapdata*mask
    mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
        
    
    cm = cb_dlnd
    if climits:
        clim = climits['dlnd']
    else:
        clim = [-1,1]
    unit = units['dlnd']
    
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')
    
    # ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
    ax=ax0
    
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()   
     
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
    
   
    #------------------------------------------
    # middle plot: datm/dlnd
    #------------------------------------------
    
    ax1 = axes.flatten()[1]
    plt.sca(ax1)
    
    unit = units['datm']
#    if prop == 'alb':
#        #scale =' 0.01'
#    elif prop =='rs':
#        #scale = '10'
#    elif prop =='hc':
#        #scale = '0.1'
        
    ttl = '$\delta$ '+var+' per'+ np.str(scale) + ' change in '+prop
    
    mapdata = datm
    
    mapdata = mapdata*mask
    mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
    
    
    cm = cb_datm
    clim = climits['datm']
    
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')
    # ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
    ax=ax1
    
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()   
     
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
    
   



    #------------------------------------------
    # right plot: r^2 of datm/dlnd
    #------------------------------------------
    
    ax2 = axes.flatten()[2]
    plt.sca(ax2)
    
    unit = units['r2']
    ttl = 'r$^2$ value'
    
    mapdata = r2
    
    mapdata = mapdata*mask
    mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
    
    
    cm = cb_r2
    clim = climits['r2']
    
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')
    
    # ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
    ax=ax2
    
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()   
     
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
    
   


    #------------------------------------------
    # Save main figure, and subfigures if req'd
    #------------------------------------------
    
  
    # Annotate with season, variable, date
    ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', '+prop +', '+var+', '+onoff,fontsize='10',
             ha = 'left',va = 'center',
             transform = ax0.transAxes)

    
    plt.show()
    
    fig_name = figpath+'/sens_slopes/'+filename+'.png'
    fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
    
    if save_subplot==1:
        # Save the sub-plots as individual panels
                
        # (a) dlnd/datm
        extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        bbx = extent.extents
        bbx[0]=bbx[1]-0.25
        bbx[1]=bbx[1]-0.5
        bbx[2]=bbx[2]+0.25
        bbx[3]=bbx[3]+0.2
        fig_png = figpath+'/sens_slopes/subfigs/'+filename+'_a.png'
        vals = extent.extents
        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                    frameon=None)
        
        # (b) datm/dlnd
        # add datetime tag
        # Annotate with season, variable, date
        ax1.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop+','+onoff ,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax1.transAxes)
        extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig_png = figpath+'/sens_slopes/subfigs/'+filename+'_b.png'
        vals = extent.extents
        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                    frameon=None)
        
        # (c) r^2
        # Annotate with season, variable, date
        ax2.text(0.,-0.35,time.strftime("%x")+'\n'+sea +', '+prop +','+onoff,fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax2.transAxes)
        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig_png = fig_png = figpath+'/sens_slopes/subfigs/'+filename+'_c.png'
        vals = extent.extents
        new_extent = extent.from_extents(vals[0]-0.45,vals[1]-1.,vals[2]+0.25,vals[3]+0.45)
        fig.savefig(fig_png,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches=new_extent.expanded(1.,1.), 
                    frameon=None)
            
            
            
    plt.close()
                
                
            

    
    return fig, axes



def mml_map_NA_local(datm=None,onoff=None,var=None,
                     units=None,prop=None,sea=None,mask=None,maskname=None, 
                     LN=None, LT=None,
                     cb=None,climits=None,save_subplot=None,
                     figpath=None,scale=None):
    # need to have already opened a figure/axis
    #plt.sca(ax)
    
    from mpl_toolkits.basemap import Basemap, cm
    
    
    
    if cb:
        cb = cb
    else:
        cb = plt.cm.viridis
          
    if prop:
        ttl_main = prop + ' (NA)'
    else:
        ttl_main = 'datm/dlnd'
    
    #mask_name = np.str(mask)
    
        
    filename = 'NA_datmdlnd'+var+'_'+prop+'_'+maskname+'_'+sea+'_'+onoff
                
    fig, axes = plt.subplots(1, 1, figsize=(5,4))
    
    
    #------------------------------------------
    # left plot: dlnd/datm
    #------------------------------------------
    
    ax0 = plt.gca()
    
    mapdata = datm
    
    mapdata = mapdata*mask
    mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
        
    
    
    if climits:
        clim = climits
    else:
        clim = [-1,1]
    
    
    
    # 
    lat2 = -88.105262756347656
 
    #mp = Basemap(llcrnrlon=-150.,llcrnrlat=5.,urcrnrlon=-30.,urcrnrlat=80.,
    #        projection='lcc',lat_1=10.,lon_0=-30.,
    #        resolution ='l',area_thresh=10000.) 
#    latcorners = nc.variables['lat'][:]
#    loncorners = -nc.variables['lon'][:]
#    lon_0 = -nc.variables['true_lon'].getValue()
#    lat_0 = nc.variables['true_lat'].getValue()
    mp = Basemap(resolution='l',projection='stere', lat_ts=50,lat_0=50,lon_0=-107,width=10000000,height=8000000)

    mp.drawcoastlines()
    mp.drawmapboundary(fill_color='1.')  # make map background white
    mp.drawparallels(np.arange(-80.,81.,20.))
    mp.drawmeridians(np.arange(-180.,181.,20.))
    #mp.drawparallels(np.arange(10,80,10),labels=[1,1,0,0])
    #mp.drawmeridians(np.arange(-140,0,20),labels=[0,0,0,1])    
    
    
    #parallels = np.arange(0.,90,10.)
    #mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
    #meridians = np.arange(180.,360.,10.)
    #mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
    #(x, y) = m(LONXY, LATXY)
    cs = mp.pcolormesh(LN,LT,mapdata,cmap=cb,latlon=True)
    
#    if colmap:
    cs.cmap = cb
#    else:
#        cs.cmap = plt.cm.inferno    
    
    cbar = mp.colorbar(cs,location='bottom',pad="5%")
    
    cs.cmap.set_bad('white',1.)
    
    cb_ttl = units+' per '+prop
    cbar.set_label(cb_ttl,fontsize=12)
    
    plt.title(ttl_main,fontsize=12)
    
    if climits:
        cbar.set_clim(climits[0],climits[1])
        cs.set_clim(climits[0],climits[1])
    
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()  
    
    #------------------------------------------
    # Save main figure, and subfigures if req'd
    #------------------------------------------
    
  
    # Annotate with season, variable, date
    ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', '+prop +', '+var+', '+onoff,fontsize='10',
             ha = 'left',va = 'center',
             transform = ax0.transAxes)

    
    plt.show()
    
    fig_name = figpath+'/sens_slopes/NA/'+filename+'.png'
    fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
    
    plt.close()
    
    #plt.suptitle('units?')
    #plt.show()
    
    
    #plt.show()
    return mp, cbar, cs , fig, axes



def mml_map_Africa_local(datm=None,onoff=None,var=None,
                     units=None,prop=None,sea=None,mask=None,maskname=None, 
                     LN=None, LT=None,
                     cb=None,climits=None,save_subplot=None,
                     figpath=None,scale=None):
    # need to have already opened a figure/axis
    #plt.sca(ax)
    
    from mpl_toolkits.basemap import Basemap, cm
    
    
    
    if cb:
        cb = cb
    else:
        cb = plt.cm.viridis
          
    if prop:
        ttl_main = prop + ' (NA)'
    else:
        ttl_main = 'datm/dlnd'
    
    #mask_name = np.str(mask)
    
        
    filename = 'Africa_datmdlnd'+var+'_'+prop+'_'+maskname+'_'+sea+'_'+onoff
                
    fig, axes = plt.subplots(1, 1, figsize=(5,4))
    
    
    #------------------------------------------
    # left plot: dlnd/datm
    #------------------------------------------
    
    ax0 = plt.gca()
    
    mapdata = datm
    
    mapdata = mapdata*mask
    mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
        
    
    
    if climits:
        clim = climits
    else:
        clim = [-1,1]
    
    
    
    # 
    lat2 = -88.105262756347656
 
    #mp = Basemap(llcrnrlon=-150.,llcrnrlat=5.,urcrnrlon=-30.,urcrnrlat=80.,
    #        projection='lcc',lat_1=10.,lon_0=-30.,
    #        resolution ='l',area_thresh=10000.) 
#    mp = Basemap(resolution='l',projection='stere', 
#                 lat_ts=20,lat_0=20,lon_0=12,
#                 width=8000000,height=9000000)
#    mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0,resolution='c') # can't make it start anywhere other than 180???
    mp = Basemap(projection='robin',lon_0=0.,lat_0 = 0,resolution='c') # can't make it start anywhere other than 180???
    #mp = Basemap(resolution='l',projection='robin', lat_ts=27,lat_0=27,lon_0=13,width=8000000,height=10000000)
    #mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0,resolution='c')
    #mp = Basemap(projection='stere',lon_0=180.,lat_0=90.,lat_ts=90.,
    #        llcrnrlat=-45,urcrnrlat=25,
    #        llcrnrlon=-12,urcrnrlon=90,
    #        resolution='l')
    mp.drawcoastlines()
    mp.drawmapboundary(fill_color='1.')  # make map background white
    mp.drawparallels(np.arange(-90.,91.,15.))
    mp.drawmeridians(np.arange(-180.,181.,20.))
    #mp.drawparallels(np.arange(10,80,10),labels=[1,1,0,0])
    #mp.drawmeridians(np.arange(-140,0,20),labels=[0,0,0,1])    
    
    
    #parallels = np.arange(0.,90,10.)
    #mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
    #meridians = np.arange(180.,360.,10.)
    #mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
    #(x, y) = m(LONXY, LATXY)
    cs = mp.pcolormesh(LN,LT,mapdata,cmap=cb,latlon=True)
    
#    if colmap:
    cs.cmap = cb
#    else:
#        cs.cmap = plt.cm.inferno    
    
    cbar = mp.colorbar(cs,location='bottom',pad="5%")
    
    cs.cmap.set_bad('white',1.)
    
    cb_ttl = units+' per '+prop
    cbar.set_label(cb_ttl,fontsize=12)
    
    plt.title(ttl_main,fontsize=12)
    
    if climits:
        cbar.set_clim(climits[0],climits[1])
        cs.set_clim(climits[0],climits[1])
    
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()  
    
    #------------------------------------------
    # Save main figure, and subfigures if req'd
    #------------------------------------------
    
  
    # Annotate with season, variable, date
    ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', '+prop +', '+var+', '+onoff,fontsize='10',
             ha = 'left',va = 'center',
             transform = ax0.transAxes)

    
    plt.show()
    
    fig_name = figpath+'/sens_slopes/Africa/'+filename+'.png'
    fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
    
    plt.close()
    
    #plt.suptitle('units?')
    #plt.show()
    
    
    #plt.show()
    return mp, cbar, cs , fig, axes


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
   
    
    cm = cb_dlnd
    if climits:
        clim = climits['dlnd']
    else:
        clim = [-1,1]
    unit = units['dlnd']
    
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=None,proj='moll',title=ttl_main,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')
    
    # ncol=NCo,nticks=NTik_dlnd,ext='both',disc=True )
    ax=ax0
    
    tick_locator = ticker.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()   
     
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(12)
#    else:


    
    plt.show()
    
    fig_name = figpath+'/sens_slopes/'+filename+'.png'
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
        loop over lots of 3x1 plots for baseline response
    """

#%% Try a 3x1 plot
#    
#climits = {}
#climits['alb']={}
#climits['rs']={}
#climits['hc']={}
#
#climits['alb']['dlnd'] = [-5.,0.]
#climits['alb']['datm'] = [-35., 0.]
#climits['alb']['r2'] = [0.6,1.]
#
#climits['rs']['dlnd'] = [0.,20.]
#climits['rs']['datm'] = [0.,0.1]
#climits['rs']['r2'] = [0.6,1.]
#
#climits['hc']['dlnd'] = [0.,0.5]
#climits['hc']['datm'] = [0.,1.]
#climits['hc']['r2'] = [0.6,1.]
#
#cb = {}
#cb['alb']={}
#cb['rs']={}
#cb['hc']={}
#
#cb['alb']['datm'] = plt.cm.viridis_r
#cb['alb']['dlnd'] = plt.cm.plasma
#cb['alb']['dlnd'] = plt.cm.RdPu
#cb['alb']['r2'] = plt.cm.cool    
#    
#cb['rs']['datm'] = plt.cm.viridis
#cb['rs']['dlnd'] = plt.cm.plasma
#cb['rs']['dlnd'] = plt.cm.RdPu
#cb['rs']['r2'] = plt.cm.cool    
#
#cb['hc']['datm'] = plt.cm.viridis
#cb['hc']['dlnd'] = plt.cm.plasma
#cb['hc']['dlnd'] = plt.cm.RdPu
#cb['hc']['r2'] = plt.cm.cool 
#    
#var = 'TREFHT'
#prop = 'alb'
#sea = 'ANN'
#dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea]
#datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]
#r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
#
#
##-----------------------------------
## Atmospheric Variables first, online
##-----------------------------------
#
## list of atmos vars:
#atm_vars = list(slope_analysis_dict['datm_dlnd_scaled']['online']['atm'].keys())
#
## actually, just use a subset of those for now
#atm_vars = ['TREFHT','PRECIP','FSNS','MSE']
#atm_vars = ['TREFHT','SHFLX','LHFLX']
#
## Loop over variables
#for var in atm_vars:
#    
#    for prop in ['alb','rs']:
#
#        units = {}
#        
#        if prop=='alb':
#            units['units'] = 'unitless'
#        elif prop=='rs':
#            units['units'] = 's/m'
#        elif prop =='hc':
#            units['units'] = 'm'
#        
#        clims = {}
#        clims['dlnd'] = climits[prop]['dlnd']
#        clims['datm'] = climits[prop]['datm']
#        clims['r2'] = climits[prop]['r2']
#            
#        units['datm'] = '$\Delta$ var per $\Delta$ '+ np.str(slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop]['scale_factor'])
#        units['dlnd'] = ' $\Delta$ '+prop+'['+units['units']+'] per 1 increase in '+var
#        units['r2'] = '$r^2$'
#        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=1., LN=LN, LT=LT,
#                                 cb=cb,climits=clims,save_subplot=1,figpath=figpath['main'],scale=scale)


#%%
"""
    Go through and manually make 3x1s for different variables - colorbars change too much for a good loop. (my messy coding...)
"""

# Scale factors are in there for datm/dlnd -> 
# datm / d 0.1 alb
# datm / d 10 s/m rs
# datm / d 0.1 m hc
#
# but NOT there for dlnd/datm -> adjust for each variable, e.g. 0.1 T, 1 W/m2 SH, etc

prop_units = {}
prop_units['alb'] = 'unitless'
prop_units['rs'] = 's/m'
prop_units['hc'] = 'm'

cb = {}
climits = {}

#%%
#-----------------------------------
# TREFHT
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'TREFHT'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs','hc']:
        
        if prop == 'alb':
            climits['dlnd'] = [0.0,0.01]
            climits['datm'] = [0.0,2.5]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.viridis
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [0.0,25.0]
            climits['datm'] = [0.0,0.8]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu_r
            cb['datm'] = plt.cm.viridis
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units = {}
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 0.1 K increase in T2m'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea] * 0.1 * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=1.0,maskname='nomask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)


#%%
#-----------------------------------
# TREFHT
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'TREFHT'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs','hc']:
        
        if prop == 'alb':
            climits['dlnd'] = [0.0,0.01]
            climits['datm'] = [0.0,2.5]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.viridis
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [0.0,25.0]
            climits['datm'] = [0.0,0.8]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu_r
            cb['datm'] = plt.cm.viridis
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units = {}
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 0.1 K increase in T2m'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea] * 0.1 * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# cldlow
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'CLDLOW'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.01,0.01]
            climits['datm'] = [-0.075,0.075]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-25.,25.0]
            climits['datm'] = [-0.075,0.075]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units = {}
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 0.1 K increase in T2m'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea] * 0.1 * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# cldmed
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'CLDMED'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.01,0.01]
            climits['datm'] = [-0.075,0.075]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-25.,25.0]
            climits['datm'] = [-0.075,0.075]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units = {}
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 0.1 K increase in T2m'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea] * 0.1 * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
#%%
#-----------------------------------
# cldHGH
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'CLDHGH'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.2,0.2]
            climits['datm'] = [-0.075,0.075]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-25.,25.0]
            climits['datm'] = [-0.075,0.075]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units = {}
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 0.1 K increase in T2m'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea] * 0.1 * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# FSNS (for albedo, any non-uniform change comes from clouds or maybe snow)
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'FSNS'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [0.0,0.025,]
            climits['datm'] = [-10.,10.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-3.,3.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdBu_r
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea]*sign    # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale*sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# FSNSC (for albedo, any non-uniform change comes from clouds or maybe snow)
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'FSNSC'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [0.0,0.025]
            climits['datm'] = [-12.0,12.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-1.,1.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdBu_r
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea]*sign    # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale*sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

  

#%%
#-----------------------------------
# SHFLX pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'SHFLX'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-6.,6.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-2.,2.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea] * sign    # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
#        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# LHFLX pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'LHFLX'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-6.,6.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-2.,2.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea]* sign    # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
#%%
#-----------------------------------
# MSE 
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MSE'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-10.,10.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-3.,3.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea] * sign    # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
        
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# PRECIP 
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'PRECIP'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-0.3,0.3]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-0.15,0.15]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['atm'][var][prop][sea]  * sign  # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['atm'][var][prop][sea]
        
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
#                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)


#%% 
"""
Surface variables for online
"""
#%%
#-----------------------------------
# TREFHT
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_ts'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [0.0,0.01]
            climits['datm'] = [0.0,2.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.viridis
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [0.0,25.0]
            climits['datm'] = [0.0,0.8]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu_r
            cb['datm'] = plt.cm.viridis
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 0.1 K increase in T2m'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['lnd'][var][prop][sea] * 0.1 * sign  # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# FSNS (for albedo, any non-uniform change comes from clouds or maybe snow)
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_fsns'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [0.0,0.025]
            climits['datm'] = [-10.,10.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-5.,5.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdBu_r
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['lnd'][var][prop][sea] * sign    # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)


#%%
#-----------------------------------
# SHFLX pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_shflx'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-5.,5.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-1.,1.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        #dlnd = slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea]    # how much land for an 0.1 K change
        dlnd = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]**(-1) * sign
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# LHFLX pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_lhflx'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-5.,5.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-1.,1.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['lnd'][var][prop][sea] * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# EVAPFRAC pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'EVAPFRAC'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-0.2,0.2]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign  = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-0.05,0.05]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['online']['lnd'][var][prop][sea] * sign    # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['online']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# Total Turbulent flux: SHFLX + LHFLX pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'sh_plus_lh'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-10.,10.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-2.5,2.5]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = 0.0*slope_analysis_dict['dlnd_datm_scaled']['online']['lnd'][var][prop][sea] * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        # not sure I can add slopes, vs calculate on its own... should calcl on its own... 
        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale  * sign
        r2 = slope_analysis_dict['r_value']['online']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='online',var='TotalTurbulentFlux',
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%% 
"""
Surface variables, Repeat for offline
"""
#%%
#-----------------------------------
# TREFHT
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_ts'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [0.0,0.01]
            climits['datm'] = [0.0,2.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.viridis
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [0.0,25.0]
            climits['datm'] = [0.0,0.8]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu_r
            cb['datm'] = plt.cm.viridis
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 0.1 K increase in T2m'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea] * 0.1 * sign  # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# FSNS (for albedo, any non-uniform change comes from clouds or maybe snow)
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_fsns'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [0.0,0.025,]
            climits['datm'] = [-5.,5.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-1.,1.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdBu_r
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea] * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)


#%%
#-----------------------------------
# SHFLX pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_shflx'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-5.,5.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-1.,1.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        #dlnd = slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea]    # how much land for an 0.1 K change
        dlnd = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]**(-1) * sign
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# LHFLX pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_lhflx'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-5.,5.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-5.,5.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea] * sign    # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# EVAPFRAC pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'EVAPFRAC'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-0.05,0.05]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-0.05,0.05]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea] * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# Total Turbulent flux: SHFLX + LHFLX pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'sh_plus_lh'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-10.,10.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-2.5,2.5]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = 0.0*slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea] * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        # not sure I can add slopes, vs calculate on its own... should calcl on its own... 
        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale  * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='offline',var='TotalTurbulentFlux',
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
"""
    Difference in sensitivity online-offline - hard to interpret, but highlights feedback regions?
"""                
  
#%%
#-----------------------------------
# TREFHT
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_ts'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.01,0.01]
            climits['datm'] = [-2.2,2.2]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-10.,10.0]
            climits['datm'] = [-1.25,1.25]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
        
        units = {}
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 0.1 K increase in T2m'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = ( slope_analysis_dict['dlnd_datm_scaled']['online']['lnd'][var][prop][sea] 
        - slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea]) * 0.1  * sign # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = (slope_analysis_dict['slope']['online']['lnd'][var][prop][sea] - 
                slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea])*scale * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='on_m_offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='on_m_offline',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# FSNS (for albedo, any non-uniform change comes from clouds or maybe snow)
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_fsns'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [0.0,0.025]
            climits['datm'] = [-5.,5.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-5.,5.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.RdPu
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        dlnd = ( slope_analysis_dict['dlnd_datm_scaled']['online']['lnd'][var][prop][sea] 
            - slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea]) * sign   # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = (slope_analysis_dict['slope']['online']['lnd'][var][prop][sea] - 
                slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea])*scale * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='on_m_offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='on_m_offline',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#

#%%
#-----------------------------------
# SHFLX pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_shflx'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-2.,2.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-2.,2.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr
            cb['datm'] = plt.cm.RdBu_r
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = ( slope_analysis_dict['dlnd_datm_scaled']['online']['lnd'][var][prop][sea] 
        - slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea]) * sign  # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = (slope_analysis_dict['slope']['online']['lnd'][var][prop][sea] - 
                slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea])*scale * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='on_m_offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'])
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='on_m_offline',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# LHFLX pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_lhflx'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-2.,2.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-2.,2.]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = ( slope_analysis_dict['dlnd_datm_scaled']['online']['lnd'][var][prop][sea] 
        - slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea]) * sign  # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = (slope_analysis_dict['slope']['online']['lnd'][var][prop][sea] - 
                slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea])*scale * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='on_m_offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='on_m_offline',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# EVAPFRAC pay attention to SE USA
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'EVAPFRAC'
for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits['dlnd'] = [-0.015,0.015]
            climits['datm'] = [-0.1,0.1]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
        elif prop== 'rs':
            climits['dlnd'] = [-50.,50.]
            climits['datm'] = [-0.04,0.04]
            climits['r2'] = [0.6,1.0]
            
            cb['dlnd'] = plt.cm.PuOr_r
            cb['datm'] = plt.cm.RdBu
            cb['r2'] = plt.cm.viridis
            
            scale = 50.
            sign = 1.
        elif prop=='hc':
            print('uhoh')
            
        units['datm'] = '$\Delta$ var per $\Delta$ '+ prop
        units['dlnd'] = ' $\Delta$ '+prop_units[prop]+' per 1 W/m2 increase in FSNS'
        units['r2'] = '$r^2$'
        
        
        
        dlnd = ( slope_analysis_dict['dlnd_datm_scaled']['online']['lnd'][var][prop][sea] 
        - slope_analysis_dict['dlnd_datm_scaled']['offline']['lnd'][var][prop][sea]) * sign  # how much land for an 0.1 K change
        #datm = slope_analysis_dict['datm_dlnd_scaled']['online']['atm'][var][prop][sea]         # should already be scaled
        datm = (slope_analysis_dict['slope']['online']['lnd'][var][prop][sea] - 
                slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea])*scale * sign
        r2 = slope_analysis_dict['r_value']['offline']['lnd'][var][prop][sea]
        
#        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='on_m_offline',var=var,
#                                 units=units,prop=prop,sea=sea,mask=nomask,maskname='nomask', LN=LN, LT=LT,
#                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)
    
        fig, axes = sensitivity_map_3by1(datm=datm,dlnd=dlnd,r2=r2,onoff='on_m_offline',var=var,
                                 units=units,prop=prop,sea=sea,mask=bareground_mask,maskname='landmask', LN=LN, LT=LT,
                                 cb=cb,climits=climits,save_subplot=1,figpath=figpath['main'],scale=scale)


   
#%%
"""
    Make Deltas of those 3x1 plots of dlnd/datm, datm/dlnd, r^2 for nomask and landmask cases
    for ONLINE - OFFLINE (how much of this do we attribute to atm?)
    (make a fn to do this... could be a local fn...)
"""


#%%

"""
    Do plots just of North America for cloud rs responses in SE USA

"""
#%% Online
"""
 Online, atm vars
 """
#%%
#-----------------------------------
# TREFHT
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'TREFHT'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [0.0,2.5]
            
            cb = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [0.0,0.8]
            
            cb = plt.cm.viridis
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ T2m [K] per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='online',var='TREFHT',
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)


# dlnd/datm: per 0.1 change in T
var = 'MML_ts'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [0.0,2.5]
            
            cb = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [0.0,0.8]
            
            cb = plt.cm.viridis
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ T2m [K] per '+ prop
        

        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='offline',var=var,
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# SHFLX
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'SHFLX'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu_r
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu_r
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ SH [W/m2] per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)
#%%
#-----------------------------------
# LHFLX
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'LHFLX'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ LH [W/m2] per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# CLDLOW
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'CLDLOW'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [-.1,0.1]
            
            cb = plt.cm.RdBu
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-0.1,0.1]
            
            cb = plt.cm.RdBu
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ CLDLOW [fraction] per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)


#%% Online
""" 
Online, lnd vars
"""
#%%
#-----------------------------------
# TS
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_ts'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [0.0,2.5]
            
            cb = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [0.0,0.8]
            
            cb = plt.cm.viridis
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ '+var+ ' per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='online',var='TREFHT',
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)
#%%
#-----------------------------------
# fsns
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_fsns'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [-10.,10.]
            
            cb = plt.cm.RdBu_r
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-8.,8.]
            
            cb = plt.cm.RdBu_r
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ '+var+ ' per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='online',var='MML_fsns',
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)
#%%
#-----------------------------------
# SHFLX
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_shflx'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu_r
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu_r
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ '+var+ ' per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)
#%%
#-----------------------------------
# LHFLX
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_lhflx'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ '+var+ ' per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# LHFLX
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'EVAPFRAC'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [-0.2,0.2]
            
            cb = plt.cm.RdBu
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-0.05,0.05]
            
            cb = plt.cm.RdBu
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ '+var+ ' per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['lnd'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)

#%% Offline
""" 
Offline, lnd vars
"""
#%%
#-----------------------------------
# TS
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_ts'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [0.0,2.5]
            
            cb = plt.cm.viridis
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [0.0,0.8]
            
            cb = plt.cm.viridis
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ '+var+ ' per '+ prop
        

        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='offline',var='TREFHT',
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)
#%%
#-----------------------------------
# SHFLX
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_shflx'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu_r
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu_r
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$ '+var+ ' per '+ prop
        

        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='offline',var=var,
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)
#%%
#-----------------------------------
# LHFLX
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'MML_lhflx'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-2.5,2.5]
            
            cb = plt.cm.RdBu
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = units = ' $\Delta$ '+var+ ' per '+ prop
        

        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='offline',var=var,
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# LHFLX
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'EVAPFRAC'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['rs']:
        
        if prop == 'alb':
            climits= [-.2,.2]
            
            cb = plt.cm.RdBu
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-.05,.05]
            
            cb = plt.cm.RdBu
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = units = ' $\Delta$ '+var+ ' per '+ prop
        

        datm = slope_analysis_dict['slope']['offline']['lnd'][var][prop][sea]*scale * sign


        mml_map_NA_local(datm=datm,onoff='offline',var=var,
                     units=units,prop=prop,sea=sea,mask=landmask,maskname='landmask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)


#%%
"""
    Africa plots for precip - albedo - rs
"""

#%% loop over online, offline

#%%
#-----------------------------------
# CLDLOW
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'CLDLOW'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits= [-.1,0.1]
            
            cb = plt.cm.RdBu
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-0.1,0.1]
            
            cb = plt.cm.RdBu
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$  '+var+' per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign


        mml_map_Africa_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=1.0,maskname='nomask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)
#%%
#-----------------------------------
# CLDLOW
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'PRECIP'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits= [-.4,0.4]
            
            cb = plt.cm.RdBu
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-0.2,0.2]
            
            cb = plt.cm.RdBu
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$  '+var+' per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign


        mml_map_Africa_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=1.0,maskname='nomask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# CLDLOW
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'LHFLX'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits= [-5,5]
            
            cb = plt.cm.RdBu
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-2,2]
            
            cb = plt.cm.RdBu
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$  '+var+' per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign


        mml_map_Africa_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=1.0,maskname='nomask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# CLDLOW
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'SHFLX'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits= [-5,5]
            
            cb = plt.cm.RdBu_r
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-2,2]
            
            cb = plt.cm.RdBu_r
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$  '+var+' per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign


        mml_map_Africa_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=1.0,maskname='nomask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)

#%%
#-----------------------------------
# CLDLOW
#----------------------------------

# dlnd/datm: per 0.1 change in T
var = 'TREFHT'
del cb, climits, units

for sea in ['ANN','DJF','JJA']:#,'DJF','JJA']:
    for prop in ['alb','rs']:
        
        if prop == 'alb':
            climits= [-2,2]
            
            cb = plt.cm.RdBu_r
            
            scale = 0.1
            sign = -1.
            
        elif prop== 'rs':
            climits= [-1,1]
            
            cb = plt.cm.RdBu_r
            
            scale = 50.
            sign = 1.
            
        elif prop=='hc':
            print('uhoh')
            
        units = ' $\Delta$  '+var+' per '+ prop
        

        datm = slope_analysis_dict['slope']['online']['atm'][var][prop][sea]*scale * sign


        mml_map_Africa_local(datm=datm,onoff='online',var=var,
                     units=units,prop=prop,sea=sea,mask=1.0,maskname='nomask', 
                     LN=LN, LT=LT,
                     cb=cb,climits=climits,save_subplot=1,
                     figpath=figpath['main'],scale=scale)



#%%
"""
    Make some scatter plots (make a fn to do so)
"""


#%% Really struggling...

#Surface T
ts_online = response_dict['lnd']['online']['MML_ts']['alb']['ANN']

delta_ts_online = ts_online[0,:,:]-ts_online[2,:,:]

ts_offline = response_dict['lnd']['offline']['MML_ts']['alb']['ANN']

delta_ts_offline = ts_offline[0,:,:]-ts_offline[2,:,:]

plt.imshow(delta_ts_online,clim=[-0.5,5])
plt.colorbar()
plt.show()

plt.imshow(delta_ts_offline,clim=[-0.5,5])
plt.colorbar()
plt.show()

on_m_off_ts = delta_ts_online - delta_ts_offline

plt.imshow(on_m_off_ts,cmap=plt.cm.RdBu_r,clim=[-5,5])
plt.colorbar()
plt.show()

# SHFLX
ts_online = response_dict['lnd']['online']['MML_shflx']['alb']['ANN']

delta_ts_online = ts_online[0,:,:]-ts_online[2,:,:]

ts_offline = response_dict['lnd']['offline']['MML_shflx']['alb']['ANN']

delta_ts_offline = ts_offline[0,:,:]-ts_offline[2,:,:]

plt.imshow(delta_ts_online,clim=[-5,10])
plt.colorbar()
plt.show()

plt.imshow(delta_ts_offline,clim=[-5,10])
plt.colorbar()
plt.show()

on_m_off_ts = delta_ts_online - delta_ts_offline

plt.imshow(on_m_off_ts,cmap=plt.cm.RdBu_r,clim=[-10,10])
plt.colorbar()
plt.show()
#%%
"""
    Select some locations to do a bar-chart surface energy budget analysis on. 
    Make fn to plot bar chart. ... Even make fn to do sfc E budget analysis... 
"""

# do land vars for easy on/offline comparison
sfc_e_vars = {}
sfc_e_vars['atm'] = ['SWCF','LWCF']
sfc_e_vars['lnd'] = ['MML_fsns','MML_lhflx','MML_shflx','MML_ts','MML_fsns',
          'MML_flns','MML_lwdn','MML_lwup','MML_fsnsc'] # MML_flns

# Do bar charts for max - min experiment... e.g. darkest - lightest in albedo, most - least resistnace in rs
bar_dat = {}
bar_dat['alb'] = {}
bar_dat['rs'] = {}
bar_dat['alb']['dark'] = {}
bar_dat['alb']['light'] = {}
bar_dat['rs']['high'] = {}
bar_dat['rs']['low'] = {}


location = {}

tag = 'PNW'
location[tag] = {}
location[tag]['lat_bounds'] = [44,50]
location[tag]['lon_bounds'] = [240,250]
location[tag]['color'] = 'forestgreen'

tag = 'AMZ'
location[tag] = {}
location[tag] = {}
location[tag]['lat_bounds'] = [-10,0]
location[tag]['lon_bounds'] = [293,305]
location[tag]['color'] = 'lime'

tag = 'SE_USA'
location[tag] = {}
location[tag] = {}
location[tag]['lat_bounds'] = [32,37]
location[tag]['lon_bounds'] = [270,280]
location[tag]['color'] = 'hotpink'

tag = 'SIB'
location[tag] = {}
location[tag] = {}
location[tag]['lat_bounds'] = [63,70]
location[tag]['lon_bounds'] = [106,124]
location[tag]['color'] = 'deepskyblue'

tag = 'RUSSIA'
location[tag] = {}
location[tag] = {}
location[tag]['lat_bounds'] = [55,65]
location[tag]['lon_bounds'] = [55,75]
location[tag]['color'] = 'red'

tag = 'SAHARA'
location[tag] = {}
location[tag] = {}
location[tag]['lat_bounds'] = [16,26]
location[tag]['lon_bounds'] = [0.1,23]
location[tag]['color'] = 'darkorange'

#%%
# Get deltas for max-min experiment at the above locaitons
scales = {}
scales['alb'] = 0.1
scales['rs'] = 50.
for onoff in ['online','offline']:
#    bar_dat['alb']['dark'][onoff] = {}
#    bar_dat['alb']['light'][onoff]  = {}
#    bar_dat['rs']['high'][onoff]  = {}
#    bar_dat['rs']['low'][onoff]  = {}
    bar_dat[onoff]  = {}

    for prop in ['alb','rs']: #['alb','rs']
        
        bar_dat[onoff][prop]  = {}
    
        for var in sfc_e_vars['lnd']:
            
            bar_dat[onoff][prop][var]  = {}
            
            if var == 'MML_ts':
                bar_dat[onoff][prop]['sigT4'] = {}
            
            for loc in list(location.keys()):
                
                lat_bounds = location[loc]['lat_bounds']
                lon_bounds = location[loc]['lon_bounds']
                
                
                
                bar_dat[onoff][prop][var][loc]  = {}
                
                if var =='MML_ts':
                    bar_dat[onoff][prop]['sigT4'][loc] = {}
            
                for sea in ['ANN','DJF','JJA']:
                    
#                    raw_dat = response_dict['lnd'][onoff][var][prop][sea]
#                    
#                    bar_dat[onoff][prop][var][loc][sea]  = ( avg_over_box(dat_to_avg=raw_dat[2,:,:], 
#                                                           area_grid=area_f19, lat=lat, lon=lon, 
#                                                           lat_bounds=lat_bounds, lon_bounds=lon_bounds) - 
#                                                        avg_over_box(dat_to_avg=raw_dat[1,:,:], 
#                                                           area_grid=area_f19, lat=lat, lon=lon, 
#                                                           lat_bounds=lat_bounds, lon_bounds=lon_bounds)  )
                    if var == 'MML_fsnsc':
                        raw_dat = slope_analysis_dict['slope']['online']['atm']['FSNSC'][prop][sea] * scales[prop]
                    
                    else:
                        raw_dat = slope_analysis_dict['slope'][onoff]['lnd'][var][prop][sea] * scales[prop]
                    
                    bar_dat[onoff][prop][var][loc][sea]  = avg_over_box(dat_to_avg=raw_dat, 
                                                           area_grid=area_f19, lat=lat, lon=lon, 
                                                           lat_bounds=lat_bounds, lon_bounds=lon_bounds)
                    
                    if var == 'MML_ts':
                        raw_dat = response_dict['lnd'][onoff][var][prop][sea]
                        sigma = 5.67e-8
                        bar_dat[onoff][prop]['sigT4'][loc][sea]   = ( sigma*(avg_over_box(dat_to_avg=raw_dat[2,:,:], 
                                                           area_grid=area_f19, lat=lat, lon=lon, 
                                                           lat_bounds=lat_bounds, lon_bounds=lon_bounds)**4) - 
                                                                sigma*(avg_over_box(dat_to_avg=raw_dat[1,:,:], 
                                                           area_grid=area_f19, lat=lat, lon=lon, 
                                                           lat_bounds=lat_bounds, lon_bounds=lon_bounds))**4  )

#%% Make bar graphs                    

# Just FLNS (not separated; separated below)
# Online & offline on the same plot; do fsns, sigT^4, lhflx, shsflx. These are all deltas of 2-0, so flip sign on albedo to get dark-light
   
for sea in ['ANN','DJF','JJA']:
             
    # albedo:
    
    prop = 'alb'
    
    for loc in list(location.keys()):
        
        # make negative for albedo so we're doing dark-light
        
        # Online sfc e budget:
        fsns_on = -bar_dat['online'][prop]['MML_fsns'][loc][sea]
        flns_on = -bar_dat['online'][prop]['MML_flns'][loc][sea]
        sigT4_on = -bar_dat['online'][prop]['sigT4'][loc][sea]
        #lw_down_on = bar_dat['online'][prop]['MML_lwdn'][loc][sea]
        #lw_down_off = bar_dat['online'][prop]['MML_lwup'][loc][sea]
        lhflx_on = -bar_dat['online'][prop]['MML_lhflx'][loc][sea]
        shflx_on  = -bar_dat['online'][prop]['MML_shflx'][loc][sea]
        
        # Offline sfc e budget:
        fsns_off = -bar_dat['offline'][prop]['MML_fsns'][loc][sea]
        flns_off = -bar_dat['offline'][prop]['MML_flns'][loc][sea]
        #lw_down_off = bar_dat['offline'][prop]['MML_lwdn'][loc][sea]
        #lw_up_off = bar_dat['offline'][prop]['MML_lwup'][loc][sea]
        sigT4_off = -bar_dat['offline'][prop]['sigT4'][loc][sea]
        lhflx_off = -bar_dat['offline'][prop]['MML_lhflx'][loc][sea]
        shflx_off  = -bar_dat['offline'][prop]['MML_shflx'][loc][sea]
                
        # Bar chart:
        N = 4   # sfc e budget terms
        #N = 5
        
        energy_on = (fsns_on, flns_on, lhflx_on, shflx_on)
        energy_off = (fsns_off, flns_off, lhflx_off, shflx_off)
       # energy_on = (fsns_on, lw_down_on,lw_up_on, lhflx_on, shflx_on)
       # energy_off = (fsns_off, lw_down_off,lw_up_off, lhflx_off, shflx_off)
        
        ind = np.arange(N)
        width=0.35
        
        fig, ax = plt.subplots()
        
        # figure out individual colours later
        bars_on = ax.bar(ind,energy_on,width,color='b')        
        
        bars_off = ax.bar(ind+width,energy_off,width,color='g')
        
        # add some text for labels, title and axes ticks
        ax.set_ylabel('$\delta$ Energy [W/m$^2$] per '+np.str(scales[prop])+' $\delta$ '+prop)
        ax.set_title('Change in surface energy fluxes, dark - light albedo experiment at \n'+loc+', '+sea,y=1.05)
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('Net SW', 'Net LW', 'LH', 'SH'))
#        ax.set_xticklabels(('Net SW', 'LW down','LW up', 'LH', 'SH'))
        
        ax.legend((bars_on[0], bars_off[0]), ('Coupled', 'Offline'))
        
        xlim = ax.get_xlim()
        ax.plot([xlim[0],xlim[-1]],[0,0],"k-")
        
        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%.1f' % (height),
                        ha='center', va='bottom')
    
#        autolabel(bars_on)
#        autolabel(bars_off)
        
        
        # Annotate with season, variable, date
        plt.text(0.,-0.25,time.strftime("%x")+', ' + loc + ', ' + prop+', '+sea+'_flns',fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        # Save to sfc_bar_plots folder:
        filename = 'sfc_e_' + prop + '_' + loc + '_'+sea+'_onoff'
        
        plt.show()
        
        fig_name = figpath['main']+'/sfc_bar_plots/'+filename+'.png'
        fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)
        
        plt.close()
    
       
    # evap rs:
    
    prop = 'rs'
    
    for loc in list(location.keys()):
        
        # make negative for albedo so we're doing dark-light
        
        # Online sfc e budget:
        fsns_on = bar_dat['online'][prop]['MML_fsns'][loc][sea]
        flns_on = bar_dat['online'][prop]['MML_flns'][loc][sea]
        sigT4_on = bar_dat['online'][prop]['sigT4'][loc][sea]
        lhflx_on = bar_dat['online'][prop]['MML_lhflx'][loc][sea]
        shflx_on  = bar_dat['online'][prop]['MML_shflx'][loc][sea]
        
        # Offline sfc e budget:
        fsns_off = bar_dat['offline'][prop]['MML_fsns'][loc][sea]
        flns_off = bar_dat['offline'][prop]['MML_flns'][loc][sea]
        sigT4_off = bar_dat['offline'][prop]['sigT4'][loc][sea]
        lhflx_off = bar_dat['offline'][prop]['MML_lhflx'][loc][sea]
        shflx_off  = bar_dat['offline'][prop]['MML_shflx'][loc][sea]
                
        # Bar chart:
        N = 4   # sfc e budget terms
        
        
        energy_on = (fsns_on, flns_on, lhflx_on, shflx_on)
        energy_off = (fsns_off, flns_off, lhflx_off, shflx_off)
        
        ind = np.arange(N)
        width=0.35
        
        fig, ax = plt.subplots()
        
        # figure out individual colours later
        bars_on = ax.bar(ind,energy_on,width,color='b')        
        
        bars_off = ax.bar(ind+width,energy_off,width,color='g')
        
        # add some text for labels, title and axes ticks
        #ax.set_ylabel('$\Delta$ Energy [W/m$^2$]')
        ax.set_ylabel('$\delta$ Energy [W/m$^2$] per '+np.str(scales[prop])+' $\delta$ '+prop)
        ax.set_title('Change in surface energy fluxes, high - low resistance experiment at \n'+loc+', '+sea,y=1.05)
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('Net SW', 'Net LW', 'LH', 'SH'))
        
        ax.legend((bars_on[0], bars_off[0]), ('Coupled', 'Offline'))
        
        xlim = ax.get_xlim()
        ax.plot([xlim[0],xlim[-1]],[0,0],"k-")
        
        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%.1f' % (height),
                        ha='center', va='bottom')
    
#        autolabel(bars_on)
#        autolabel(bars_off)
        
        
        # Annotate with season, variable, date
        plt.text(0.,-0.25,time.strftime("%x")+', ' + loc + ', ' + prop+', '+sea+'_flns',fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        # Save to sfc_bar_plots folder:
        filename = 'sfc_e_' + prop + '_' + loc + '_'+sea+'_onoff'
        
        plt.show()
        
        fig_name = figpath['main']+'/sfc_bar_plots/'+filename+'.png'
        fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)
        
        plt.close()

#%% fsns and fsnsc - show cloud portion
# Just FLNS (not separated; separated below)
# Online & offline on the same plot; do fsns, sigT^4, lhflx, shsflx. These are all deltas of 2-0, so flip sign on albedo to get dark-light
   
for sea in ['ANN','DJF','JJA']:
             
    # albedo:
    
    prop = 'alb'
    
    for loc in list(location.keys()):
        
        # make negative for albedo so we're doing dark-light
        
        # Online sfc e budget:
        fsns_on = -bar_dat['online'][prop]['MML_fsns'][loc][sea]
        fsnsc_on = -bar_dat['online'][prop]['MML_fsnsc'][loc][sea]
        flns_on = -bar_dat['online'][prop]['MML_flns'][loc][sea]
        sigT4_on = -bar_dat['online'][prop]['sigT4'][loc][sea]
        #lw_down_on = bar_dat['online'][prop]['MML_lwdn'][loc][sea]
        #lw_down_off = bar_dat['online'][prop]['MML_lwup'][loc][sea]
        lhflx_on = -bar_dat['online'][prop]['MML_lhflx'][loc][sea]
        shflx_on  = -bar_dat['online'][prop]['MML_shflx'][loc][sea]
    
                
        # Bar chart:
        N = 5   # sfc e budget terms
        #N = 5
        
        energy_on = (fsns_on, fsnsc_on, flns_on, lhflx_on, shflx_on)
#        energy_off = (fsns_off, fsnsc_off, flns_off, lhflx_off, shflx_off)
       # energy_on = (fsns_on, lw_down_on,lw_up_on, lhflx_on, shflx_on)
       # energy_off = (fsns_off, lw_down_off,lw_up_off, lhflx_off, shflx_off)
        
        ind = np.arange(N)
        width=0.35
        
        fig, ax = plt.subplots()
        
        # figure out individual colours later
        bars_on = ax.bar(ind,energy_on,width,color='b')        
        
        
        xlim = ax.get_xlim()
        ax.plot([xlim[0],xlim[-1]],[0,0],"k-")
        
#        bars_off = ax.bar(ind+width,energy_off,width,color='g')
        
        # add some text for labels, title and axes ticks
        ax.set_ylabel('$\delta$ Energy [W/m$^2$] per '+np.str(scales[prop])+' $\delta$ '+prop)
        ax.set_title('Change in surface energy fluxes, dark - light albedo experiment at \n'+loc+', '+sea,y=1.05)
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('Net SW', 'Clearsky SW','Net LW', 'LH', 'SH'))
#        ax.set_xticklabels(('Net SW', 'LW down','LW up', 'LH', 'SH'))
        
#        ax.legend((bars_on[0], bars_off[0]), ('Coupled', 'Offline'))
#        ax.legend((bars_on[0]), ('Coupled'))
        
        
        
        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%.1f' % (height),
                        ha='center', va='bottom')
    
#        autolabel(bars_on)
#        autolabel(bars_off)
        
        
        # Annotate with season, variable, date
        plt.text(0.,-0.25,time.strftime("%x")+', ' + loc + ', ' + prop+', '+sea+'_flns_fsnsc',fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        # Save to sfc_bar_plots folder:
        filename = 'sfc_e_' + prop + '_' + loc + '_'+sea+'_onoff_flns_fsnsc'
        
        plt.show()
        
        fig_name = figpath['main']+'/sfc_bar_plots/'+filename+'.png'
        fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)
        
        plt.close()
    
       
    # evap rs:
    
    prop = 'rs'
    
    for loc in list(location.keys()):
        
        # make negative for albedo so we're doing dark-light
        
        # Online sfc e budget:
        fsns_on = bar_dat['online'][prop]['MML_fsns'][loc][sea]
        fsnsc_on = bar_dat['online'][prop]['MML_fsnsc'][loc][sea]
        flns_on = bar_dat['online'][prop]['MML_flns'][loc][sea]
        sigT4_on = bar_dat['online'][prop]['sigT4'][loc][sea]
        lhflx_on = bar_dat['online'][prop]['MML_lhflx'][loc][sea]
        shflx_on  = bar_dat['online'][prop]['MML_shflx'][loc][sea]
        

                
        # Bar chart:
        N = 5   # sfc e budget terms
        
        
        energy_on = (fsns_on, fsnsc_on, flns_on, lhflx_on, shflx_on)
#        energy_off = (fsns_off, fsnsc_off, flns_off, lhflx_off, shflx_off)
        
        ind = np.arange(N)
        width=0.35
        
        fig, ax = plt.subplots()
        
        # figure out individual colours later
        bars_on = ax.bar(ind,energy_on,width,color='b')        
        
#        bars_off = ax.bar(ind+width,energy_off,width,color='g')
        
        # add some text for labels, title and axes ticks
        #ax.set_ylabel('$\Delta$ Energy [W/m$^2$]')
        ax.set_ylabel('$\delta$ Energy [W/m$^2$] per '+np.str(scales[prop])+' $\delta$ '+prop)
        ax.set_title('Change in surface energy fluxes, high - low resistance experiment at \n'+loc+', '+sea,y=1.05)
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('Net SW', 'Net LW', 'LH', 'SH'))
        
#        ax.legend((bars_on[0], bars_off[0]), ('Coupled', 'Offline'))
#        ax.legend((bars_on[0]), ('Coupled'))
        
        xlim = ax.get_xlim()
        ax.plot([xlim[0],xlim[-1]],[0,0],"k-")
        
        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%.1f' % (height),
                        ha='center', va='bottom')
    
#        autolabel(bars_on)
#        autolabel(bars_off)
        
        
        # Annotate with season, variable, date
        plt.text(0.,-0.25,time.strftime("%x")+', ' + loc + ', ' + prop+', '+sea+'_flns_fsnsc',fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        # Save to sfc_bar_plots folder:
        filename = 'sfc_e_' + prop + '_' + loc + '_'+sea+'_onoff_flns_fsnsc'
        
        plt.show()
        
        fig_name = figpath['main']+'/sfc_bar_plots/'+filename+'.png'
        fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)
        
        plt.close()

#%%
# Separte flns into lwdown and lwup
        
# Online & offline on the same plot; do fsns, sigT^4, lhflx, shsflx. These are all deltas of 2-0, so flip sign on albedo to get dark-light
   
for sea in ['ANN','DJF','JJA']:
             
    # albedo:
    
    prop = 'alb'
    
    for loc in list(location.keys()):
        
        # make negative for albedo so we're doing dark-light
        
        # Online sfc e budget:
        fsns_on = -bar_dat['online'][prop]['MML_fsns'][loc][sea]
        flns_on = -bar_dat['online'][prop]['MML_flns'][loc][sea]
        sigT4_on = -bar_dat['online'][prop]['sigT4'][loc][sea]
        lw_down_on = -bar_dat['online'][prop]['MML_lwdn'][loc][sea]
        lw_up_on = -bar_dat['online'][prop]['MML_lwup'][loc][sea]
        lhflx_on = -bar_dat['online'][prop]['MML_lhflx'][loc][sea]
        shflx_on  = -bar_dat['online'][prop]['MML_shflx'][loc][sea]
        
        # Offline sfc e budget:
        fsns_off = -bar_dat['offline'][prop]['MML_fsns'][loc][sea]
        flns_off = -bar_dat['offline'][prop]['MML_flns'][loc][sea]
        lw_down_off = -bar_dat['offline'][prop]['MML_lwdn'][loc][sea]
        lw_up_off = -bar_dat['offline'][prop]['MML_lwup'][loc][sea]
        sigT4_off = -bar_dat['offline'][prop]['sigT4'][loc][sea]
        lhflx_off = -bar_dat['offline'][prop]['MML_lhflx'][loc][sea]
        shflx_off  = -bar_dat['offline'][prop]['MML_shflx'][loc][sea]
                
        # Bar chart:
        #N = 4   # sfc e budget terms
        N = 5
        
#        energy_on = (fsns_on, flns_on, lhflx_on, shflx_on)
#        energy_off = (fsns_off, flns_off, lhflx_off, shflx_off)
        energy_on = (fsns_on, lw_down_on,lw_up_on, lhflx_on, shflx_on)
        energy_off = (fsns_off, lw_down_off,lw_up_off, lhflx_off, shflx_off)
        
        ind = np.arange(N)
        width=0.35
        
        fig, ax = plt.subplots()
        
        # figure out individual colours later
        bars_on = ax.bar(ind,energy_on,width,color='b')        
        
        bars_off = ax.bar(ind+width,energy_off,width,color='g')
        
        # add some text for labels, title and axes ticks
        ax.set_ylabel('$\delta$ Energy [W/m$^2$] per '+np.str(scales[prop])+' $\delta$ '+prop)
        ax.set_title('Change in surface energy fluxes, dark - light albedo experiment at \n'+loc+', '+sea,y=1.05)
        ax.set_xticks(ind + width / 2)
        #ax.set_xticklabels(('Net SW', 'Net LW', 'LH', 'SH'))
        ax.set_xticklabels(('Net SW', 'LW down','LW up', 'LH', 'SH'))
        
        ax.legend((bars_on[0], bars_off[0]), ('Coupled', 'Offline'))
        
        xlim = ax.get_xlim()
        ax.plot([xlim[0],xlim[-1]],[0,0],"k-")
        
        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%.1f' % (height),
                        ha='center', va='bottom')
    
        #autolabel(bars_on)
        #autolabel(bars_off)
        
        
        # Annotate with season, variable, date
        plt.text(0.,-0.25,time.strftime("%x")+', ' + loc + ', ' + prop+', '+sea+' lw_separated',fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        # Save to sfc_bar_plots folder:
        filename = 'sfc_e_' + prop + '_' + loc + '_'+sea+'_onoff_lw_separated'
        
        plt.show()
        
        fig_name = figpath['main']+'/sfc_bar_plots/'+filename+'.png'
        fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)
        
        plt.close()
    
       
    # evap rs:
    
    prop = 'rs'
    
    for loc in list(location.keys()):
        
        # make negative for albedo so we're doing dark-light
        
        # Online sfc e budget:
        fsns_on = bar_dat['online'][prop]['MML_fsns'][loc][sea]
        flns_on = bar_dat['online'][prop]['MML_flns'][loc][sea]
        sigT4_on = bar_dat['online'][prop]['sigT4'][loc][sea]
        lw_down_on = bar_dat['online'][prop]['MML_lwdn'][loc][sea]
        lw_down_off = bar_dat['online'][prop]['MML_lwup'][loc][sea]
        lhflx_on = bar_dat['online'][prop]['MML_lhflx'][loc][sea]
        shflx_on  = bar_dat['online'][prop]['MML_shflx'][loc][sea]
        
        # Offline sfc e budget:
        fsns_off = bar_dat['offline'][prop]['MML_fsns'][loc][sea]
        flns_off = bar_dat['offline'][prop]['MML_flns'][loc][sea]
        lw_down_off = bar_dat['offline'][prop]['MML_lwdn'][loc][sea]
        lw_up_off = bar_dat['offline'][prop]['MML_lwup'][loc][sea]
        sigT4_off = bar_dat['offline'][prop]['sigT4'][loc][sea]
        lhflx_off = bar_dat['offline'][prop]['MML_lhflx'][loc][sea]
        shflx_off  = bar_dat['offline'][prop]['MML_shflx'][loc][sea]
                
        # Bar chart:
        #N = 4   # sfc e budget terms
        N = 5
        
#        energy_on = (fsns_on, flns_on, lhflx_on, shflx_on)
#        energy_off = (fsns_off, flns_off, lhflx_off, shflx_off)
        energy_on = (fsns_on, lw_down_on,lw_up_on, lhflx_on, shflx_on)
        energy_off = (fsns_off, lw_down_off,lw_up_off, lhflx_off, shflx_off)
        
        ind = np.arange(N)
        width=0.35
        
        fig, ax = plt.subplots()
        
        # figure out individual colours later
        bars_on = ax.bar(ind,energy_on,width,color='b')        
        
        bars_off = ax.bar(ind+width,energy_off,width,color='g')
        
        # add some text for labels, title and axes ticks
        #ax.set_ylabel('$\Delta$ Energy [W/m$^2$]')
        ax.set_ylabel('$\delta$ Energy [W/m$^2$] per '+np.str(scales[prop])+' $\delta$ '+prop)
        ax.set_title('Change in surface energy fluxes, high - low resistance experiment at \n'+loc+', '+sea,y=1.05)
        ax.set_xticks(ind + width / 2)
        #ax.set_xticklabels(('Net SW', 'Net LW', 'LH', 'SH'))
        ax.set_xticklabels(('Net SW', 'LW down','LW up', 'LH', 'SH'))
        
        ax.legend((bars_on[0], bars_off[0]), ('Coupled', 'Offline'))
        
        xlim = ax.get_xlim()
        ax.plot([xlim[0],xlim[-1]],[0,0],"k-")
        
        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%.1f' % (height),
                        ha='center', va='bottom')
    
        #autolabel(bars_on)
        #autolabel(bars_off)
        
        
        # Annotate with season, variable, date
        plt.text(0.,-0.25,time.strftime("%x")+', ' + loc + ', ' + prop+', '+sea+' lw_separated',fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax.transAxes)
    
        # Save to sfc_bar_plots folder:
        filename = 'sfc_e_' + prop + '_' + loc + '_'+sea+'_onoff_lw_separated'
        
        plt.show()
        
        fig_name = figpath['main']+'/sfc_bar_plots/'+filename+'.png'
        fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)
        
        plt.close()
#%%

    
#%% Plot locations

# draw box on map
fig, ax = plt.subplots(1, 1, figsize=(6,4))
mp = Basemap(resolution='l',projection='robin',lon_0=-180)#, lat_ts=50,lat_0=50,lon_0=-107,width=10000000,height=8000000)
mp.drawcoastlines()
mp.drawmapboundary(fill_color='1.')  # make map background white
mp.drawparallels(np.arange(-80.,81.,20.))
mp.drawmeridians(np.arange(-180.,181.,20.))

mp_ax = {}

for loc in list(location.keys()):
                
    lat_bounds = location[loc]['lat_bounds']
    lon_bounds = location[loc]['lon_bounds']
    
    
            
    mp_ax[loc] = draw_box(mp_ax=mp,lat_bounds=lat_bounds,lon_bounds=lon_bounds,lat=lat,lon=lon,line_col=location[loc]['color'],label=loc)
    
    
plt.legend(bbox_to_anchor=(1.35,1.1))
    
# Annotate with season, variable, date
plt.text(0.,-0.05,time.strftime("%x"),fontsize='10',
             ha = 'left',va = 'center',
             transform = ax.transAxes)

# Save to sfc_bar_plots folder:
filename = 'sfc_e_locations'

plt.show()

fig_name = figpath['main']+'/sfc_bar_plots/'+filename+'.png'
fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
plt.close()
    
    # save figure


#%% Plot landmask

# draw box on map
fig, ax = plt.subplots(1, 1, figsize=(6,4))
mp = Basemap(resolution='l',projection='robin',lon_0=-180)#, lat_ts=50,lat_0=50,lon_0=-107,width=10000000,height=8000000)
mp.drawcoastlines()
mp.drawmapboundary(fill_color='1.')  # make map background white
mp.drawparallels(np.arange(-80.,81.,20.))
mp.drawmeridians(np.arange(-180.,181.,20.))

mapdata = bareground_mask
mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
cs = mp.pcolormesh(LN,LT,mapdata,cmap=plt.cm.Greens,latlon=True)

cbar = mp.colorbar(cs,location='bottom',pad="5%",spacing='uniform')#,format='%.3f')

cs.cmap.set_bad('white',1.)

clim = [0,3]

cbar.set_clim(clim[0],clim[1])
cs.set_clim(clim[0],clim[1])

# Save to sfc_bar_plots folder:
filename = 'bareground_mask_'+'light'

plt.show()

fig_name = figpath['main']+'/'+filename+'.png'
fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
plt.close()
    
    # save figure


#%%
    
"""
    Just for fun, lets see what happens if we take anual mean background T
    and test how much dT is needed for a 1 W/m2 LW change using sig T^4. What is THAT pattern?
    
    Then, use actual fsnsc change and again back out how much dT would be needed.
"""

# use avg T from a2 run
T_avg= response_dict['lnd']['online']['MML_ts']['alb']['ANN'][1,:,:]
print(np.shape(T_avg))

dLW = 1.0   # W/m2 - see how much dT is needed for a 1 W/m2 change in energy, depending on background T

# LW = sig T^4
# dLW = 4 sig T^3 * dT
sig = 5.67e-8

dT = dLW / (3 * sig * T_avg**3)

plt.imshow(T_avg)
plt.colorbar()
plt.title('Background annual mean Ts')
plt.show()
plt.close()

plt.imshow(dT)
plt.colorbar()
plt.title('dTs required to drive a 1 W/m2 change in LW, given background Ts')
plt.show()
plt.close()

cmap = plt.cm.viridis

clim = [230,310]

ttl = 'Annual mean Ts for a2 [K]'
filename = 'Ts_ann_a2_global_landmask'

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=T_avg,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)

clim = [0.2,0.5]
ttl = 'dT [K] required for a 1 W/m2 change in LW up'
filename = 'dTs_ann_for_1Wm2_a2_global_landmask'

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=dT,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)

#%%
"""
    Now use the avg dFSNSC clearsky instead of 1 W/m2, avgd between a3->a2 and a2->a1, 
    then est dT required to offset that
"""

T_avg= response_dict['lnd']['online']['MML_ts']['alb']['ANN'][1,:,:]

dLW = (  ( response_dict['atm']['online']['FSNSC']['alb']['ANN'][2,:,:]-
          response_dict['atm']['online']['FSNSC']['alb']['ANN'][1,:,:]) + 
         ( response_dict['atm']['online']['FSNSC']['alb']['ANN'][1,:,:]-
          response_dict['atm']['online']['FSNSC']['alb']['ANN'][0,:,:])  )/2   # W/m2 - see how much dT is needed for a 1 W/m2 change in energy, depending on background T

# LW = sig T^4
# dLW = 4 sig T^3 * dT
sig = 5.67e-8

dT = dLW / (3 * sig * T_avg**3)
#
#%%
#"""
#    Plot delta T2m between a few different roughness simulations
#"""
#
## Start by plotting r2 of roughness regression
#
#mapdata = slope_analysis_dict['r_value']['online']['atm']['TREFHT']['hc']['ANN'][:]
#
#filename = 'hc_r2_nomask'
#cmap = plt.cm.viridis
#clim = [0,1]
#ttl = "r^2 value of hc linear fit"
#
#mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=mapdata,
#                                         units='K',prop=None,sea=None,mask=np.ones(np.shape(landmask)),maskname='none', 
#                                         LN=LN, LT=LT,
#                                         cb=cmap,climits=clim,save_subplot=None,
#                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)
#
#
## 1.0 m - 20 m
#
#
#
#
#
#
#
#
#
## 0.1 m - 1.0 m (should have same sign as 1.0 - 20 if the linearity is holding)
#





#%%
