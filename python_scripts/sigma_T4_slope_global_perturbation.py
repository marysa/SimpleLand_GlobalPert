#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:34:13 2017

@author: mlague
"""

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
import brewer2mpl as cbrew
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

load_location = dict_path + 'glbl_pert_dict_file_test_oct30_pickle'

with open(load_location, 'rb') as f:
    slope_analysis_dict = pickle.load(f)

load_location = dict_path + 'response_dict_file_test_oct30_pickle'

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
    
    mapdata = np.ma.masked_where(np.isnan(mapdata),mapdata)
    
    
    plt.sca(ax0)
   
    
    cm = cb
    if climits:
        clim = climits
    else:
        clim = [-1,1]
    unit = units
    
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

clim = [0.2,0.32]
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
cmap = plt.cm.viridis

#
clim = [230,310]

ttl = 'Annual mean Ts for a2 [K]'
filename = 'Ts_ann_a2_global_landmask'

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=T_avg,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)


#
clim = [2,13.5]

ttl = 'Annual mean change in FSNSC for an 0.1 darkening in alb'

filename = 'dFSNSC_ann_dalb0.1_global_landmask'

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=dLW,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)

#
clim = [1.5,3.2]

ttl = 'Annual mean change in Ts required to balance \n dFSNSC for an 0.1 darkening in alb'

filename = 'dTs_for_dFSNSC_ann_dalb0.1_global_landmask'

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=dT,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)



#%%
#%%
"""
    Now use 0.1*FSDS instead of 1 W/m2, as an approximation about the mean value in the a2 case, 
    then est dT required to offset that
"""

T_avg= response_dict['lnd']['online']['MML_ts']['alb']['ANN'][1,:,:]

dLW =  0.1*response_dict['atm']['online']['FSDS']['alb']['ANN'][1,:,:]
             # W/m2 - see how much dT is needed for a 1 W/m2 change in energy, depending on background T

# LW = sig T^4
# dLW = 4 sig T^3 * dT
sig = 5.67e-8

dT = dLW / (3 * sig * T_avg**3)

#
cmap = plt.cm.viridis

#
clim = [230,310]

ttl = 'Annual mean Ts for a2 [K]'
filename = 'Ts_ann_a2_global_landmask'

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=T_avg,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)


#
clim = [0,14]

ttl = 'Annual mean change in FSNSC for an 0.1 darkening in alb'

filename = '0.1FSDS_ann_global_landmask'

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=dLW,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)

#
clim = [1.5,3.2]

ttl = 'Annual mean change in Ts required to balance \n dFSNSC for an 0.1 darkening in alb'

filename = 'dTs_for_0.1FSDS_global_landmask'

mp, cbar, cs , fig, axes = mml_map_local(mapdata_raw=dT,
                                         units='K',prop=None,sea=None,mask=landmask,maskname='land', 
                                         LN=LN, LT=LT,
                                         cb=cmap,climits=clim,save_subplot=None,
                                         figpath=figpath['main'],scale=None,filename=filename,ttl=ttl)


#%%
