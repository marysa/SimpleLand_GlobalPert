#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:42:20 2018

@author: mlague
"""

# In[]:
# Import appropriate packages for plotting, analysis, etc

# For inline plots:
get_ipython().magic(u'matplotlib inline')

import matplotlib
import numpy as np
import os
import datetime
import time 
import netCDF4 as nc
import xarray as xr
from scipy import interpolate
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
#import brewer2mpl as cbrew
import scipy.io as sio

from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)
import time as tm
#from scipy.interpolate import griddata
from mpl_toolkits import basemap


# MML's functions:
from mml_mapping_fun import mml_map, discrete_cmap
from custom_python_mml_cmap import make_colormap, mml_cmap
from sensitivity_slope_fun import sensitivity_slope
#from sens_slope_fun2 import sensitivity_slope
from load_masks_coords_fun import get_masks, get_coords, get_seasons
from load_global_pert_data import make_variable_arrays, get_online, get_offline
from box_avg_fun import avg_over_box, draw_box


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

# Load ncl processed netcdfs

sims = ['global_a2_cv2_hc0.1_rs100_cheyenne',
       'global_a1_cv2_hc0.1_rs100_cheyenne','global_a3_cv2_hc0.1_rs100_cheyenne',
       'global_a2_cv2_hc0.01_rs100_cheyenne','global_a2_cv2_hc0.05_rs100_cheyenne',
       'global_a2_cv2_hc0.5_rs100_cheyenne',
       'global_a2_cv2_hc1.0_rs100_cheyenne','global_a2_cv2_hc2.0_rs100_cheyenne',
       'global_a2_cv2_hc0.1_rs30_cheyenne','global_a2_cv2_hc0.1_rs200_cheyenne']


# load the file paths and # Open the coupled data sets in xarray
MSE_files = {}
phih_files = {}
ds_MSE = {}
ds_phih = {}

ext_dir = '/home/disk/eos18/mlague/simple_land/intermediate_netcdfs/global_pert_cheyenne/'

for run in sims:
    #print ( ext_dir + run + '/means/' + run + '.cam.h0.05-end_year_avg.nc' )
    MSE_files[run] = ext_dir + run + '/' + run + '.cam.h0.20-50.MSE-year_avg.nc'
    phih_files[run] = ext_dir + run + '/' + run + '.cam.20-50.phih_fromQ_year_avg.nc'
    
    ds_MSE[run] = xr.open_dataset(MSE_files[run])
    ds_phih[run] = xr.open_dataset(phih_files[run])
    
figpath =  '/home/disk/eos18/mlague/simple_land/intermediate_netcdfs/global_pert_cheyenne/figures/'
#%%

"""

    MSE, Phi_h
    
    Plot annual mean, save figures

"""

for run in sims:
    
    ##------------------------------------------
    """
        MSE
    """
    
    var = 'MSE_source'
    ds = ds_MSE[run]
    
    data = ds[var][:]
    
    
    # Annual plot:
    sea = 'ANN'
    ttl = 'Annual MSE ' + run
    clim = [-300,5]
    cm = plt.cm.viridis_r
    unit = 'W/m2'
    
    mapdata = np.mean(data,0)
    
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')
    
    
    ax0 = plt.gca()
    fig = plt.gcf()
    
     # Annotate with season, variable, date
    ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', ' +', '+var+', ',fontsize='10',
             ha = 'left',va = 'center',
             transform = ax0.transAxes)

    filename = run + '_MSE_'+sea
    
    plt.show()
    
    fig_name = figpath+'/'+filename+'.png'
    fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
    
    plt.close()

    # Monthly:
    
    for n in range(12):
        
        print('n = '+np.str(n))
        
        # Annual plot:
        sea = np.str(n)
        ttl = sea + ' MSE ' + run
        clim = [-400,50]
        cm = plt.cm.viridis_r
        unit = 'W/m2'
        
        mapdata = data[n,:,:]
        
        mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')
        
        
        ax0 = plt.gca()
        fig = plt.gcf()
        
         # Annotate with season, variable, date
        ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', ' +', '+var+', ',fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax0.transAxes)
    
        filename = run + '_MSE_'+sea
        
        plt.show()
        
        fig_name = figpath+'/'+filename+'.png'
        fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)
        
        plt.close()
        
    # patch months together into movie
    #os.system("ffmpeg -f image2 -r 1/1 -i )# maybe not... need to have transient filenames. 



    ##------------------------------------------
    """
        Phi_h
    """
    
    
    var = 'phih'
    ds = ds_phih[run]
    
    data = ds[var][:]
    
    
    # Annual plot:
    sea = 'ANN'
    ttl = 'Annual phi_h ' + run
    clim = [-3.e14,3.e14]
    cm = plt.cm.RdBu_r
    unit = 'W/m2'
    
    mapdata = np.mean(data,0)
    
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')
    
    
    ax0 = plt.gca()
    fig = plt.gcf()
    
     # Annotate with season, variable, date
    ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', ' +', '+var+', ',fontsize='10',
             ha = 'left',va = 'center',
             transform = ax0.transAxes)

    filename = run + '_MSE_'+sea
    
    plt.show()
    
    fig_name = figpath+'/'+filename+'.png'
    fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
    
    plt.close()

    # Monthly:
    
    for n in range(12):
        # Annual plot:
        sea = np.str(n)
        ttl = sea + ' MSE ' + run
        clim = [-3.e15,3.e15]
        cm = plt.cm.RdBu_r
        unit = 'W/m2'
        
        mapdata = data[n,:,:]
        
        mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')
        
        
        ax0 = plt.gca()
        fig = plt.gcf()
        
         # Annotate with season, variable, date
        ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', ' +', '+var+', ',fontsize='10',
                 ha = 'left',va = 'center',
                 transform = ax0.transAxes)
    
        filename = run + '_phih_'+sea
        
        plt.show()
        
        fig_name = figpath+'/'+filename+'.png'
        fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                    edgecolor='w',orientation='portrait',bbox_inches='tight', 
                    pad_inches=0.1,frameon=None)
        
        plt.close()
#%%
        
        """
            Select two runs, and plot their DIFFERENCE for the above - see 
            if I get that longitudinally migrating blob... 
        """

a1 = 'global_a1_cv2_hc0.1_rs100_cheyenne'
a3 = 'global_a3_cv2_hc0.1_rs100_cheyenne'



    
##------------------------------------------
"""
    MSE
"""

var = 'MSE'
ds1 = ds_MSE[a1]
ds3 = ds_MSE[a3]

data = ds1[var][:] - ds3[var][:]


# Annual plot:
sea = 'ANN'
ttl = 'Annual MSE ' + a3 +'-' + a1 
clim = [-20,20]
cm = plt.cm.viridis_r
unit = 'W/m2'

mapdata = np.mean(data,0)

mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')


ax0 = plt.gca()
fig = plt.gcf()

 # Annotate with season, variable, date
ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', ' +', '+var+', ',fontsize='10',
         ha = 'left',va = 'center',
         transform = ax0.transAxes)

filename = a3 +'_m_' + a1  + '_MSE_'+sea

plt.show()

fig_name = figpath+'/'+filename+'.png'
fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
            edgecolor='w',orientation='portrait',bbox_inches='tight', 
            pad_inches=0.1,frameon=None)

plt.close()

# Monthly:

for n in range(12):
    
    print('n = '+np.str(n))
    
    # Annual plot:
    sea = np.str(n)
    ttl = sea + ' MSE ' + a3 +'-' + a1 
    clim = [-30,30]
    cm = plt.cm.viridis_r
    unit = 'W/m2'
    
    mapdata = data[n,:,:]
    
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')
    
    
    ax0 = plt.gca()
    fig = plt.gcf()
    
     # Annotate with season, variable, date
    ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', ' +', '+var+', ',fontsize='10',
             ha = 'left',va = 'center',
             transform = ax0.transAxes)

    filename = a3 +'_m_' + a1  + '_MSE_'+sea
    
    plt.show()
    
    fig_name = figpath+'/'+filename+'.png'
    fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
    
    plt.close()
    
# patch months together into movie
#os.system("ffmpeg -f image2 -r 1/1 -i )# maybe not... need to have transient filenames. 



#%%
##------------------------------------------
"""
    Phi_h
"""


var = 'phih'

ds1 = ds_phih[a1]
ds3 = ds_phih[a3]

data = ds1[var][:] - ds3[var][:]


# Annual plot:
sea = 'ANN'
ttl = 'Annual phi_h ' + a3 +'-' + a1 
clim = [-1.e14,1.e14]
cm = plt.cm.RdBu_r
unit = 'W/m2'

mapdata = np.mean(data,0)

mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')


ax0 = plt.gca()
fig = plt.gcf()

 # Annotate with season, variable, date
ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', ' +', '+var+', ',fontsize='10',
         ha = 'left',va = 'center',
         transform = ax0.transAxes)

filename = a3 +'_m_' + a1 + '_MSE_'+sea

plt.show()

fig_name = figpath+'/'+filename+'.png'
fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
            edgecolor='w',orientation='portrait',bbox_inches='tight', 
            pad_inches=0.1,frameon=None)

plt.close()

# Monthly:

for n in range(12):
    # Annual plot:
    sea = np.str(n)
    ttl = sea + ' phih ' + a3 +'-' + a1 
    clim = [-2.e14,2.e14]
    cm = plt.cm.RdBu_r
    unit = 'W/m2'
    
    mapdata = data[n,:,:]
    
    mp, cbar, cs = mml_map(LN,LT,mapdata,ds=None,myvar=var,proj='moll',title=ttl,clim=clim,colmap=cm, cb_ttl='units: '+unit,ext='both')
    
    
    ax0 = plt.gca()
    fig = plt.gcf()
    
     # Annotate with season, variable, date
    ax0.text(0.,-0.4,time.strftime("%x")+'\n'+sea +', ' +', '+var+', ',fontsize='10',
             ha = 'left',va = 'center',
             transform = ax0.transAxes)

    filename = a3 +'_m_' + a1  + '_phih_'+sea
    
    plt.show()
    
    fig_name = figpath+'/'+filename+'.png'
    fig.savefig(fig_name,dpi=600,transparent=True,facecolor='w',
                edgecolor='w',orientation='portrait',bbox_inches='tight', 
                pad_inches=0.1,frameon=None)
    
    plt.close()
   

#%%
# try and patch into a movie?
#for run in sims:
#    
    #ended up running this from the terminal:
    #
    # ffmpeg -f image2 -r 1/1 -i ./global_a3_cv2_hc0.1_rs100_cheyenne_m_global_a1_cv2_hc0.1_rs100_cheyenne_phih_%d.png -vcodec mpeg4 -y a3_m_a1_phih.mp4
    #
    #
    
#    os.system("ffmpeg -f image2 -r 1/1 -i ")
    
#os.system('ffmpeg -f image2 -r 1/1 -i ' + figpath + '/' + a3 +'_m_' + a1  + '_phih_%d.png -vcodec -mpeg4 -y ' + figpath + '/' + a3 +'_m_' + a1  + '_phih.mp4')
#%%

    

#%%

    

#%%

    

#%%

