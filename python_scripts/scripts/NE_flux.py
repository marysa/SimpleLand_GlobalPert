#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:39:08 2017

@author: mlague
"""

"""
    Calculates northward energy transport at each latitude for a given input vector (or array)
    
    options:
        zonal_mean = True 
            take zonally averaged meridional energy transport
            otherwise do it lon by lon and return an array with a value for each longitude
"""
# In[]:

# For interactive in-line plots:
#%matplotlib nbagg  

# For inline plots:
#%matplotlib inline     


import matplotlib
from mpl_toolkits.basemap import Basemap, cm


import numpy as np
import matplotlib.pyplot as plt

#%%


def NE_flux(lat,lon,FSNT,FLNT,area_grid,zonal_mean=None,stats=None):
    
    # lat = vector of latitudes
    # lon = vector lof longitudes
    # FSNT = TOA shortwave [W/m2]
    # FLNT = TOA longwave [W/m2]
    # area = matrix for area weighting, optional
    # zonal_mean = tells if we want to average zonally, or do lon by lon. 
    #               Default: lon by lon, returns array (vs vector)
    #
    # Assumes only one time slice is given (no time series, no 12 month things... modify this later to be more flexible)
    
    area = area_grid
   
    if stats == False:
        # Define toa energy imbalance/residual as shortwave - longwave
        # multiply by energy to work in clean watts
        Rtoa = (FSNT - FLNT)*area
        
        # spaces between lats and lons
        dlat = np.array(np.diff(lat))
        dlat = np.append(dlat,dlat[-1])
        dlon = np.array(np.diff(lon))
        dlon = np.append(dlon,dlon[-1])
        
        
        # create empty flux matrix to fill in
        flux = np.zeros([np.shape(lat)[0],np.shape(lon)[0]])
        
        # Loop over latitudes and sum energy
        for k in range(np.shape(lat)[0]):
            # sum energy of southern lats up to that lat
            for r in range(k):
                flux[k,:] = flux[k,:] + Rtoa[r,:]
            
        # calculate how much we overshot zero at the north pole
        imbal = flux[-1,:]
        imbal_mat = imbal / np.shape(lat)[0] * np.ones([np.shape(lat)[0],np.shape(lon)[0]])
        
        # modify Rtoa to account for discrete step sizes & overshoot
        Rtoa_mod = Rtoa - imbal_mat
        
        # re-calculate flux
        flux_mod = np.zeros([np.shape(lat)[0],np.shape(lon)[0]])
        for k in range(np.shape(lat)[0]):
            for r in range(k):
                flux_mod[k,:] = flux_mod[k,:] + Rtoa_mod[r,:]
                
        # Put back into W/m2 by dividing by area
        #NE_flux = flux_mod / area
        NE_flux = flux_mod*10**(-15)
        
        fig, axes = plt.subplots(1, 1, figsize=(5,5))
        ax = plt.gca()
        plt.plot(lat,NE_flux)
        ax.set_ylabel('NE Flux in PW')
        ax.set_xlabel('latitude')
        xlim = ax.get_xlim()
        xline = [xlim[0], xlim[1]]
        ylim = [0,0]
        plt.plot([xlim[0],xlim[1]], [0,0] ,linestyle='dashed',color='gray')
        plt.show()
        plt.close()
        
        if zonal_mean == True:
            # average zonally, and divide through by area
            NE_flux_zonal = np.sum(NE_flux,1)
    
        fig, axes = plt.subplots(1, 1, figsize=(5,5))
        ax = plt.gca()
        plt.plot(lat,NE_flux_zonal)
        ax.set_ylabel('NE Flux in PW')
        ax.set_xlabel('latitude')
        xlim = ax.get_xlim()
        xline = [xlim[0], xlim[1]]
        ylim = [0,0]
        plt.plot([xlim[0],xlim[1]], [0,0] ,linestyle='dashed',color='gray')
        plt.show()
        plt.close()
    
    
    # 
    elif stats == True:
        # do std, dof, etc
        NE_flux = 'not defined yet!' # but a differnt shape this time
    else:
        # otherwise give up, go home...
        NE_flux = 'specify stats'
    
    
    
    
    
    
    
    
    
    return NE_flux, NE_flux_zonal
