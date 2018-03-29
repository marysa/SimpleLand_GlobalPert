#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:38:13 2018

@author: mlague

t-test for 2 means, following Abby's matlab script
 ttest_for_twomeans_2tail_ABBY.m

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

def ttest_2tail_2means(xbar1,xbar2,std1,std2,n1,n2,dof1,dof2):
    #
    # ttest_2tail_2means(xbar1,xbar2,std1,std2,n1,n2,dof1,dof2)
    #
    #   INPUTS
    #       xbar1, xbar2 = means to test against each other
    #       std1, std2 = standard deviations of each datatset (use dof = n-1; calculate with numpy.std function)
    #       
    #
    #   OUTPUTS
    #       t = t-value (map)
    #       pval = p value (map)
    #       reject95flag2tail = flag if we reject the null hypotehsis (null = they're the same) at 95% confidnce -> basically the map of where p<pval=0.05. 
    #
    #
    #----------------------------------------------
    
    
    
    #------------------
    #   s = standard deviation (zar eq. 4.8)
    #   s2 = s**2 = sum([xs - xbar]**2) / dof 
    #------------------
    s2_1 = (std1**2)*(n1-1)/dof1
    s2_2 = (std2**2)*(n2-1)/dof2
    
    # sx = standard deviation of the mean (zar eq 6.18)
    sx1x2 = np.sqrt(s2_1/n1 + s2_2/n2)
    
    #------------------
    #   t-value (zar eq 7.1)
    #------------------
    t = np.abs( (xbar1 - xbar2) / sx1x2 )
    
    
    #------------------
    #  pt-value (zar eq 7.1)
    #------------------
    pval = 2*(1 - stats.t.cdf(t,dof2))
    
    
    #------------------
    #   95% confidence interval
    #------------------
    t_table95 = np.abs(stats.t.interval(0.025,dof2))
    t_table99 = np.abs(stats.t.interval(0.005,dof2))
    
    reject95_flag = np.ones(np.shape(xbar1))
    reject95_flag = np.where(pval>0.05,0.0,1.0)
    
    
    
    return t, pval, reject95_flag
    
    
def ttest_mask(dataset1,dataset2,pval):
    # return a mask where the two datasets dataset1 and dataset2 pass a t-test with p<pval
    # using 2tailed student's ttest
    #
    #   INPUTS
    #       dataset1, dataset2 = time series of the variable in question (lat x lon x time)
    #       pval = p value (e.g p<0.05) that you want to use to say if they're significantly different or not
    #
    #   We'll always assume lagged autocorrelation of 2 years (dof = n/2) since I'm usually dealing with 
    #   climate model output; if that isn't the case, THIS IS NOT THE FUNCTION TO USE
    #
    #   OUTPUTS
    #       pmask
    #
    #
    #----------------------------------------------
    
    dims1 = np.shape(dataset1)  # lat x lon x time, otherwise we have a problem... 
    n1 = dims1[2]  
    std1 = np.std(dataset1,axis=2)
    xbar1 = np.mean(dataset1,axis=2)
    
    dims2 = np.shape(dataset2)  # lat x lon x time, otherwise we have a problem... 
    n1 = dims1[2]  
    std2 = np.std(dataset2,axis=2)
    xbar2 = np.mean(dataset2,axis=2)
    
    
    t, p, reject95_flag = ttest_2tail_2means(xbar1,xbar2,std1,std2,n1,n2,dof1,dof2)
    
    p_mask = np.ones(np.shape(xbar1))
    p_mask = np.where(p<pval,1.0,0.0)
    
    
    
    
    
    
    






