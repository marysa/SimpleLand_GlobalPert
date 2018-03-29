#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:27:18 2017

@author: mlague

Function to calculate slopes datm/dlnd and dlnd/datm from multiple perturbed model simulations

"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:46:35 2017

@author: mlague
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

def sensitivity_slope(forcing,response):
    #-----------------------------
    # Inputs:
    #
    # forcing : vector of surface property values (eg [1,2,3])
    # response: array of responses ( # of forcings x lat x lon )
    #
    #
    #
    #
    #-----------------------------
    
    
    #-----------------------------
    #  Do the regression
    #-----------------------------
    
    # Get the perturbation values, make an np.array (default is list)
    xvals = np.array(forcing)
    k = np.size(xvals)  
    print('k = ' + np.str(k) )
    # slope model doesn't like nans; set nans to zero
    response = np.where(np.isnan(response),0,response)
    
    # grab atmospheric response data for current property, make an np.array
    raw_data = np.array(response)   # k x lat x lon

    # flatten response data into a single long vector (Thanks to Andre for showing me how to do this whole section)
    raw_data_v = raw_data.reshape(k, -1)

    
    
    # create an "empty" model
    model = linear_model.LinearRegression()
    
    # Fit the model to tmp_data
    model.fit(xvals[:, None], raw_data_v)
    
    #  grab the linear fit vector
    slope_vector = model.coef_
    intercept_vector = model.intercept_
    
    # put back into lat/lon (hard coded to be 2.5 degrees right now...)
    slope = slope_vector.reshape(96,144)
    intercept = intercept_vector.reshape(96,144)
    
    #-----------------------------
    #   Calculate the r^2 value
    #-----------------------------

    # grab the linear fit using the slope and intercept, so we can calculate the correlation coefficient
    fit_data_v = np.transpose(slope_vector*xvals)

    #x_bar = np.mean(xvals)
    #std_x = stats.tstd(xvals)

    r_v = np.zeros(np.shape(raw_data_v[0,:]))
    p_v = np.zeros(np.shape(raw_data_v[0,:]))
    #print(np.shape(r_v))
    
    for j in range(np.size(raw_data_v,1)):
       
        # compare to using the pearson-r function:
        r, p = stats.pearsonr(raw_data_v[:,j],fit_data_v[:,j])
        r_v[j] = r
        p_v[j] = p

  
    #print(np.shape(r_v.reshape(96,144)))
    
    r_value = r_v.reshape(96,144)
    p_value = p_v.reshape(96,144)

    
    
    # Do standard deviation analysis
    
    sigma = np.std(raw_data_v,axis=0)
    
    std_dev = sigma.reshape(96,144)
    
    del model
    
    ########################################################
    # repeat model slope calculation for dlnd/datm
    # create an "empty" model
    model = linear_model.LinearRegression()
    
    # Fit the model to tmp_data
    model.fit(raw_data_v, xvals[:, None])
    
    #  grab the linear fit vector
    slope_vector = model.coef_
    #intercept_vector = model.intercept_
    
    # put back into lat/lon (hard coded to be 2.5 degrees right now...)
    slope_inv = slope_vector.reshape(96,144)
#    intercept_inv = intercept_vector.reshape(96,144)
 
    del model
    
    return slope, slope_inv, intercept, r_value, p_value, std_dev
  
#%%
    
