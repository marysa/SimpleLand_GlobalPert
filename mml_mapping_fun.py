#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:46:35 2017

@author: mlague
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

def mml_map(LN,LT,mapdata,ds,myvar,proj,title=None,clim=None,colmap=None,cb_ttl=None):
    # need to have already opened a figure/axis
    #plt.sca(ax)
    
    
    # There is a way to make a single basemap, save it, and just call that vover and over, ie a blank little map, 
    # that we then build maps on top of (pcolours)
    mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0,resolution='c') # can't make it start anywhere other than 180???
    mp.drawcoastlines()
    mp.drawmapboundary(fill_color='1.')  # make map background white
    parallels = np.arange(-90.,90,20.)
    mp.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.5,dashes=[1,2])
    meridians = np.arange(0.,360.,20.)
    mp.drawmeridians(meridians,linewidth=0.5,dashes=[1,2])
    #(x, y) = m(LONXY, LATXY)
    cs = mp.pcolormesh(LN,LT,mapdata,cmap=plt.cm.inferno,latlon=True)
    
    if colmap:
        cs.cmap = colmap
    else:
        cs.cmap = plt.cm.inferno    
    
    cbar = mp.colorbar(cs,location='bottom',pad="5%")
    
    cs.cmap.set_bad('white',1.)
    
    if cb_ttl:
        cbar.set_label(cb_ttl,fontsize=12)
    else:
        cbar.set_label('units: '+ds[myvar].units,fontsize=12)
    
    if title:
        plt.title(title,fontsize=12)
    else:
        plt.title(ds[myvar].long_name,fontsize=12)
    
    if clim:
        cbar.set_clim(clim[0],clim[1])
        cs.set_clim(clim[0],clim[1])
    
    
    
    
    #plt.suptitle('units?')
    #plt.show()
    
    
    #plt.show()
    return mp, cbar, cs 
  

def mml_map_NA(LN,LT,mapdata,ds,myvar,proj,title=None,clim=None,colmap=None,cb_ttl=None):
    # need to have already opened a figure/axis
    #plt.sca(ax)
    
    from mpl_toolkits.basemap import Basemap, cm
    
    # There is a way to make a single basemap, save it, and just call that vover and over, ie a blank little map, 
    # that we then build maps on top of (pcolours)
    #mp = Basemap(projection='robin',lon_0=180.,lat_0 = 0,resolution='c') # can't make it start anywhere other than 180???
 #   m = Basemap(projection='stere',lon_0=180,lat_0=0.,lat_ts=0,\
 #           llcrnrlat=10,urcrnrlat=90,\
 #           llcrnrlon=180,urcrnrlon=70,\
 #           rsphere=6371200.,resolution='l',area_thresh=10000)  
#    m = Basemap(llcrnrlon=-180.,llcrnrlat=10.,urcrnrlon=-40.,urcrnrlat=90.,
#            projection='lcc',lat_1=-90.,lat_2=lat[1],lon_0=-180.,
#            resolution ='l',area_thresh=10000.) 
    lat2 = -88.105262756347656
 
    #mp = Basemap(llcrnrlon=-150.,llcrnrlat=5.,urcrnrlon=-30.,urcrnrlat=80.,
    #        projection='lcc',lat_1=10.,lon_0=-30.,
    #        resolution ='l',area_thresh=10000.) 
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
    cs = mp.pcolormesh(LN,LT,mapdata,cmap=plt.cm.inferno,latlon=True)
    
    if colmap:
        cs.cmap = colmap
    else:
        cs.cmap = plt.cm.inferno    
    
    cbar = mp.colorbar(cs,location='bottom',pad="5%")
    
    cs.cmap.set_bad('white',1.)
    
    if cb_ttl:
        cbar.set_label(cb_ttl,fontsize=12)
    else:
        cbar.set_label('units: '+ds[myvar].units,fontsize=12)
    
    if title:
        plt.title(title,fontsize=12)
    else:
        plt.title(ds[myvar].long_name,fontsize=12)
    
    if clim:
        cbar.set_clim(clim[0],clim[1])
        cs.set_clim(clim[0],clim[1])
    
    
    
    
    #plt.suptitle('units?')
    #plt.show()
    
    
    #plt.show()
    return mp, cbar, cs 

def mml_neon_box(east,west,mpE,mpW,lat,lon):
    
    # draw the east box
    if east==1:
        x1,y1 = mpE(lon[110],lat[67])
        x2,y2 = mpE(lon[114],lat[67])
        x3,y3 = mpE(lon[114],lat[71])
        x4,y4 = mpE(lon[110],lat[71])
        poly = plt.Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor=None,edgecolor='green',linewidth=3,fill=False)
        plt.gca().add_patch(poly)

        
    # draw the west box
    if west==1:
        x1,y1 = mpW(lon[94],lat[67])
        x2,y2 = mpW(lon[98],lat[67])
        x3,y3 = mpW(lon[98],lat[71])
        x4,y4 = mpW(lon[94],lat[71])
        poly = plt.Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor=None,edgecolor='green',linewidth=3,fill=False)
        plt.gca().add_patch(poly)
        
    return
    