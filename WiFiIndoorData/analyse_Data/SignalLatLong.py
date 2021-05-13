# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 18:37:58 2021

@author: neshragh
"""

import pandas as pd
import time
start_time = time.time()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from itertools import cycle
import matplotlib.dates as mdates
import datetime
import os
import cProfile
import re
import time
import psutil
from sklearn import metrics
start_time = time.time()
from sklearn.metrics import davies_bouldin_score
import matplotlib
matplotlib.use('Qt5Agg')
##################################################################
#open a dataset
df = pd.read_csv('C:/Users/neshragh/ecounter/Affinity_Sample_SPY/2-Dataset_Wifi/archive/TrainingDataWithUniQID.csv')



### RSSI+ LAT+ LONG
#
#df = df.loc[(df['BUILDINGID'] == 0) & (df['FLOOR'] == 2)] 
#colormap = df.WAP034
#plt.scatter(df.LATITUDE, df.LONGITUDE, c = colormap)
#
#plt.colorbar()
#plt.clim(-98, -57)
#

########### 
#df = df.loc[(df['BUILDINGID'] >= 0) & (df['FLOOR'] == 2)] 
#df[:]
#colormap = df.USERID
#plt.scatter(df.LATITUDE, df.LONGITUDE, c = colormap)
#
#plt.colorbar()
#plt.clim( 69926990,  70339570)




#####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#### one floor + one user



df = df.loc[(df['BUILDINGID'] == 1)  & (df['PHONEID'] == 9)] 

plt.scatter(df.SPACEID, df.TIMESTAMP)






