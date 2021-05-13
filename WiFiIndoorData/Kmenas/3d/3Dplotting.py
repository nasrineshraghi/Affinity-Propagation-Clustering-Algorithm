# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 09:54:02 2020

@author: neshragh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import os
import cProfile
import re
import time
start_time = time.time()
import psutil
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

df = pd.read_csv("week.csv")
#df = dc.loc[(dc.Date == '4/29/2019') &  (dc.Time >= '10:00:00') 
#&  (dc.Time <= '18:00:00')] 

X= df.loc[df.index,['SPACEID','PHONEID']].to_numpy()
plt.figure(1)
plt.scatter(X[:,0],X[:,1],marker='.')
X = preprocessing.normalize(X)
plt.figure()
plt.scatter(X[:,1],X[:,0],marker='.')
#df = df.loc[df.SPACEID >1]

################################
#X = preprocessing.normalize(X)
num_clusters = 7
kmeans = KMeans(n_clusters = num_clusters).fit(X)
labels = kmeans.labels_

n_clusters = kmeans.cluster_centers_

plt.figure(2)
plt.scatter(X[:,0],X[:,1], c= labels,marker='.')
#
#lat = df['LATITUDE']
#lon = df['LONGITUDE']
#floor = df['SPACEID']
#
#fig = plt.figure(2)
#ax = Axes3D(fig)
#
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(lat,lon ,floor, c= labels, marker="o", picker=True)
#ax.set_xlabel('LATITUDE')
#ax.set_ylabel('LONGITUDE')
#ax.set_zlabel('RELATIVEPOSITION')
#plt.show()
#











#
#plt.scatter(df['SPACEID'],df['PHONEID'],c=labels,marker='.')
##plt.scatter(df['LONGITUDE'],df['LATITUDE'],c=labels,marker='.')
#plt.scatter(n_clusters[:,0],n_clusters[:,1], c='black',marker='o', s=100, alpha=0.2)
#plt.title('K-means Clustering Algorthm, one week of Wifi data')
plt.xlabel('SPACE ID')
plt.ylabel('PHONE ID')
plt.show()

###############################################################################
print("Processing time: %s seconds" % (time.time() - start_time))

#Profiling and memory usage--------------------------------------------------
process = psutil.Process(os.getpid())
print("Memory Consumption:",process.memory_info().rss, 'bytes')   
M= process.memory_info().rss /1000000
print('(Or Megabyte:', M,')')


from sklearn.metrics import davies_bouldin_score
print('Calinski-Harabasz Index:',metrics.calinski_harabasz_score(X, labels))  
print('Silhouette Coefficient:',metrics.silhouette_score(X, labels, metric='euclidean'))
print('Davies-Bouldin Index:',davies_bouldin_score(X, labels))
# =============================================================================
