# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:01:18 2020

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


df = pd.read_csv("week.csv")
#df = dc.loc[(dc.Date == '4/29/2019') &  (dc.Time >= '10:00:00') 
#&  (dc.Time <= '18:00:00')] 

X= df.loc[df.index,['SPACEID','PHONEID']].to_numpy()



#
num_clusters = 7
kmeans = KMeans(n_clusters = num_clusters).fit(X)
labels = kmeans.labels_

n_clusters = kmeans.cluster_centers_

plt.figure(1)

plt.scatter(df['SPACEID'],df['PHONEID'],c=labels,marker='.')
#plt.scatter(df['LONGITUDE'],df['LATITUDE'],c=labels,marker='.')

plt.scatter(n_clusters[:,0],n_clusters[:,1], c='black',marker='o', s=100, alpha=0.2)
plt.title('K-means Clustering Algorthm, one week of Wifi data')
plt.xlabel('SPACE ID')
plt.ylabel('PHONE ID')
plt.show()
#plt.show()

# #############################################################################



################################################################################
#statistically give the colors to kmeans
#km = KMeans(n_clusters = 4)
#y_pre = km.fit_predict(df[['SPACEID', 'PHONEID']])
#df['cluster'] = y_pre
#
#df0 = df[df.cluster ==0]
#df1 = df[df.cluster ==1]
#df2 = df[df.cluster ==2]
#df3 = df[df.cluster ==3]
#
#
#
#plt.scatter(df1.SPACEID, df1['PHONEID'], color= 'red')
#plt.scatter(df2.SPACEID, df2['PHONEID'], color= 'green')
#plt.scatter(df0.SPACEID, df0['PHONEID'], color= 'blue')
#plt.scatter(df3.SPACEID, df3['PHONEID'], color= 'black')





# =============================================================================
print("Processing time: %s seconds" % (time.time() - start_time))

#Profiling and memory usage--------------------------------------------------
process = psutil.Process(os.getpid())
print("Memory Consumption:",process.memory_info().rss, 'bytes')   
M= process.memory_info().rss /1000000
print('(Or Megabyte:', M,')')


#from sklearn.metrics import davies_bouldin_score
#print('Calinski-Harabasz Index:',metrics.calinski_harabasz_score(X, labels))  
#print('Silhouette Coefficient:',metrics.silhouette_score(X, labels, metric='euclidean'))
#print('Davies-Bouldin Index:',davies_bouldin_score(X, labels))
# =============================================================================
