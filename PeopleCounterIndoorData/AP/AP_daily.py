# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 15:43:34 2020
One week Time_Series dataset
Calculate AP daily
@author: neshragh
"""

from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

import pandas as pd
from itertools import cycle
import os
import time
start_time = time.time()
import psutil
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import euclidean_distances
##################################################################
##open a dataset
#df = pd.read_csv('ecounter.csv')

df = pd.read_csv('1weekIntervention.csv')
# #############################################################################

#Choose data for algorithm
#df = df.loc[(df.Date >= '5/30/2019') & (df.Date <= '5/5/2019')] 

dff = df.loc[(df.Date == '5/1/2019')& (df.Date <= '5/1/2019') ]
dff['Timestamp'] = dff['Date'] + ' ' + dff['Time']
dff['Timestamp'] = pd.to_datetime(dff['Timestamp'])
dff = dff.sort_values(by=['Timestamp'])

X= dff.loc[dff.index,['Position','Count']].to_numpy()

#
#plt.close('all')
#plt.figure(10)
#plt.clf()
#plt.plot(dff.Count,'+')
##
#plt.figure(100)
#plt.clf()
##dff.set_index('Time').plot()
#dff.plot(x='Time', y='Count',style='+')
#plt.plot(df.Time,'+')
#
#
#plt.figure(11)
#plt.clf()
#plt.plot(dff.Position,'o')
#
#plt.figure(111)
#plt.clf()
#dff.plot(x='Time', y='Position',style='o')

#labels_true = df_data.loc[df_data.index<90000,'Time'].to_numpy()

### Nasrin's min max scaler: Possibly scales inliers into narrow range ##########################################
#a = X[:,0]
#b= X[:,1]
#max_a = np.max(a)
#max_b = np.max(b)
#min_a = np.min(a)
#min_b = np.min(b)
#a = (a-min_a) / (max_a-min_a)
#b = (b - min_b) / (max_b-min_b)
#X = np.array([[a,b]]) 
#p=X.reshape(2,len(a))
#X=p.T

### Level max scaler: scales all data between 1-6 ##########################################
a = X[:,0]
b= X[:,1]
max_a = np.max(a)
max_b = np.max(b)
#a = (a*max_b)/(max_a)
b = (b*max_a)/(max_b)
X = np.array([[a,b]]) 
p=X.reshape(2,len(a))
X=p.T
##### scale #################################
#X = preprocessing.scale(X)
### Other normalizations including Robust scaler: Should be more robust in the presence of outliers #################################
#trans = RobustScaler().fit(X)
#trans = MaxAbsScaler().fit(X)
#trans = MinMaxScaler().fit(X)
#trans = StandardScaler().fit(X)
#
#X = trans.transform(X)
init = np.mean(pdist(X))
pref = np.median(euclidean_distances(X, squared=True))
print('init = ',init, 'pref = ', pref)
#############################################

#X = X.reshape((1,2))
#
#z = np.array([1,2])
#t=[3,4]
#x = np.array([[1, 2], [3, 4]])

# #############################################################################
# Compute Affinity Propagation -ecounter dataset
#af = AffinityPropagation(preference=-0.1, damping=.92, max_iter= 100 ).fit(X)# 13 clusters, normalization
#af = AffinityPropagation(preference=-0.1, damping=.95, max_iter= 100 ).fit(X) # 3 clusters, normalization
#af = AffinityPropagation(preference=-4, damping=.95, max_iter= 100 ).fit(X) #without normalization Result
#af = AffinityPropagation(preference=-pref, damping=.93, max_iter= 100 ).fit(X) # 3 clusters, normalization
af = AffinityPropagation(preference=[-pref*0.15, -pref, -pref*10.5 ], damping=.93, max_iter= 100, verbose = True ).fit(X) # 3 clusters, normalization
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels))
'''print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))'''

# #############################################################################
# Plot result

#plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
#    for x in X[class_members]:
#        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)


plt.title('AP Clustering Algorthm, one week of ecounter data')
plt.xlabel('Position')
plt.ylabel('Number of People')
plt.show()
##############################################################################

print("Processing time: %s seconds" % (time.time() - start_time))

#Profiling and memory usage--------------------------------------------------
process = psutil.Process(os.getpid())
print("Memory Consumption:",process.memory_info().rss, 'bytes')   
M= process.memory_info().rss /1000000
print('(Or Megabyte:', M,')')
print('Estimated number of clusters: %d' % n_clusters_)


from sklearn.metrics import davies_bouldin_score
print('Calinski-Harabasz Index:',metrics.calinski_harabasz_score(X, labels))  
print('Silhouette Coefficient:',metrics.silhouette_score(X, labels, metric='euclidean'))
print('Davies-Bouldin Index:',davies_bouldin_score(X, labels))
