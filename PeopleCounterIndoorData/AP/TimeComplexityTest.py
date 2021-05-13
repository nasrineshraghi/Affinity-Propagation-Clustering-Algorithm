# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:34:51 2021

@author: 
    
    Testing the time complexity of AP
"""
from sklearn.cluster import AffinityPropagation
import pandas as pd
from itertools import cycle
import os
import time
start_time = time.time()
import psutil
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


timecom = [3, 15, 6, 8, 13, 12,13]
datapoint=[980, 2179, 3287, 4300, 5354, 5371,5418]
memo = [204, 243, 290, 358, 433, 434, 1593]

plt.plot(datapoint,memo)
plt.show()

#df = pd.read_csv('C:/Users/neshragh/ecounter/Affinity_Sample_SPY/1-Dataset_ecounter/data analysis/HourlyAP/29apriloneweek.csv')#one day
df = pd.read_csv('C:/Users/neshragh/ecounter/Affinity_Sample_SPY/1-Dataset_ecounter/data analysis/monthIntervention.csv')#one day
# #############################################################################

#Choose data for algorithm
#df = df.loc[(df.Date >= '5/30/2019') & (df.Date <= '5/5/2019')] 
df = df.loc[(df['Date'] >= '4/29/2019') & (df.Date <= '5/4/2019')]
#df = df[df]
df = df.loc[(df.Time >= '07') & (df.Time <= '19')] 
X= df.loc[df.index,['Count','Position']].to_numpy()


# Compute Affinity Propagation
af = AffinityPropagation(preference=-4, damping=.95, max_iter= 500 ).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

#print('Estimated number of clusters: %d' % n_clusters_)
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

plt.close('all')
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

plt.title('AP Clustering Algorthm, one week of Wifi data')
plt.xlabel('SPACE ID')
plt.ylabel('PHONE ID')
plt.show()
##############################################################################

print("Processing time: %s seconds" % (time.time() - start_time))

#Profiling and memory usage--------------------------------------------------
process = psutil.Process(os.getpid())
print("Memory Consumption:",process.memory_info().rss, 'bytes')   
M= process.memory_info().rss /1000000
print('(Or Megabyte:', M,')')
print('Estimated number of clusters: %d' % n_clusters_)


#Silhouette Coefficient------------------------------------------------------
from sklearn.metrics import davies_bouldin_score
print('Calinski-Harabasz Index Macro:',metrics.calinski_harabasz_score(X, labels))  
print('Silhouette Coefficient Macro:',metrics.silhouette_score(X, labels, metric='euclidean'))
print('Davies-Bouldin Index Macro:',davies_bouldin_score(X, labels))
        


