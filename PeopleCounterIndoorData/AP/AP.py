# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 15:43:34 2020
Time_Series dataset

@author: neshragh
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
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.cm as cm

##################################################################
##open a dataset
#df = pd.read_csv('ecounter.csv')
df = pd.read_csv('C:/Users/neshragh/OneDrive - University of New Brunswick/'
                 'UNB_thesis_Work/Affinity_Sample_SPY/1-Dataset_ecounter/Ecounter_Dataset/weekly/1weekbefore.csv')
#df = pd.read_csv('1weekIntervention.csv')
# #############################################################################
#df = df.loc[(df['Date'] == '5/1/2019')]

#Choose data for algorithm
#df = df.loc[(df.Date >= '5/3/2019') & (df.Date <= '5/5/2019')] 

X= df.loc[df.index,['Position','Count']].to_numpy()
#labels_true = df_data.loc[df_data.index<90000,'Time'].to_numpy()

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

#X = X.reshape((1,2))
#
#z = np.array([1,2])
#t=[3,4]
#x = np.array([[1, 2], [3, 4]])
#pref = np.median(euclidean_distances(X, squared=True))

################# controlying
pref =  -np.ones(len(X))*1000
pf = - np.median(euclidean_distances(X, squared=True))
############# Before
#pref[1512] =pf
#pref[1519] =pf
#pref[1694] =pf
#pref[2221] =pf
#pref[3100] =pf
#pref[1515] =pf
#pref[79] =pf
#pref[90] =2*pf
#pref[412] =pf
#pref[1516] =pf
#pref[1514] =pf
#pref[1596] =pf
#pref[1703] =pf
#pref[5733] =pf
#pref[2955] =pf



######### after
#
pref[0] =pf
pref[380] =pf
pref[806] =pf
pref[1767] =pf
pref[4209] =pf*2
pref[4649] =pf
pref[4836] =pf*2
pref[5054] =pf
pref[5460] =pf
pref[5623] =pf
pref[2439] =pf
pref[1959] =pf
pref[4797] =pf
pref[6081] =pf
pref[1172] =pf
pref[1812] =pf
pref[4020] =pf
#3
pref[1430] =pf*1/1.2
#6
pref[5622] =pf
##############
###############durind

#pref[0] =pf*2
#pref[1] =pf*2
#pref[39] =pf*2
#pref[41] =pf*2
#pref[43] =pf
#pref[40] =pf
#pref[374] =pf
#pref[506] =pf
#pref[43] =pf
#pref[531] =pf*3
#pref[537] =pf
#pref[588] =pf
#pref[1563] =pf
#pref[1889] =pf
#pref[2940] =pf
#pref[2946] =pf
#pref[2951] =pf
#pref[2955] =pf*2
#pref[3356] =pf
##1up
#pref[3825] =pf*2
#pref[5814] =pf
#pref[4638] =pf





af = AffinityPropagation(preference=pref, damping=.96, max_iter= 100 ).fit(X)


# #############################################################################
# Compute Affinity Propagation -ecounter dataset
#af = AffinityPropagation(preference=-0.1, damping=.92, max_iter= 100 ).fit(X)# 13 clusters, normalization
#af = AffinityPropagation(preference=-0.1, damping=.95, max_iter= 100 ).fit(X) # 3 clusters, normalization
#af = AffinityPropagation(preference=-4, damping=.95, max_iter= 100 ).fit(X) #without normalization Result
# X clusters, normalization Similarity mean = 0.29, median = 0.16, min = -2
#af = AffinityPropagation(preference=[-pref*0.15, -pref, -pref*10.5 ], damping=.98, max_iter= 100 ).fit(X) ecounter time data 12 cluster
#af = AffinityPropagation(preference=-.3, damping=.96, max_iter= 100 ).fit(X) #without normalization Result
#af = AffinityPropagation(preference=-4, damping=.96, max_iter= 100 ).fit(X) #final results-before
#af = AffinityPropagation(preference=-.7, damping=.96, max_iter= 100 ).fit(X) #final results-after
#af = AffinityPropagation(preference=[-pref*0.15, -pref, -pref*10.5 ], damping=.96, max_iter= 100 ).fit(X) #final results-During




cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

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
#
#plt.close('all')
#plt.figure(1)
#plt.clf()
#
#colors = cm.plasma(np.linspace(0, 1, n_clusters_))
#
##colors = cycle('ykbmgrc')
#for k, col in zip(range(n_clusters_), colors):
#    class_members = labels == k
#    cluster_center = X[cluster_centers_indices[k]]
#    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor=col, markersize=14)
##    for x in X[class_members]:
##        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)




plt.close('all')
plt.figure(2)
plt.clf()
#colors = iter(plt.rainbow(np.linspace(0, 1, num_clusters))
colors = cm.plasma(np.linspace(0, 1, n_clusters_))
#viridis-plasma
#colors = cycle('ykbmgrc')
#for k, col in zip(range(num_clusters), colors):
for k, col in zip(range(n_clusters_), colors):    
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], '.', markerfacecolor=col,
             markeredgecolor=col ,markersize=8)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor=col , markersize=12)
    
    plt.annotate(len(X[labels==k]), (cluster_center[0], cluster_center[1]),
                 textcoords="offset points",xytext=(10,3), ha='left', color=col)



plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])
#plt.grid()
plt.title('AP Clustering Algorithm: March 18- March 24')
#plt.title('AP Clustering Algorithm: May 27- June 2')
#plt.title('AP Clustering Algorithm: April 29- May 5')
plt.xlabel('Position')
plt.ylabel('Number of People')
plt.show()
##############################################################################
print('Estimated number of clusters: %d' % n_clusters_)

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
