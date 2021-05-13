# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:01:18 2020

@author: neshragh
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import os
import time
start_time = time.time()
import psutil
from itertools import cycle
from sklearn.metrics import davies_bouldin_score
import numpy as np
import matplotlib.cm as cm

df = pd.read_csv('C:/Users/neshragh/OneDrive - University of New Brunswick/UNB_thesis_Work/Affinity_Sample_SPY/1-Dataset_ecounter/Ecounter_Dataset/weekly/1weekbefore.csv')

#df = pd.read_csv('C:/Users/neshragh/ecounter/Affinity_Sample_SPY/1-Dataset_ecounter/AP/1weekIntervention.csv')
#df = dc.loc[(dc.Date == '4/29/2019') &  (dc.Time >= '10:00:00') 
#&  (dc.Time <= '18:00:00')] 
#df = df.loc[(df['Date'] >= '4/29/2019') & (df.Date <= '5/5/2019')]

X= df.loc[df.index,['Position','Count']].to_numpy()

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



num_clusters = 7
kmeans = KMeans(n_clusters = num_clusters).fit(X)
labels = kmeans.labels_

n_clusters = kmeans.cluster_centers_

#colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#
#plt.scatter(df['Position'],df['Count'],c=labels)
#plt.title('K-means Clustering Algorthm, one week of ecounter data')
#plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
#            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])
#plt.xlabel('Position')
#plt.ylabel('Number of People')
#plt.show()
#cluster_k = X[labels == 0]
# #############################################################################


plt.close('all')
plt.figure(1)
plt.clf()
#colors = iter(plt.rainbow(np.linspace(0, 1, num_clusters))
colors = cm.plasma(np.linspace(0, 1, num_clusters))
#viridis-plasma
#colors = cycle('ykbmgrc')
#for k, col in zip(range(num_clusters), colors):
for k, col in zip(range(num_clusters), colors):    
    class_members = labels == k
#    plt.plot(X[class_members, 0], X[class_members, 1], col + '.',markersize=5)
    plt.plot(X[class_members, 0], X[class_members, 1], '.', markerfacecolor=col,
             markeredgecolor=col ,markersize=8)
    plt.plot(n_clusters[k,0], n_clusters[k,1], 'o', markerfacecolor=col,
             markeredgecolor=col , markersize=12)
    
    plt.annotate(len(X[labels==k]), (n_clusters[k,0], n_clusters[k,1]),
                 textcoords="offset points",xytext=(-8,10), ha='left', color=col)


#annotations=X[labels]
#for k, label in enumerate(annotations):
#    plt.text(n_clusters[k,0], n_clusters[k,1],label)   
    
#plt.scatter(n_clusters[:,0],n_clusters[:,1], c='red' ,marker='o', s=200, alpha=0.2)

plt.title('K-means Clustering Algorithm: March 18- March 24')
#plt.title('K-means Clustering Algorithm: May 27- June 2')
#plt.title('K-means Clustering Algorithm: April 29- May 5')
plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])
plt.xlabel('Position')
plt.ylabel('Number of People')
plt.show()


# =============================================================================
print("Processing time: %s seconds" % (time.time() - start_time))

#Profiling and memory usage--------------------------------------------------
process = psutil.Process(os.getpid())
print("Memory Consumption:",process.memory_info().rss, 'bytes')   
M= process.memory_info().rss /1000000
print('(Or Megabyte:', M,')')


# =============================================================================
print('Calinski-Harabasz Index:',metrics.calinski_harabasz_score(X, labels))  
print('Silhouette Coefficient:',metrics.silhouette_score(X, labels, metric='euclidean'))
print('Davies-Bouldin Index:',davies_bouldin_score(X, labels))
