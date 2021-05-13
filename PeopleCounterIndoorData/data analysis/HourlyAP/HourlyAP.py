# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 13:21:07 2021

@author: neshragh




Hourly AP FOR e-counter, specific Day
"""

import pandas as pd
import time
start_time = time.time()
import matplotlib.pyplot as plt

from sklearn.cluster import AffinityPropagation
from itertools import cycle
import time
start_time = time.time()
import matplotlib
matplotlib.use('Qt5Agg')
##################################################################
#open a dataset
#df = pd.read_csv('ecounter.csv')
df = pd.read_csv('29apriloneweek.csv')#one day






# Compute Affinity Propagation
#X= df.loc[df.index,['Position','Count']].to_numpy()
df = df.loc[(df['Date'] == '5/1/2019')]
#df = df[df]
df = df.loc[(df.Time >= '18') & (df.Time <= '19')] 
X = df[['Count', 'Position']] .values
af = AffinityPropagation(preference=-1, damping=.9 , max_iter= 100 ).fit(X)

#af = AffinityPropagation(preference=-1, damping=.93 , max_iter= 100 ).fit(X) #13
#af = AffinityPropagation(preference=-10, damping=.85 , max_iter= 100 ).fit(X) Hour=15-16
#af = AffinityPropagation(preference=-10, damping=.94 , max_iter= 100 ).fit(X) #hour 14
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

# #########################################
# Plot result
from itertools import cycle


plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
   #CODE FOR POINTS OF BRANCHES 
    plt.plot(X[class_members, 1], X[class_members, 0], col + '.')
    plt.plot(cluster_center[1], cluster_center[0], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[1], x[1]], [cluster_center[0], x[0]], col)
#CODE FOR LINES OF BRANCHES    
#    for x in X[class_members]:
#        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Time: 18 pm - 19 pm',fontsize=20)

plt.xlabel('Postions')
plt.ylabel('Number of people')

frame =plt.gca()
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)

plt.ylim(0,6)

plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
           "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])
#plt.plot(X[:,0],X[:,1],'*') # plot all the data    
#plt.xlim(1,6)
plt.show()
