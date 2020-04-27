# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 15:43:34 2020
Time_Series dataset

@author: neshragh
"""

from sklearn.cluster import AffinityPropagation
#from sklearn import metrics
import pandas as pd


##################################################################
##open a dataset
df_data = pd.read_csv('Februay/Feb26/26-2-2019.5.csv')
#df_data = pd.read_csv('March5/5-3-2019.10.csv')


#Filter dataset
#data = df_data[['Date', 'Time','Count', 'Status', 'Sensor', 'Type', 'Position','Location', 'Location Code']] .values
#df_filtered = data[(df_data.Time >='11') & (df_data.Time <='11:59') ]
#print(len(df_filtered))
#print(df_filtered)

#Split Apply Combine
#g = df_data.groupby('Time')
#for Time, Time_df_data in g :
#    print(Time_df_data)
#    print(Time)
#g.get_group('Time' == '9:59:51' )

# #############################################################################

#Choose data for algorithm
X= df_data.loc[df_data.index<90000,['Position','Count']].to_numpy()
labels_true = df_data.loc[df_data.index<90000,'Time'].to_numpy()



# #############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(preference=-50, damping=.78, max_iter= 100 ).fit(X)
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
import matplotlib.pyplot as plt
from itertools import cycle

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
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
