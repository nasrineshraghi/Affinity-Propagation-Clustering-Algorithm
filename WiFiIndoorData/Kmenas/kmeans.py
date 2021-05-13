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


#df = pd.read_csv("TrainingData.csv")
df = pd.read_csv('C:/Users/neshragh/ecounter/Affinity_Sample_SPY/2-Dataset_Wifi/archive/TrainingDataWithUniQID.csv')
df = df.loc[(df['BUILDINGID'] == 2) ] 
plt.scatter(df.uqid, df.PHONEID)
#df = pd.read_csv("wifi.csv")
#df = dc.loc[(dc.Date == '4/29/2019') &  (dc.Time >= '10:00:00') 
#&  (dc.Time <= '18:00:00')] 

# #######Create on column with all feature for uniqe space

#df['BUILDINGID']=df['BUILDINGID'].replace(0,3)
#
#
#
#
#bi = df['BUILDINGID']
#fl = df['FLOOR']
#rp = df['RELATIVEPOSITION']
#si = df['SPACEID']
#
#bi = bi*100000
#fl= fl*10000
#rp = rp *1000
#
#uqid = bi+fl+rp+si
#
#df['uqid'] = uqid
#
#
#
#data = df.values
#df.describe()
#userIdIdx = -3
#phoneIdIdx = -2
#spaceIdIdx = -6
#buildingIdIdx = -7
#floorIdx = -8
#numData = 19937
#y = -1
#


###SPACEID + All data point
#N = len(set(data[: , spaceIdIdx]))
#plt.hist(data[: , spaceIdIdx] , range(N), color = 'deepskyblue',edgecolor='red')
##plt.xticks(range(N))
#plt.title('Hisogram of all space IDs')
##plt.ylabel('Number of Data points')
#plt.xlabel('SPACEID')


#bi = np.char.mod('%d', bi)
#fl = np.char.mod('%d', fl)
#rp = np.char.mod('%d', rp)
#si = np.char.mod('%d', si)
#
#
#uid1 = np.char.add(bi,fl)
#uid2 = np.char.add(rp,si)
#uid = np.char.add(uid1,uid2)
#
#df['uid'] =uid

#uid =  uid.astype(np.int)

#df['uqid'] = uid






###SPACEID + All data point
#N = len(set(data[: , spaceIdIdx]))
#plt.hist(data[: , spaceIdIdx] , range(N), color = 'deepskyblue',edgecolor='red')
##plt.xticks(range(N))
#plt.title('Hisogram of all space IDs')
##plt.ylabel('Number of Data points')
#plt.xlabel('SPACEID')



X= df.loc[df.index,['uqid','PHONEID']].to_numpy()
### Level max scaler: scales all data between 1-6 ##########################################
#a = X[:,1]
#b= X[:,0]
#max_a = np.max(a)
#max_b = np.max(b)
##a = (a*max_b)/(max_a)
#b = (b*max_a)/(max_b)
#X = np.array([[a,b]]) 
#p=X.reshape(2,len(a))
#X=p.T



#######################################################################\

num_clusters = 6
kmeans = KMeans(n_clusters = num_clusters).fit(X)
labels = kmeans.labels_

n_clusters = kmeans.cluster_centers_

#plt.scatter(df['LONGITUDE'],df['LATITUDE'],c=labels,marker='o')

#plt.scatter(df['uqid'],df['PHONEID'],c=labels,marker='.')
plt.scatter(X[:,0],X[:,1],c=labels,marker='*')

#plt.scatter(n_clusters[:,0],n_clusters[:,1], c='black',marker='o', s=100, alpha=0.2)
plt.title('K-means Clustering Algorthm, one week of Wifi data')
#plt.xlabel('SPACE ID')
#plt.ylabel('PHONE ID')
plt.show()








#############################################################################
#X= df.loc[df.index,['SPACEID','PHONEID']].to_numpy()
##X= df.loc[df.index,['LATITUDE','LONGITUDE']].to_numpy()
#
#
#
#
#num_clusters = 10
#kmeans = KMeans(n_clusters = num_clusters).fit(X)
#labels = kmeans.labels_
#
#n_clusters = kmeans.cluster_centers_
#
#
#plt.scatter(df['SPACEID'],df['PHONEID'],c=labels,marker='.')
#plt.scatter(df['LONGITUDE'],df['LATITUDE'],c=labels,marker='.')
#plt.figure()
#plt.scatter(n_clusters[:,0],n_clusters[:,1], c='black',marker='o', s=100, alpha=0.2)
#plt.title('K-means Clustering Algorthm, one week of Wifi data')
#plt.xlabel('SPACE ID')
#plt.ylabel('PHONE ID')
#plt.show()
#plt.show()

# #############################################################################
# Plot result

#plt.close('all')
#plt.figure(1)
#plt.clf()
#
#for k, col in zip(range(n_clusters_), colors):
#    class_members = labels == k
#    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=14)
##    for x in X[class_members]:
##        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#


################################################################################
#Met = metrics.silhouette_score(df[['Position', 'Count']], y_pre, metric='euclidean')
#print('Silhouette Coefficient:',Met)
#
#print("Processing time: %s seconds" % (time.time() - start_time))    
#
#
##Profiling and memory usage--------------------------------------------------
#process = psutil.Process(os.getpid())
#print("Memory Consumption:",process.memory_info().rss, 'bytes')   
#M= process.memory_info().rss /1000000
#print('(Or Megabyte:', M,')')
#    
##Profiling and runtime calculation--------------------------------------------
#cProfile.run('re.compile("foo|bar")')
# =============================================================================
#print("Processing time: %s seconds" % (time.time() - start_time))
#
##Profiling and memory usage--------------------------------------------------
#process = psutil.Process(os.getpid())
#print("Memory Consumption:",process.memory_info().rss, 'bytes')   
#M= process.memory_info().rss /1000000
#print('(Or Megabyte:', M,')')
#
#
#from sklearn.metrics import davies_bouldin_score
#print('Calinski-Harabasz Index:',metrics.calinski_harabasz_score(X, labels))  
#print('Silhouette Coefficient:',metrics.silhouette_score(X, labels, metric='euclidean'))
#print('Davies-Bouldin Index:',davies_bouldin_score(X, labels))
# =============================================================================
