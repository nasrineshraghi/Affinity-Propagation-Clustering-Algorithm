# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:58:39 2020

@author: neshragh
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans



#df = pd.read_csv(r"ecounter_time.csv" )

df = pd.read_csv('C:/Users/neshragh/OneDrive - University of New Brunswick/UNB_thesis_Work/Affinity_Sample_SPY/1-Dataset_ecounter/Ecounter_Dataset/weekly/1weekduring.csv')

#df = pd.read_csv('29apriloneweek.csv')

#df = pd.read_csv('onedaynotime.csv')
# #############################################################################

#Choose data for algorithm
#df = df.loc[(df['Date'] > '4/29/2019') & (df['Date'] < '5/5/2019')] 

X= df.loc[df.index,['Position','Count']].to_numpy()
num_clusters = 6
kmeans = KMeans(n_clusters = num_clusters).fit(X)
labels = kmeans.labels_
#
n_clusters = kmeans.cluster_centers_
distortions = []

#K = range(1,10)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k)
#    kmeanModel.fit(X)
#    distortions.append(kmeanModel.inertia_)

#plt.plot(K, distortions, 'm*-')
#plt.title('Elbow Method with distortion')
#plt.xlabel('Value of k')
#plt.ylabel('Distortion')
#plt.vlines(4,0,25000,colors='red',linestyles ="dashed")
#plt.grid()
#plt.show()


###############################################################################
#elbow 2 

from yellowbrick.cluster import KElbowVisualizer
model = kmeans
visualizer = KElbowVisualizer(model, k=(4,12))
plt.figure(11)
visualizer.fit(X)       
visualizer.show()  






















#
#
#
#
#X= df.loc[df.index,['Position','Count']].to_numpy()
#num_clusters = 3
#kmeans = KMeans(n_clusters = num_clusters).fit(X)
#labels = kmeans.labels_
#n_clusters_ = kmeans.cluster_centers_
#
#
#################   1
#distortions = []
#K = range(1,10)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k)
#    kmeanModel.fit(df)
#    distortions.append(kmeanModel.inertia_)
#
#plt.plot(K, distortions, 'bx-')
#
#plt.show()
#



####################################################
######   2
#from yellowbrick.cluster import KElbowVisualizer
#model = kmeans
#visualizer = KElbowVisualizer(model, k=(4,12))
#
#visualizer.fit(X)       
#visualizer.show()  

