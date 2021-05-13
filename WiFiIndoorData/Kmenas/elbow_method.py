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



#df = pd.read_csv(r"wifi.csv" )
df = pd.read_csv('C:/Users/neshragh/ecounter/Affinity_Sample_SPY/2-Dataset_Wifi/archive/TrainingDataWithUniQID.csv')



X= df.loc[df.index,['uqid','PHONEID']].to_numpy()
num_clusters = 3
kmeans = KMeans(n_clusters = num_clusters).fit(X)
labels = kmeans.labels_
n_clusters_ = kmeans.cluster_centers_


################   1
#distortions = []
#K = range(1,10)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k)
#    kmeanModel.fit(df)
#    distortions.append(kmeanModel.inertia_)
#
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('Position')
#plt.ylabel('Count')
#plt.title('Elbow Method')
#plt.show()




####################################################
######   2
from yellowbrick.cluster import KElbowVisualizer
model = kmeans
visualizer = KElbowVisualizer(model, k=(4,12))

visualizer.fit(X)       
visualizer.show()  

