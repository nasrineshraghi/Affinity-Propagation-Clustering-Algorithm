# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 20:39:24 2021

@author: neshragh
"""

import numpy as np
import pandas as pd
import datetime as DT
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans




df = pd.read_csv('C:/Users/neshragh/ecounter/Affinity_Sample_SPY/2-Dataset_Wifi/archive/TrainingData.csv')
df = df.loc[(df['BUILDINGID'] == 1) ] 



#
#
############## ADDIBG UNIQ SPACEID




df['BUILDINGID']=df['BUILDINGID'].replace(0,3)
bi = df['BUILDINGID']
fl = df['FLOOR']
rp = df['RELATIVEPOSITION']
si = df['SPACEID']

#bi = bi*10000
#fl= fl*1000
#rp = rp *100
#uqid = bi+fl+rp+si



bi = bi*10000
fl= fl*1000
uqid = bi+fl+si



df['uqid'] = uqid
X= df.loc[df.index,['uqid','PHONEID']].to_numpy()


a = X[:,1]
b= X[:,0]
max_a = np.max(a)
max_b = np.max(b)
#a = (a*max_b)/(max_a)
b = (b*max_a)/(max_b)
X = np.array([[a,b]]) 
p=X.reshape(2,len(a))
X=p.T




num_clusters = 15
kmeans = KMeans(n_clusters = num_clusters).fit(X)
labels = kmeans.labels_

n_clusters = kmeans.cluster_centers_

#plt.scatter(df['LONGITUDE'],df['LATITUDE'],c=labels,marker='o')

#plt.scatter(df['uqid'],df['PHONEID'],c=labels,marker='.')
plt.scatter(X[:,1],X[:,0],c=labels,marker='*')

#plt.scatter(n_clusters[:,0],n_clusters[:,1], c='black',marker='o', s=100, alpha=0.2)
plt.title('K-means Clustering Algorthm, one week of Wifi data')
plt.xlabel('SPACE ID')
plt.ylabel('PHONE ID')
plt.show()


