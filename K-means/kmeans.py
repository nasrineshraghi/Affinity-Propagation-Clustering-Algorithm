# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:01:18 2020

@author: neshragh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics



dc = pd.read_csv("event.csv")
df = dc.loc[(dc.Date >= '4/29/2019') & (dc.Date <= '5/5/2019')] 
#df = dc.loc[(dc.Date == '4/29/2019') &  (dc.Time >= '10:00:00') &  (dc.Time <= '18:00:00')] 





km = KMeans(n_clusters = 4)
y_pre = km.fit_predict(df[['Position', 'Count']])
df['cluster'] = y_pre
df1 = df[df.cluster ==1]
df2 = df[df.cluster ==2]
df0 = df[df.cluster ==0]
df3 = df[df.cluster ==3]

plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])
plt.xlabel('Position')
plt.ylabel('Number of People')


plt.scatter(df1.Position, df1['Count'])
plt.scatter(df2.Position, df2['Count'])
plt.scatter(df0.Position, df0['Count'])
plt.scatter(df3.Position, df3['Count'], color= 'cyan')


Met = metrics.silhouette_score(df[['Position', 'Count']], y_pre, metric='euclidean')
print('Silhouette Coefficient:',Met)
    
    
# =============================================================================
# 
# k_r = range(1,10)
# sse =[]
# for k in k_r:
#     km = KMeans(n_clusters=k)
#     km.fit(df[['Position', 'Count']])
#     sse.append(km.inertia_)
#     
# plt.xlabel('K')
# plt.ylabel('Some of Squred Error')
# plt.plot(k_r, sse)   
# =============================================================================
