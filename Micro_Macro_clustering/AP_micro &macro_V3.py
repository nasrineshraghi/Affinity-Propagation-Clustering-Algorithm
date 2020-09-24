
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:43:34 2020
Micro and Macro data stream clustering

@author: neshragh
"""

from sklearn.cluster import AffinityPropagation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import datetime
import os


##################################################################
"""
Stage 1: Micro Clustering: based on original dataset and Affinity propagation algorithm
Number of Clusters can vary based on the damping factor

"""

#delete the prev csv file created from the last run
if os.path.exists('micro_cluster_data.csv'):
    os.remove("micro_cluster_data.csv")
if os.path.exists('macro_cluster_data.csv'):
    os.remove("macro_cluster_data.csv")
    
##open a dataset
df_dat = pd.read_csv('week4event.csv')

    
# =============================================================================
# str = input("Enter start Date(Format:mm/dd/yyyy): ")
# end = input("Enter End Date(Format:mm/dd/yyyy): ")
# =============================================================================

df_dat['Date'] = pd.to_datetime(df_dat['Date'], format='%m/%d/%Y')

#dtrang = df_dat.loc[(df_dat.Date >= '2019-05-08') & (df_dat.Date <= '2019-05-10')]

#dtrang= df_dat[(df_dat['Date'] == '2019-05-10')]
dtrang= df_dat[(df_dat['Date']>=datetime.date(2019,4,29)) & (df_dat['Date']<=datetime.date(2019,5,5))]
#dtrang = df_dat.loc(df_dat['Date'] >= '4/29/2019') &(df_dat['Date'] < '5/1/2019')
#print(df_dat['Date'][5] )

# =============================================================================
# start_date = pd.to_datetime('05-02-2019')
# e_date = pd.to_datetime('05-03-2019')
# 
# for j in pd.date_range(start_date,e_date):
#     print(j)
# =============================================================================

# =============================================================================
# fname = input("Enter start time: ")
# fdamping = input("Enter Damping Factor Value for Micro clusters(BETWEEN 0.5-0.99): ")
# =============================================================================

hrstrng = '7:00:00'
gg=hrstrng
#hrstrng = gg =  7
#fname

for i in range(7,19):
    print(str(i))
    hrstrng = str(i) + ':00:00'
    nhr = str(i+1) + ':00:00'

# =============================================================================
#     hrstrng = hrstrng.replace(hrstrng[0:2],format(i,'02d'))
#     nhr =     hrstrng.replace(hrstrng[0:2],format(i +1,'02d'))
# =============================================================================

    if i<10 & i+1>=10:
        hrstrng = '0'+ hrstrng
        
    
# =============================================================================
#     if i+1 < 10:
#         nhr= '0' + nhr
# =============================================================================
        
             
    print(hrstrng)
            
    df = dtrang.loc[(dtrang.Time >= hrstrng) & (dtrang.Time <= nhr)]
    
        
    #AP
    #X= df.loc[df.index>90000,['Position','Count']].to_numpy()
    X = df.to_numpy()
    X = X[:,[2, 6]]
    #Damp_my_default:.88 var: float(fdamping)
    af = AffinityPropagation(preference=-50, damping=.8888888, max_iter= 100 ).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    
    #Plotting the result
    plt.close('all')
    plt.figure(1)
    plt.clf()
    plt.scatter(df.Position, df.Count, color='c',  linewidth=4)
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

    for k, col in zip(range(n_clusters_), colors):
         class_members = labels == k
         cluster_center = X[cluster_centers_indices[k]]
         #plot branches
         plt.plot(X[class_members, 0], X[class_members, 1], col + '.', markersize=10)
         plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
              markeredgecolor='k', markersize=15)
         
         #plot:lines between branches
         for x in X[class_members]:
             plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
        
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.xlabel('Postions',labelpad=15)
    plt.ylabel('Number of people')
    
    frame =plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.axes.get_yaxis().set_visible(True)
    
    #x and y axis values
    plt.ylim(-0.7,5)
    
    plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
               "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])
    
    plt.show()
    
    
    #save the result in the excel file
    exc = pd.DataFrame(X[cluster_centers_indices], columns= ['Position','Count']).T
    #change the row and colomns directions
    exc = exc.transpose()
    #add pandas data to excisting csv file mode a is append
     
    exc.to_csv( "micro_cluster_data.csv", mode='a', header = False)
   # exc.drop(exc[], inplace=True, axis=1)
    #exc = exc.drop(1, axis=1)
  
        

    ################################################################################
    
"""
Stage 2:  Macro clustering:
    Generating Macro based on the prev level:Micro Clustering
    Save in the seprate csv file    
"""
    
    
mac = pd.read_csv("micro_cluster_data.csv",  header=None)
#X= mac.loc[mac.index<90000,['Position','Count']].to_numpy()
mac = mac.to_numpy()
mac =np.delete(mac,0,axis=1)
#X = X.astype(float )
#X = X[:,[1, 2]]
#X = np.array(X)
aaf = AffinityPropagation(preference=-50, damping=.6456, max_iter= 100 ).fit(mac)
#87-833
cluster_centers_indices = aaf.cluster_centers_indices_
labels = aaf.labels_
#week1=.6
#wek3=77
n_clusters_ = len(cluster_centers_indices)


#plot
plt.close('all')
plt.figure(1)
plt.clf()
plt.scatter(mac[:,1], mac[:,0], linewidth=3,facecolors='none', s=120, edgecolor="silver")
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = mac[cluster_centers_indices[k]]
    #plt.plot(X[class_members, 0], X[class_members, 1], col + '+', markersize=8)
    plt.plot(cluster_center[1], cluster_center[0], '+', markerfacecolor=col,
             markeredgecolor='crimson', markersize=40 )
    
    
#plt.title('Estimated number of clusters for Macro Clustering: %d' % n_clusters_)
plt.title('Follow-up Intervention Month')
#plt.title('%d' % n_clusters_)


plt.xlabel('Postions',labelpad=15)
plt.ylabel('Number of people')

frame =plt.gca()
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)

plt.ylim(0,10)

plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
           "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])

plt.show()

exc_my = pd.DataFrame(X[cluster_centers_indices], columns= ['Position','Count']).T
#change the row and colomns directions
exc_my = exc_my.transpose()
exc_my.to_csv( "macro_cluster_data.csv",index =False)




