# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:00:29 2021

@author: neshragh
"""

import pandas as pd
import time
start_time = time.time()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from itertools import cycle
import matplotlib.dates as mdates
import datetime
import os
import cProfile
import re
import time
import psutil
from sklearn import metrics
start_time = time.time()
from sklearn.metrics import davies_bouldin_score
import matplotlib
matplotlib.use('Qt5Agg')
##################################################################

#open a dataset
#df = pd.read_csv('ecounter.csv')
df = pd.read_csv('29apriloneweek.csv')#one day

#df = pd.read_csv('monthIntervention.csv')#one month intervention
#df = pd.read_csv('C:/Users/neshragh/OneDrive - University of New Brunswick/UNB_thesis_Work/Affinity_Sample_SPY/1-Dataset_ecounter/Ecounter_Dataset/weekly/1weekbeforeAllData.csv')
#df = pd.read_csv('C:/Users/neshragh/OneDrive - University of New Brunswick/UNB_thesis_Work/Affinity_Sample_SPY/1-Dataset_ecounter/Ecounter_Dataset/weekly/1weekbeforeAllData.csv')

df = pd.read_csv('C:/Users/neshragh/OneDrive - University of New Brunswick/UNB_thesis_Work/Affinity_Sample_SPY/1-Dataset_ecounter/data analysis/1monthbefore.csv')
# #############################################################################

#Choose data for algorithm
#df = dff.loc[(dff['Date'] =='4/29/2019') & (dff['Date'] <= '5/5/2019')] 
#df = df.loc[(df.Date >= '5/3/2019') & (df.Date <= '5/4/2019')]

###@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#Time + Count      DURING INTERVEn

#df['t'] = pd.to_datetime(df['Time'])
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot_date(df['t'], df['Count'], color = 'darkcyan')
#myFmt = mdates.DateFormatter('%H:%M')
#ax.xaxis.set_major_formatter(myFmt)
#
#plt.title(' stairs usage hourly: APril 29- May 5')
#plt.ylabel('Number People')
#plt.xlabel('Hours of the Day')

####################### Before

#df['t'] = pd.to_datetime(df['Time'])
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot_date(df['t'], df['Count'], color = 'tomato')
#myFmt = mdates.DateFormatter('%H:%M')
#ax.xaxis.set_major_formatter(myFmt)
#
#plt.title(' stairs usage hourly: March 18- March 24')
#plt.ylabel('Number People')
#plt.xlabel('Hours of the Day')
#
#
###################### AFTER
#
#df['t'] = pd.to_datetime(df['Time'])
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot_date(df['t'], df['Count'], color = 'royalblue')
#myFmt = mdates.DateFormatter('%H:%M')
#ax.xaxis.set_major_formatter(myFmt)
#
#plt.title(' stairs usage hourly: May 27 - June 2')
#plt.ylabel('Number People')
#plt.xlabel('Hours of the Day')
 
###############################################################################

##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




###### position+ Count

#plt.scatter(df['Position'], df['Count'], color = 'cornflowerblue',edgecolor='steelblue')
#plt.title(' stairs usage vs., number of People:  April 29 - May 5')
#plt.ylabel('Number People')
#plt.xlabel('Positions')
#plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
#            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])


################## After
#
#plt.scatter(df['Position'], df['Count'], color = 'royalblue',edgecolor='royalblue')
#plt.title(' stairs usage vs number of People:  May 27 - June 2')
#plt.ylabel('Number People')
#plt.xlabel('Positions')
#plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
#            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])


############## Before

##
#plt.scatter(df['Position'], df['Count'], color = 'tomato',edgecolor='tomato')
#plt.title(' stairs usage vs number of People:  March 18- March 24')
#plt.ylabel('Number People')
#plt.xlabel('Positions')
#plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
#            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])



############################################################################


####, color = 'cornflowerblue',edgecolor='steelblue'
### color = 'darkcyan',edgecolor='darkgreen'

##################################################### count histogram one day
#
#position = -1
#count = -3
#data = df.values
#df.describe()
#numData = 1126
#N = np.max(data[: , count])
#
#
#
#
#bins = np.arange(13)+1 - 0.5
#plt.hist(data[: , count] , bins=bins, color = 'darkcyan',edgecolor='darkgreen' )
###!!color = 'darkorchid',edgecolor='red'
#plt.yscale('log')
##plt.xticks(range(N))
##plt.yticks([5,10,50,100,500,1000,4900],[5,10,50,100,500,1000,4900])
##plt.title('Hisogram of Number of people in April 29')
###plt.ylabel('Number of Data points')
#plt.xlabel('Number of People')
#plt.ylabel('Data Distribution')
#plt.title('Histogram of Count: April 29- May 5')
#



########## Before

#
#position = -2
#count = -4
#data = df.values
#df.describe()
#numData = 6421
#N = np.max(data[: , count])
##
#bins = np.arange(31)+1 - 0.5
#plt.hist(data[: , count] , bins=bins, color = 'tomato',edgecolor='darkred' )
###!!color = 'darkorchid',edgecolor='red'
##plt.yscale('log')
##plt.xticks(range(N))
##plt.yticks([5,10,50,100,500,1000,4900],[5,10,50,100,500,1000,4900])
##plt.title('Hisogram of Number of people in April 29')
###plt.ylabel('Number of Data points')
#plt.xlabel('Number of People')
#plt.ylabel('Data Distribution')
#plt.title('Histogram of Count: March 18- March 24')


############## After
#position = -2
#count = -4
#data = df.values
#df.describe()
#numData = 6421
#N = np.max(data[: , count])
#
#position = -2
#count = -4
#data = df.values
#df.describe()
#numData = 6421
#N = np.max(data[: , count])
#bins = np.arange(20)+1 - 0.5
#plt.hist(data[: , count] , bins=bins, color = 'royalblue',edgecolor='mediumblue' )
###!!color = 'darkorchid',edgecolor='red'
##plt.yscale('log')
##plt.xticks(range(N))
##plt.yticks([5,10,50,100,500,1000,4900],[5,10,50,100,500,1000,4900])
##plt.title('Hisogram of Number of people in April 29')
###plt.ylabel('Number of Data points')
#plt.xlabel('Number of People')
#plt.ylabel('Data Distribution')
#plt.title('Histogram of Count: May 27- June 2')


############### one day
##
#position = -1
#count = -3
#data = df.values
#df.describe()
#numData = 1126
#N = np.max(data[: , count])
#
#bins = np.arange(13)+1 - 0.5
#plt.hist(data[: , count] , bins=bins, color = 'cornflowerblue',edgecolor='steelblue' )
###!!color = 'darkorchid',edgecolor='red'
##plt.yscale('log')
##plt.xticks(range(N))
##plt.yticks([5,10,50,100,500,1000,4900],[5,10,50,100,500,1000,4900])
##plt.title('Hisogram of Number of people in April 29')
###plt.ylabel('Number of Data points')
#plt.xlabel('Number of People')
#plt.ylabel('Data Distribution')
#plt.title('Hisogram of Count: May 1')
#plt.xticks([1,2,3,4,5,6,7,8,9,11],[1,2,3,4,5,6,7,8,9,11])





##########################################################################
##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#df = df.loc[(df['Count'] >= 4) & (df['Count'] <= 7) ]



########DURING

#df = df.loc[(df['Count']== 1) | (df['Count']== 2) |(df['Count']== 3) |(df['Count']== 4) ]
#position = -1
#count = -3
#data = df.values
#df.describe()
#numData = 1126


#N = len(set(data[: , position]))
#bins = np.arange(7)+1 - 0.5
##rang = range(N-1)
#plt.hist(data[: , position] , bins=bins, color = 'darkcyan',edgecolor='darkgreen')
##plt.xticks(range(N))
#plt.title('Hisogram of postions: April 29- May 5')
#plt.ylabel('Number of Data points')
#plt.xlabel('Position')
#plt.ylabel('Data Distribution')
#plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
#            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])



#######BEFORE
#
#position = -2
#count = -4
#data = df.values
#df.describe()
#numData = 6421
#N = np.max(data[: , count])
#
#N = len(set(data[: , position]))
#bins = np.arange(7)+1 - 0.5
##rang = range(N-1)
#plt.hist(data[: , position] , bins=bins, color = 'tomato',edgecolor='darkred')
##plt.xticks(range(N))
#plt.title('Hisogram of postions: March 18- March 24')
#plt.ylabel('Number of Data points')
#plt.xlabel('Position')
#plt.ylabel('Data Distribution')
#plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
#            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])




#####  AFTER

#position = -2
#count = -4
#data = df.values
#df.describe()
#numData = 6421
#N = np.max(data[: , count])
#
#N = len(set(data[: , position]))
#bins = np.arange(7)+1 - 0.5
##rang = range(N-1)
#plt.hist(data[: , position] , bins=bins, color = 'royalblue',edgecolor='mediumblue')
##plt.xticks(range(N))
#plt.title('Hisogram of postions: May 27 - June 2')
#plt.ylabel('Number of Data points')
#plt.xlabel('Position')
#plt.ylabel('Data Distribution')
#plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
#            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])








###### hostogram of all data points
#N = len(set(data[: , position]))
#plt.hist(data[: , position] , bins =N, color = 'darkorchid',edgecolor='red')
##plt.xticks(range(N))
#plt.title('Hisogram of postions in one day')
##plt.ylabel('Number of Data points')
#plt.xlabel('Position')
#plt.ylabel('Data Distribution')
##



############# kmeans

#X= df.loc[df.index,['Position','Count']].to_numpy()
#
#num_clusters = 4
#kmeans = KMeans(n_clusters = num_clusters).fit(X)
#labels = kmeans.labels_
#
#n_clusters = kmeans.cluster_centers_
#
##plt.scatter(df['LONGITUDE'],df['LATITUDE'],c=labels,marker='o')
#
##plt.scatter(df['uqid'],df['PHONEID'],c=labels,marker='.')
#
#plt.close('all')
#plt.figure(1)
#plt.clf()
#
#colors = cycle('rygmykbgrcmykbgrcmykbgrcmyk')
#for k, col in zip(range(num_clusters), colors):
#    class_members = labels == k
#    plt.plot(X[class_members, 0], X[class_members, 1], col + '.',markersize=5)
#    plt.plot(n_clusters[k,0], n_clusters[k,1], 'o', markerfacecolor=col,
#             markeredgecolor=col , markersize=12)
##plt.scatter(n_clusters[:,0],n_clusters[:,1], c='red' ,marker='o', s=200, alpha=0.2)
#
#plt.title('K-means Clustering Algorthm: e-counter data')
#plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
#            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])
#plt.xlabel('Position')
#plt.ylabel('Number of People')
#plt.show()
#
#
#
#
#
#print("Processing time: %s seconds" % (time.time() - start_time))
#
##Profiling and memory usage--------------------------------------------------
#process = psutil.Process(os.getpid())
#print("Memory Consumption:",process.memory_info().rss, 'bytes')   
#M= process.memory_info().rss /1000000
#print('(Or Megabyte:', M,')')
#
#cProfile.run('re.compile("foo|bar")')
#print('Silhouette Coefficient Macro:',metrics.silhouette_score(X, labels, metric='euclidean'))
#print('Calinski-Harabasz Index Macro:',metrics.calinski_harabasz_score(X, labels))  
#print('Davies-Bouldin Index Macro:',davies_bouldin_score(X, labels))









################################### elbow K-means
#num_clusters = 10
#kmeans = KMeans(n_clusters = num_clusters).fit(X)
#labels = kmeans.labels_
##
#n_clusters = kmeans.cluster_centers_
#distortions = []
#
#K = range(1,10)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k)
#    kmeanModel.fit(X)
#    distortions.append(kmeanModel.inertia_)
#
#plt.plot(K, distortions, 'm*-')
#plt.title('Elbow Method with distortion')
#plt.xlabel('Value of k')
#plt.ylabel('Distortion')
#plt.vlines(4,0,120000,colors='red',linestyles ="dashed")
#plt.grid()
#plt.show()
#plt.scatter(X[:,0],X[:,1],c=labels,marker='o')

###################################

####################################################
#X= df.loc[df.index,['Position','Count']].to_numpy()
#
#af = AffinityPropagation(preference=-1, damping=.98, max_iter= 100 ).fit(X) 
#
#cluster_centers_indices = af.cluster_centers_indices_
#labels = af.labels_
#
#n_clusters_ = len(cluster_centers_indices)
#
#print('Estimated number of clusters: %d' % n_clusters_)
#plt.scatter(X[:,0],X[:,1],c=labels,marker='o')
##plt.scatter(n_clusters[:,0],n_clusters[:,1], c='black',marker='o', s=100, alpha=0.2)
#plt.title('K-means Clustering Algorthm, one week of Wifi data')
#plt.xlabel('SPACE ID')
#plt.ylabel('PHONE ID')
#plt.show()
#
##################################################
#plt.scatter(df['LONGITUDE'],df['LATITUDE'],c=labels,marker='o')
#
#plt.scatter(df['uqid'],df['PHONEID'],c=labels,marker='.')
#
#plt.scatter(df['Position'], df['Count'],color = 'darkorchid')
##
#plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
#            "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])
#
##plt.title('Data Distribution of one day')
#plt.xlabel('Position')
#plt.ylabel('Number of People')
#plt.show()
#
#






#plt.show()

##############################################################################


#
#from sklearn.cluster import AffinityPropagation
##from sklearn import metrics
#
#dc = pd.read_csv('Monday.csv')
#
#df = dc.loc[(dc.Time >= '18') & (dc.Time <= '19')] 
#X = df[['Count', 'Position']] .values

#fix until this date
# =============================================================================
# dc=df_data.loc[(df_data.Sensor =='UP')]
# print(dc)
# =============================================================================
#Filter dataset
#data = df_data[['Date', 'Time','Count', 'Status', 'Sensor', 'Type', 'Position','Location', 'Location Code']] .values
#df_filtered = data[(df_data.Time >='11') & (df_data.Time <='11:30') ]
##print(len(df_filtered))
#print(df_filtered)

#Split Apply Combine
#g = df_data.groupby('Time')
#for Time, Time_df_data in g :
#    print(Time_df_data)
#    print(Time)
#g.get_group('Time' == '9:59:51' )

# #############################################################################

#Choose data for algorithm
#X= df.loc[df.index<90000,['Position','Count']].to_numpy()

#labels_true = dc.loc[df.index<90000,'Time'].to_numpy()


# #############################################################################
