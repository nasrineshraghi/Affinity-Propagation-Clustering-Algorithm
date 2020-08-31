
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:43:34 2020
Micro and Macro data stream clustering

@author: neshragh
"""

from sklearn.cluster import AffinityPropagation
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import os
import datetime


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
    
#-------------------------------------------------------------------------------    
#d_parser = lambda x:pd.datetime.strptime(x, '%m/%d/%Y')
#df_dat = pd.read_csv('week1.csv', parse_dates=['Date'], date_parser=d_parser)
#using Date time format
#df_dat['Date'] = pd.to_datetime(df_dat['Date'])
#df_dat['Time'] = pd.to_datetime(df_dat['Time'], format='%H:%M:%S')
#min, max and boundary of dates in dataset
    #print(df_dat['Date'].min())
    #print(df_dat['Date'].max())
    #print(df_dat['Date'].max() - df_dat['Date'].min())
    #set an index
    #df_dat.set_index('Date', inplace=True)
#dtrang = (df_dat['Date'] >= pd.to_datetime(fstr)) &(df_dat['Date'] <= pd.to_datetime(fend))
#dtrang = (df_dat['Date'] >= pd.to_datetime('4/29/2019')) &(df_dat['Date'] < pd.to_datetime('5/1/2019'))
#print(df_dat.loc[dtrang])


#df = df_dat.loc[(df_dat.Date == '4/29/2019') & (df_dat.Time >= '10:00:00') & (df_dat.Time <= '11:00:00')]
# df = dc.loc[(dc.Date == '4/29/2019') & (dc.Time >= '10:00:00') & (dc.Time <= '11:00:00')]
 #-----------------------------------------------------------------   
##open a dataset
dat = input("enter the dataset name with .csv:")
df_dat = pd.read_csv(dat)

    
# =============================================================================
# str = input("Enter start Date(Format:mm/dd/yyyy): ")
# end = input("Enter End Date(Format:mm/dd/yyyy): ")
# =============================================================================

start_date = pd.to_datetime('05-02-2019')
e_date = pd.to_datetime('05-03-2019')

for j in pd.date_range(start_date,e_date):
    print(j)

dtrang = df_dat.loc[(df_dat.Date >= '5/13/2019') & (df_dat.Date <= '5/14/2019')]
#dtrang = df_dat.loc(df_dat['Date'] >= '4/29/2019') &(df_dat['Date'] < '5/1/2019')




fname = input("Enter start time: ")
fdamping = input("Enter Damping Factor Value for Micro clusters(BETWEEN 0.5-0.99): ")
hrstrng = fname
gg=hrstrng
#hrstrng = gg =  7
#fname

for i in range(10,11):

# =============================================================================
#     hrstrng = hrstrng.replace(hrstrng[0:2],format(i,'02d'))
#     nhr =     hrstrng.replace(hrstrng[0:2],format(i +1,'02d'))
# =============================================================================

    if i<10:
        hrstrng = hrstrng.replace(hrstrng[0:1],str(i))
        nhr =     hrstrng.replace(hrstrng[0:1],str(i +1))
        gg =      nhr
    
    else:
            hrstrng = gg.replace(hrstrng[0:2],str(i))
            gg=hrstrng
            nhr =     hrstrng.replace(hrstrng[0:2],str(i +1))
            
    df = dtrang.loc[(dtrang.Time >= hrstrng) & (dtrang.Time <= nhr)]
    
        
    #AP
    X= df.loc[df.index<90000,['Position','Count']].to_numpy()
    #Damp_my_default:.88 var: float(fdamping)
    af = AffinityPropagation(preference=-50, damping=float(fdamping), max_iter= 100 ).fit(X)
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
  
        
# =============================================================================
#     df = pd.read_table('micro_cluster_data.csv', header=None, delim_whitespace=True, skiprows=1)
#     df.columns = ['position','ID']
# 
# # filtering out the rows with `POSITION_T` value in corresponding column
#     df = df[df.POSITION_T.str.contains('POSITION_T') == False]
# 
# =============================================================================
    
    ################################################################################
    
"""
Stage 2:  Macro clustering:
    Generating Macro based on the prev level:Micro Clustering
    Save in the seprate csv file    
"""
    
    
mac = pd.read_csv("micro_cluster_data.csv", names = ["Date", "Position" , "Count" ])
X= mac.loc[mac.index<90000,['Position','Count']].to_numpy()
af = AffinityPropagation(preference=-50, damping=.6, max_iter= 100 ).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
#week1=.6
#wek2
n_clusters_ = len(cluster_centers_indices)


#plot
plt.close('all')
plt.figure(1)
plt.clf()
plt.scatter(mac.Position, mac.Count, linewidth=3,facecolors='none', s=120, edgecolor="silver")
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    #plt.plot(X[class_members, 0], X[class_members, 1], col + '+', markersize=8)
    plt.plot(cluster_center[0], cluster_center[1], '+', markerfacecolor=col,
             markeredgecolor='crimson', markersize=40 )
    
    
plt.title('Estimated number of clusters for Macro Clustering: %d' % n_clusters_)

plt.xlabel('Postions',labelpad=15)
plt.ylabel('Number of people')

frame =plt.gca()
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)

plt.ylim(-0.7,15)

plt.xticks([0.80,1,2,3,4,5,6,6.20], ["","Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
           "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth",""])

plt.show()

exc_my = pd.DataFrame(X[cluster_centers_indices], columns= ['Position','Count']).T
#change the row and colomns directions
exc_my = exc_my.transpose()
exc_my.to_csv( "macro_cluster_data.csv",index =False)



# =============================================================================
# index = 0
# while index < len(df):
#     mf = df[Time]
#     index = index +1
# =============================================================================
# reset index from 0 to n
#df = df.reset_index(drop = True, inplace = True)

# =============================================================================
# def daterange(start_date, end_date):
#     for n in range(int((end_date - start_date).days)):
#         yield start_date + timedelta(n)
#         
#  
# #start_date = date(2013, 1, 1)
# #end_date = date(2015, 6, 2)
# 
#        
# Sd = df.loc[df.Date == '4/29/2019']
# start_date = Sd
# 
# Ed = df.loc[df.Date == '5/1/2019']
# end_date = Ed
# 
# 
# 
# 
# for single_date in daterange(start_date, end_date):
#     print(single_date.strftime("%Y-%m-%d"))
# # df['StartDate'] = pd.to_datetime(df['StartDate'])
# # df['EndDate'] = pd.to_datetime(df['EndDate'])
# # df.EndDate = pd.to_datetime(df.EndDate)
# # df.StartDate = pd.to_datetime(df.StartDate)
# # df = df.set_index('StartDate')
# # new_df = pd.DataFrame()
# # for i, data in df.iterrows():
# #     data = data.to_frame().transpose()
# #     data = data.reindex(pd.date_range(start=data.index[0], end=data.EndDate[0])).fillna(method='ffill').reset_index().rename(columns={'index': 'StartDate'})
# #     new_df = pd.concat([new_df, data])
# # 
# # =============================================================================
# start_tim = df.loc[df['Time'] == '7:00:00'] 
# end_tim = df.loc[(df.Time == '9:00:00')]
# 
# 
# for row in df.itertuples(index=True, name='Pandas'):
#     print(start_tim)
#    
# =============================================================================
# for i in dc:
#     for j in df:
#         X= df.loc[df.index<90000,['Position','Count']].to_numpy()
#         af = AffinityPropagation(preference=-50, damping=.88, max_iter= 100 ).fit(X)
#         cluster_centers_indices = af.cluster_centers_indices_
#         labels = af.labels_
#         n_clusters_ = len(cluster_centers_indices)


