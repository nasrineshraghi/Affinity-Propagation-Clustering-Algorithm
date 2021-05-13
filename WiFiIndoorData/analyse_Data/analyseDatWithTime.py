# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:22:01 2020

@author: neshragh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AffinityPropagation
from PIL import Image, ImageDraw, ImageFont
import datetime
import time



df = pd.read_csv('C:/Users/neshragh/OneDrive - University of New Brunswick/UNB_thesis_Work/Affinity_Sample_SPY/2-Dataset_Wifi/archive/original/TrainingData.csv')

#df = df.loc[(df['Date'] > '4/29/2019') & (df['Date'] < '5/5/2019')] 

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


######## @@@@@@@@@@@@@ Add time date to dataset
#df['DATETIME'] = pd.to_datetime(df['TIMESTAMP'], unit = 's')
#df['DATE'] = pd.to_datetime(df['DATETIME'], format='%Y:%M:%D').dt.date
# =============================================================================
# isn =df.isnull().sum()
# print(isn)
# print(df.isnull())
# =================

#Building Map

#df_1 = df[(df['FLOOR'] == 1) & (df['BUILDINGID'] == 0)]
#plt.scatter(df_1.LONGITUDE,df_1.LATITUDE)
#df_1 = df[(df['FLOOR'] == 1) & (df['BUILDINGID'] == 1)]
#plt.scatter(df_1.LONGITUDE,df_1.LATITUDE)
#df_1 = df[(df['FLOOR'] == 1) & (df['BUILDINGID'] == 2)]
#plt.scatter(df_1.LONGITUDE,df_1.LATITUDE)
#
##generate building map in the loop
#for i in range(0,4):
#    for j in range(0,2):
#        df_1 = df[(df['FLOOR'] == i) & (df['BUILDINGID'] == j)]
#        plt.scatter(df_1.LONGITUDE,df_1.LATITUDE, color= 'firebrick')
#        #show seprately
#        plt.show()
#        
#
#######Drop Empty Rows or Columns
#df.dropna(axis=0,how='any', thresh=None, subset=None, inplace=True)
#
######drop with criteria
#df = df[df.WAP001 == '100']
#
#    
#for i in range(0,3):
#    b = df.iloc[:, i]
#    print(b)
#    con = b.iloc[i] == 100
#    if con:
#        con.drop()
#fig = plt.figure(3)
#
#plt.scatter(df.BUILDINGID, df.FLOOR)



#############################################################################
c = df.LATITUDE
b = df.FLOOR
a = df.LONGITUDE



############ 3D Plotting 
fig = plt.figure(2)
#ax = Axes3D(fig)
#for i in range(0,10):
#    ax.scatter(df.LONGITUDE, df.LATITUDE, df.iloc[:,i], s=20)
#plt.xlabel('Long',labelpad=15)
#plt.ylabel('Lat',labelpad=15)
#
#
#plt.show()
#df.astype(int)


am = Axes3D(fig)
am.scatter(c,b, a ,s=20, color= 'royalblue')


am.set_xlabel('Latitude',labelpad=10)
am.set_ylabel('Floor',labelpad=10)
am.set_zlabel('Longitude',labelpad=10)

plt.show()

###############################################################################

#BUILDINGID
#LATITUDE
#LONGITUDE
#PHONEID
#SPACEID
#FLOOR\
#TIMESTAMP
#RELATIVEPOSITION


c = df.BUILDINGID
b = df.TIMESTAMP
a = df.SPACEID

############ 3D Plotting 
fig = plt.figure(21)
am = Axes3D(fig)
pl = am.scatter(a,b,c ,s=20,  c = b, cmap = 'jet')

am.set_xlabel('SPaceId',labelpad=10)
am.set_ylabel('FLOOR',labelpad=10)
am.set_zlabel('BUILDINGID',labelpad=10)

fig.colorbar(pl, shrink = 0.75)
am.view_init(180, 0)
plt.show()

##############################################################################

fig = plt.figure(4)
plt.scatter(b,a, color= 'royalblue')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()
##############################################################################

 ####### TIME + All data point 

R = np.unique(df["Date"])
RR = pd.to_datetime(R)
fe = pd.to_datetime(df.Date)
ni = np.zeros(len(R))
for i in range(len(R)):
    print("")
#    u= df["DATE"] == '04-06-13'
#    pp = df.loc[(df.DATE == '2013-06-20')] 
    ni[i] = len(df.Date.loc[fe == RR[i]])
    
    
fig = plt.figure(22)
plt.bar(R,ni,color=['k', 'g', 'b','c', 'r', 'm'], width=1, align='center', capsize=5, alpha=0.5)
plt.title('Data distribution during the experiment')
plt.ylabel('Number of Data points')
plt.xlabel('Date')


plt.show()

    



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





df_1 = df[ (df['BUILDINGID'] == 0)]
plt.scatter(df_1.LONGITUDE,df_1.LATITUDE, color ='yellowgreen', label = 'Building T1')
df_1 = df[ (df['BUILDINGID'] == 1)]
plt.scatter(df_1.LONGITUDE,df_1.LATITUDE, color ='deepskyblue', label = 'Building TD')
df_1 = df[ (df['BUILDINGID'] == 2)]
plt.scatter(df_1.LONGITUDE,df_1.LATITUDE, color = 'deeppink', label = 'Building TC')
plt.grid()
#plt.title('Position of each bulding')
plt.xlabel('LONGITUDE')
plt.ylabel('LATITUDE')
plt.legend()















