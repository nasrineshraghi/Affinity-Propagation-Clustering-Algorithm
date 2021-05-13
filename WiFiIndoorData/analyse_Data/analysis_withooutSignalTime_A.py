# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 07:44:11 2021

@author: neshragh
"""

import numpy as np
import pandas as pd
import datetime as DT
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#df = pd.read_csv('TrainingData_order.csv')
df = pd.read_csv('C:/Users/neshragh/OneDrive - University of New Brunswick/UNB_thesis_Work/Affinity_Sample_SPY/2-Dataset_Wifi/archive/original/TrainingDataWithUniQID2.csv')
#df = df.loc[(df['BUILDINGID'] == 1) ] 
df = df.loc[(df['Date'] == '6/20/2013') ] 


data = df.values
df.describe()
userIdIdx = -6
phoneIdIdx = -5
spaceIdIdx = -8
buildingIdIdx = -9
floorIdx = -10
numData = 19937
uqid = -1
timestamp= -4

################################

#
#tim= df['Time']
#phon = df['PHONEID']
##colors = ['#c2c2f0','#ff9999','#99ff99', '#66b3ff','#ffb3e6']
#plt.scatter(tim, phon, c='#ff9999', alpha=0.9)
#plt.title('June 20: Time vs Phone ID')
#plt.ylabel('PhoneId')
##plt.xlabel('Time')
#
#
#
tim= df['Time']
phon = df['uqid']
#colors = ['#c2c2f0','#ff9999','#99ff99', '#66b3ff','#ffb3e6']
plt.scatter(tim, phon, c='#66b3ff', alpha=0.9)
plt.title('June 20: Time vs Space ID')
plt.ylabel('SpaceId')
#plt.xlabel('Time')


#############################################################################
#N = len(set(data[: , uqid]))
##rang = [np.min(uqid), np.max(uqid)] 
##bins = np.arange(N)+1 - 0.5
##plt.hist(data[: , phoneIdIdx] , range(N), color = 'mediumorchid',edgecolor='red')
#plt.hist(data[: , uqid] ,727,range=[np.min(df.uqid), np.max(df.uqid)] , color = 'steelblue', edgecolor='darkblue')
##plt.xticks(range(N),rotation= 90, size =8 )
##plt.title('Histogram of phone IDs')
##plt.ylabel('Data Distribution')
##plt.xlabel('PHONEID')
##
#
#plt.figure(2)
#N = len(set(data[: , spaceIdIdx]))
##bins = np.arange(20)+1 - 0.5
##plt.hist(data[: , phoneIdIdx] , range(N), color = 'mediumorchid',edgecolor='red')
#plt.hist(data[: , spaceIdIdx] , range(N) , color = 'plum', edgecolor='orchid')
#plt.xticks(range(N+1))
#plt.title('Histogram of phone IDs')
#plt.ylabel('Data Distribution')
#plt.xlabel('SPACEID')




################# ADDIBG UNIQ SPACEID ########################################

#
#
#
#df['BUILDINGID']=df['BUILDINGID'].replace(0,3)
#df['BUILDINGID']=df['BUILDINGID'].replace(1,4)
#df['BUILDINGID']=df['BUILDINGID'].replace(2,5)
#
#
#
#
#df['BUILDINGID']=df['BUILDINGID'].replace(3,1)
#df['BUILDINGID']=df['BUILDINGID'].replace(4,2)
#df['BUILDINGID']=df['BUILDINGID'].replace(5,3)
#
#bi = df['BUILDINGID']
#fl = df['FLOOR']
#rp = df['RELATIVEPOSITION']
#si = df['SPACEID']
#
##bi = bi*10000
##fl= fl*1000
##rp = rp *100
##uqid = bi+fl+rp+si
#
#
#
######@@@@@@@@@@@ remove relative position from UNIQID
#bi = bi*10000
#fl= fl*1000
#uqid = bi+fl+si
#
######@@@@@@@@@@@@@
#
#df['uqid'] = uqid
#
#######@@@@@@@@@@@@
#
#df.to_csv(r'TrainingDataWithUniQID.csv', index = False)
#
##df = df.loc[(df['BUILDINGID'] == 1) & (df['FLOOR'] == 2)] 
#
#
#plt.scatter(df.PHONEID,df.uqid)
#colormap = df.uqid-12000
#plt.figure(3)
#plt.scatter(df.LATITUDE, df.LONGITUDE, c = colormap)
#
#plt.colorbar()
##plt.clim(200, 250)
#
#plt.figure(2)
#plt.plot(df.SPACEID, '*')
#
#
#unq = df.uqid
#
##colormap = df.uqid-12000
#plt.figure(4)
#plt.scatter(unq, df.PHONEID)

########### Histogram of Uniqid

#plt.hist(df.uqid)

#N = len(set(data[: , phoneIdIdx]))
#bins = np.arange(20)+1 - 0.5
#plt.hist(data[: , phoneIdIdx] ,bins =bins , color = 'plum', edgecolor='orchid')
##plt.xticks(range(N+1))
#plt.title('Histogram of phone IDs')
#plt.ylabel('Data Distribution')
#plt.xlabel('PHONEID')




############################################ HISTOGRAMS #######################

#data = df.values
#df.describe()
#userIdIdx = -3
#phoneIdIdx = -2
#spaceIdIdx = -6
#buildingIdIdx = -7
#floorIdx = -8
#numData = 19937






#plt.hist(df.BUILDINGID,df.PHONEID)
#plt.show()
######userID + All data point

#y = -1
#N = len(set(data[: , y]))
#plt.hist(data[: , y] , range(N), color = 'royalblue',edgecolor='red')
#plt.xticks(range(N))
#plt.title('Histogram of user ID data provider')
##plt.ylabel('Number of Data points')
#plt.xlabel('UQID')
#


######@@@@@@@@@SPACEID + All data point

#N = len(set(data[: , spaceIdIdx]))
##plt.hist(data[: , spaceIdIdx] , range(N), color = 'deepskyblue',edgecolor='red')
#plt.hist(data[: , spaceIdIdx] , bins = N, color ='steelblue', edgecolor='darkblue')
#
##plt.xticks(range(N))
#plt.title('Histogram of all space IDs')
#plt.ylabel('Data Distribution')
#plt.xlabel('SPACEID')


## Surface plot lat lon and number unqid
#lt = np.reshape(df.LATITUDE,())


#
########PHONEID + All data point

#
#N = len(set(data[: , phoneIdIdx]))
#bins = np.arange(17)+1 - 0.5
##plt.hist(data[: , phoneIdIdx] , range(N), color = 'mediumorchid',edgecolor='red')
#plt.hist(data[: , phoneIdIdx] , bins=bins , color = 'lightpink', edgecolor='hotpink')
#plt.xticks(range(N+1))
#plt.title('Histogram of phone IDs')
#plt.ylabel('Data Distribution')
#plt.xlabel('PHONEID')


############ USER ID
#
#N = len(set(data[: , userIdIdx]))
#bins = np.arange(20)+1 - 0.5
##plt.hist(data[: , phoneIdIdx] , range(N), color = 'mediumorchid',edgecolor='red')
#plt.hist(data[: , userIdIdx] , bins=bins , color = 'plum', edgecolor='orchid')
#plt.xticks(range(N+1))
#plt.title('Histogram of user IDs')
#plt.ylabel('Data Distribution')
#plt.xlabel('USERID')



##################### Space id each building

#
#df = df.loc[(df['FLOOR'] == 0) ] 


#data = df.values
#df.describe()
#userIdIdx = -3
#phoneIdIdx = -2
#spaceIdIdx = -5
#buildingIdIdx = -6
#floorIdx = -7
#numData = 19937

#
#

#
#N = len(set(data[: , spaceIdIdx]))
##bins = np.arange(N)+1 - 0.5
##plt.hist(data[: , phoneIdIdx] , range(N), color = 'mediumorchid',edgecolor='red')
#plt.hist(data[: , spaceIdIdx] , range(N) , color = 'steelblue', edgecolor='darkblue')
#plt.xticks(range(N),rotation= 90, size =8 )
#plt.title('Histogram of phone IDs')
#plt.ylabel('Data Distribution')
#plt.xlabel('PHONEID')
#





##################
#N = len(set(data[: , buildingIdIdx]))
#plt.hist(data[: , buildingIdIdx] , range(N), color = 'coral',edgecolor='red')
#plt.xticks(range(N))
#plt.title('Histogram of building id')





########################################### Pie chart
#N = len(set(data[: , floorIdx]))
#H = np.histogram(data[: , floorIdx] , range(N))
#plt.pie(H[1],labels=range(N),shadow=True)
#plt.title('Pie-chart of all floors ')
#

######  1
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#ax.axis('equal')
#langs = ['Floor 1', 'Floor 2', 'Floor 3', 'Floor 4', 'Floor 5']
#ax.pie(H[1], labels = langs,autopct='%1.2f%%')
#ax.legend()
#fig.suptitle('Pie-chart of all floors')

### 2
#colors = ['#c2c2f0','#ff9999','#99ff99', '#66b3ff','#ffb3e6']
#langs = ['Floor 1', 'Floor 2', 'Floor 3', 'Floor 4', 'Floor 5']
#df.groupby(['FLOOR']).sum().plot(kind='pie', y=N,startangle=90,labels = langs,
#          colors = colors,figsize=(10,8), autopct='%1.1f%%')

##################    Time series Alalysis    #################################
#
#fig = plt.figure(2)
#winSize = 1000
#feature = userIdIdx
#N = set(data[: , feature])
#N = list(N)
#D = np.zeros((len(N) , numData - winSize))
#for i in range(numData - winSize):
#  for j in range(len(N)):
#    D[j,i] = np.sum(data[i : i + winSize , feature] == N[j])
#
#for j in range(len(N)):
#  plt.plot(D[j,:])
#plt.title('UserID Activity')


#
#
##############################
###################    SP---- Time series Alalysis    #################################
##
##fig = plt.figure(2)
#winSize = 1000
#feature = userIdIdx
#N = set(data[: , feature])
#N = list(N)
#D = np.zeros((len(N) , numData - winSize))
#wdr = np.array(1:10:20)
#for i in range(numData - winSize):
#  for j in range(len(N)):
#    D[j,i] = np.sum(data[i : i + winSize , feature] == N[j])
#
#for j in range(len(N)):
#  plt.plot(D[j,:])
#plt.title('UserID Activity')
#


###############################################
#fig = plt.figure(6)
#winSize = 19000
#feature = spaceIdIdx
#N = set(data[: , feature])
#N = list(N)
#D = np.zeros((len(N) , numData - winSize))
#for i in range(numData - winSize):
#  for j in range(len(N)):
#    D[j,i] = np.sum(data[i : i + winSize , feature] == N[j])
#
#for j in range(len(N)):
#  plt.plot(D[j,:])
#  
#plt.title('SpaceID activity During the Experiment')
##leg = plt.legend(framealpha=1, frameon=True)
#plt.ylabel('SPACEID')
#plt.xlabel('Data points')


#fig = plt.figure(3)
#winSize = 1000
#feature = phoneIdIdx
#N = set(data[: , feature])
#N = list(N)
#D = np.zeros((len(N) , numData - winSize))
#for i in range(numData - winSize):
#  for j in range(len(N)):
#    D[j,i] = np.sum(data[i : i + winSize , feature] == N[j])
#
#for j in range(len(N)):
#  plt.plot(D[j,:])
#plt.title('PhoneID activity During the Experiment')
##leg = plt.legend(framealpha=1, frameon=True)
##plt.ylabel('PHONEID')
##plt.xlabel('Data points')#
#
#
#
#fig = plt.figure(4)
#winSize = 1000
#feature = floorIdx
#N = set(data[: , feature])
#N = list(N)
#D = np.zeros((len(N) , numData - winSize))
#for i in range(numData - winSize):
#  for j in range(len(N)):
#    D[j,i] = np.sum(data[i : i + winSize , feature] == N[j])
#
#for j in range(len(N)):
#  plt.plot(D[j,:])
#plt.title('Floor activity During the Experiment')
##leg = plt.legend(framealpha=1, frameon=True)
##plt.ylabel('PHONEID')
##plt.xlabel('Data points')#
##







#fig = plt.figure(5)
#winSize = 1000
#feature = buildingIdIdx
#N = set(data[: , feature])
#N = list(N)
#D = np.zeros((len(N) , numData - winSize))
#for i in range(numData - winSize):
#  for j in range(len(N)):
#    D[j,i] = np.sum(data[i : i + winSize , feature] == N[j])
#
#for j in range(len(N)):
#  plt.plot(D[j,:])
#plt.title('Building activity During the Experiment')
#leg = plt.legend(framealpha=1, frameon=True)
#plt.ylabel('PHONEID')
#plt.xlabel('Data points')#
#

###############################################################################



 ############################################Tracking user
#print('select one from:')
#print(set(data[:,phoneIdIdx]))
#PhoneId = 8
#winSize = 1000
#feature = spaceIdIdx
#
#N = set(data[: , feature])
#N = list(N)
#D = np.zeros((len(N) , numData - winSize))
#for i in range(numData - winSize):
#  for j in range(len(N)):
#    temp = data[i : i + winSize , feature]
#    cond = data[i : i + winSize , phoneIdIdx]
#    D[j,i] = np.sum(temp[cond == PhoneId] == N[j])
#
#for j in range(len(N)):
#  plt.plot(D[j,:])
#plt.title('Space id activity for PhoneId = '+str(PhoneId))







#############################################################################


#PhoneIdAll = set(data[:,phoneIdIdx])
#PhoneIdAll = list(PhoneIdAll)
#winSize = 1000
#feature = uqid
#
#N = set(data[: , feature])
#N = list(N)
#
#for p in range(len(PhoneIdAll)):
#  PhoneId = PhoneIdAll[p]
#  D = np.zeros((len(N) , numData - winSize))
#  for i in range(numData - winSize):
#    for j in range(len(N)):
#      temp = data[i : i + winSize , feature]
#      cond = data[i : i + winSize , phoneIdIdx]
#      D[j,i] = np.sum(temp[cond == PhoneId] == N[j])
#
#  plt.subplot(4, 4, p+1)
#  for j in range(len(N)):
#    plt.plot(D[j,:])
#  plt.title('Space id activity for PhoneId = '+str(int(PhoneId)))




















