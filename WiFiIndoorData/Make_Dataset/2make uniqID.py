# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:02:30 2021

@author: neshragh



Add New Colomn as UNIQ ID --> make spaceid uniq in each building and floor
"""

import numpy as np
import pandas as pd
import datetime as DT
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder



#df = pd.read_csv('TrainingData_order.csv')
df = pd.read_csv('TrainingDataNoGap.csv')

#df = df.loc[(df['FLOOR'] == 0) ] 


#data = df.values
#df.describe()
#userIdIdx = -3
#phoneIdIdx = -2
#spaceIdIdx = -5
#buildingIdIdx = -6
#floorIdx = -7
#numData = 19937
##
##
##
#
################# ADDIBG UNIQ SPACEID ########################################




df['BUILDINGID']=df['BUILDINGID'].replace(0,3)
df['BUILDINGID']=df['BUILDINGID'].replace(1,4)
df['BUILDINGID']=df['BUILDINGID'].replace(2,5)




df['BUILDINGID']=df['BUILDINGID'].replace(3,1)
df['BUILDINGID']=df['BUILDINGID'].replace(4,2)
df['BUILDINGID']=df['BUILDINGID'].replace(5,3)

bi = df['BUILDINGID']
fl = df['FLOOR']
rp = df['RELATIVEPOSITION']
si = df['SPACEID']

#bi = bi*10000
#fl= fl*1000
#rp = rp *100
#uqid = bi+fl+rp+si



#####@@@@@@@@@@@ remove relative position from UNIQID
bi = bi*10000
fl= fl*1000
uqid = bi+fl+si
#####@@@@@@@@@@@@@

df['uqid'] = uqid

######@@@@@@@@@@@@
#######

label_encoder = LabelEncoder()
df['uqid2'] = label_encoder.fit_transform(df['uqid'])



df.to_csv(r'TrainingDataWithUniQID2.csv', index = False)

#df = df.loc[(df['BUILDINGID'] == 1) & (df['FLOOR'] == 2)] 

plt.figure(6)
plt.scatter(df.PHONEID,df.uqid)
colormap = df.uqid-12000
plt.figure(3)
plt.scatter(df.LATITUDE, df.LONGITUDE, c = colormap)

plt.colorbar()
#plt.clim(200, 250)

plt.figure(2)
plt.plot(df.SPACEID, '*')


unq = df.uqid

#colormap = df.uqid-12000
plt.figure(4)
plt.scatter(unq, df.PHONEID)


plt.figure(7)
plt.scatter(df.uqid2, df.PHONEID)
