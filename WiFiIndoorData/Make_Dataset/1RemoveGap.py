# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:17:28 2020
Convert Linux time to datetime
@author: neshragh
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.mplot3d import Axes3D

#df = pd.read_csv('hi.csv')
#
#df.sample(frac=1)
#
#df.to_csv(r'sam.csv', index = False)




#import random
#index = [i for i in range(df.shape[0])]
#random.shuffle(index)
#df.set_index([index]).sort_index()
#df.to_csv(r'mmmm.csv', index = False)

##############################################################@@@@@@@@@@@@@@
#df = pd.read_csv('TrainingData.csv')
df = pd.read_csv('C:/Users/neshragh/ecounter/Affinity_Sample_SPY/2-Dataset_Wifi/archive/original/TrainingData.csv')

df["PHONEID"].replace({1: 1, 3: 2,
   6: 3, 7: 4,
   8: 5, 10: 6,
11: 7, 13: 8,
   14: 9, 16: 10,
   17: 11, 18: 12,
   19: 13, 22: 14,
   23: 15, 24: 16, }, inplace=True)

#
df["SPACEID"].replace({
   18: 18, 22: 19,
   25: 20, 26: 21,
   27: 22, 28: 23,
   29:24 , 30:25,
   101:26 , 102:27,
   103: 28, 104: 29,
  105: 30, 106: 31,
  107: 32, 108: 33,
  109: 34, 110: 35,
  111: 36, 112: 37,
  113: 38, 114: 39,
  115: 40, 116: 41,
  117: 42, 118: 43,
  119: 44, 120: 45,
  121: 46, 122: 47,
  123: 48, 124: 49,
  125: 50, 126: 51,
  127: 52, 128: 53,
  129: 54, 130: 55,
  131: 56, 132: 57,
  133: 58, 134: 59,
  135: 60, 136: 61,
  137: 62, 138: 63,
  139: 64, 140: 65,
  141: 66, 142: 67,
  143: 68, 144: 69,
  146: 70, 147: 71,
  201: 72, 202: 73,
  203: 74, 204: 75,
  205: 76, 206: 77,
  207: 78, 208: 79,
  209: 80, 210: 81,
  211: 82, 212: 83,
  213: 84, 214: 85,
  215: 86, 216: 87,
  217: 88, 218: 89,
  219: 90, 220: 91,
  221: 92, 222: 93,
  223: 94, 224: 95,
  225: 96, 226: 97,
  227: 98, 228: 99,
  229: 100, 230: 101,
  231: 102, 232: 103,
  233: 104, 234: 105,
  235: 106, 236: 107,
  237: 108, 238: 109,
  
  239: 109, 240: 110,
  241: 111, 242: 112,
  243: 113, 244: 114,
  245: 115, 246: 116,
  247: 117, 248: 118,
  249: 119, 250: 120,
  253: 121, 254: 122,

  }, inplace=True)

#df.sort_values(by=['SPACEID'], inplace=True)
df.to_csv(r'TrainingDataNoGap.csv', index = False)
#
#print(df)




#############################################################################
#making dataset ,change time and save it

#
#timestamp = datetime.datetime.fromtimestamp(1371714660)
#print(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
#
#df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
#df.sort_values(by=['TIMESTAMP'], inplace=True)
#
#
#print(df)
#
#
#df.to_csv(r'hi.csv', index = False)
#
#
#
#
#
#timestamp = datetime.datetime.fromtimestamp(1369908924)
#print(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
#
#
#
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax = Axes3D(fig)
#    
#ax.view_init(elev=20., azim=45)
#    
#
#ax.scatter(df[0], df[1], df[4], marker="o", picker=True)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Floor')

