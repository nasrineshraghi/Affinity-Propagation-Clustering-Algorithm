# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:07:00 2021

@author: neshragh
"""
import numpy as np
import pandas as pd
#from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler


#df = pd.read_csv('C:/Users/neshragh/ecounter/Affinity_Sample_SPY/2-Dataset_Wifi/archive/TrainingDataWithUniQIdAllBuilding.csv')

df = pd.read_csv('C:/Users/neshragh/OneDrive - University of New Brunswick/UNB_thesis_Work/Affinity_Sample_SPY/2-Dataset_Wifi/indoor localization dataset/Make_Dataset/TrainingDataWithUniQID2.csv')
##
#x = df.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#df = pd.DataFrame(x_scaled)
#
#
#Col1 is the last element of the dataset = uniqid
Col1 = -1
Col2 = -6
timeColNum = 2

T = df.values
a = T[:,Col1].reshape((-1,1))
b = T[:,Col2].reshape((-1,1))



#max_a = np.max(a)
#max_b = np.max(b)
#min_a = np.min(a)
#min_b = np.min(b)
#a = (a-min_a) / (max_a-min_a)
#b = (b - min_b) / (max_b-min_b)

### Level max scaler: scales all data between 1-720 ##########################################

max_a = np.max(a)
max_b = np.max(b)
#a = (a*max_b)/(max_a)
b = (b*max_a)/(max_b)
X = np.array([[a,b]]) 
p=X.reshape(2,len(a))
X=p.T





df['uqidNorm'] = a
df['PHONEIDNorm'] = b





  
df.to_csv(r'NormalizeAll.csv',index = False, header = True)
