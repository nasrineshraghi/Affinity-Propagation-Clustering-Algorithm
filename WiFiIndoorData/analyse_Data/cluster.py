# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:58:21 2020

@author: neshragh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime as DT
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from PIL import Image, ImageDraw, ImageFont



data = pd.read_csv('TrainingData.csv').values

#numData = 19937
#numFeature = 520
#X = data[:numData , 0 : numFeature]
#
#lat = data[:numData,520]
#lan = data[:numData,521]
#floor = data[:numData,522]
#set(floor)
#
#num_clusters = 4
#kmeans = KMeans(n_clusters = num_clusters).fit(X)
#labels = kmeans.labels_
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(lat, lan, floor, c=labels, marker="o", picker=True)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Floor')
#
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#ax.scatter(lat, lan, c=labels, marker="o", picker=True)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax = Axes3D(fig)
#ax.view_init(elev=20., azim=45)
#ax.scatter(lat, lan, floor, c=labels, marker="o", picker=True)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Floor')
##################################################################

numData = 1000
numFeature = 500

N = np.size(data,0)
idx = np.random.randint(0 , N , numData)
X = data[idx,0:numFeature]

lat = data[idx,520]
lan = data[idx,521]
floor = data[idx,522]

ap = AffinityPropagation(damping=0.75).fit(X)
labels = ap.labels_
num_clusters = len(set(labels))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lat, lan, floor, c=labels, marker="o", picker=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Floor')

