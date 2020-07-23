#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:16:46 2019

@author: nitesh
"""

import pandas as pd
from sklearn.cluster import KMeans as km
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
def purity_score(y_true, y_pred):
     contingency_matrix = metrics.cluster.contingency_matrix(y_true,
y_pred)
     return np.sum(np.amax(contingency_matrix,
axis=0))/np.sum(contingency_matrix)

f = open("2D_points.txt",'r')
a = f.read()
f.close()
a=a.split()
b0=[]
b1=[]
for i in range(len(a)):
    if i%2==0:
        b0.append(a[i])
    else:
        b1.append(a[i])
a0 = [float(x) for x in b0]
a1 = [float(x) for x in b1]
df = pd.DataFrame()
df['x'] = a0
df['y'] = a1
y_true = [0 for i in range(500)]
for i in range(len(a0)-500):
     y_true.append(1)
model = km(n_clusters=3)
y = model.fit_predict(df)
plt.title("KMeans")
plt.scatter(df[y==0]['x'],df[y==0]['y'])
plt.scatter(df[y==1]['x'],df[y==1]['y'])
plt.show()
print("Purity score for KMeans: ",purity_score(y_true,y))

clustering = AgglomerativeClustering().fit(df)
y=clustering.labels_
plt.title("Agglo_Clustering")
plt.scatter(df[y==0]['x'],df[y==0]['y'])
plt.scatter(df[y==1]['x'],df[y==1]['y'])
plt.show()
print("Purity score for Agglomerative Clustering: ",purity_score(y_true,y))

clustering = DBSCAN().fit(df)
y=clustering.labels_
plt.title("DBSCAN")
plt.scatter(df[y==0]['x'],df[y==0]['y'])
plt.scatter(df[y==1]['x'],df[y==1]['y'])
plt.show()
print("Purity score for DBSCAN: ",purity_score(y_true,y))
l1=[0.05,0.50,0.95]
l2=[1,10,30,50]

for i in range(3):
    clustering = DBSCAN(eps=l1[i]).fit(df)
    y=clustering.labels_
    print("Purity score for DBSCAN with eps=",l1[i],": ",purity_score(y_true,y))

for i in range(4):
    clustering = DBSCAN(min_samples=l2[i]).fit(df)
    y=clustering.labels_
    print("Purity score for DBSCAN with min samples=",l2[i],": ",purity_score(y_true,y))
 
f = open("2D_pointsR.txt",'r')
a = f.read()
f.close()
a=a.split()
b0=[]
b1=[]
for i in range(len(a)):
    if i%2==0:
        b0.append(a[i])
    else:
        b1.append(a[i])
a0 = [float(x) for x in b0]
a1 = [float(x) for x in b1]
df = pd.DataFrame()
df['x'] = a0
df['y'] = a1
y_true = [0 for i in range(150)]
for i in range(len(a0)-150):
     y_true.append(1)
model = km(n_clusters=3)
y = model.fit_predict(df)
plt.title("KMeans")
plt.scatter(df[y==0]['x'],df[y==0]['y'])
plt.scatter(df[y==1]['x'],df[y==1]['y'])
plt.show()
print("Purity score for KMeans: ",purity_score(y_true,y))

clustering = AgglomerativeClustering().fit(df)
y=clustering.labels_
plt.title("Agglo_Clustering")
plt.scatter(df[y==0]['x'],df[y==0]['y'])
plt.scatter(df[y==1]['x'],df[y==1]['y'])
plt.show()
print("Purity score for Agglomerative Clustering: ",purity_score(y_true,y))

clustering = DBSCAN().fit(df)
y=clustering.labels_
plt.title("DBSCAN")
plt.scatter(df[y==0]['x'],df[y==0]['y'])
plt.scatter(df[y==1]['x'],df[y==1]['y'])
plt.show()
print("Purity score for DBSCAN: ",purity_score(y_true,y))
l1=[0.05,0.50,0.95]
l2=[1,10,30,50]

for i in range(3):
    clustering = DBSCAN(eps=l1[i]).fit(df)
    y=clustering.labels_
    print("Purity score for DBSCAN with eps=",l1[i],": ",purity_score(y_true,y))

for i in range(4):
    clustering = DBSCAN(min_samples=l2[i]).fit(df)
    y=clustering.labels_
    print("Purity score for DBSCAN with min samples=",l2[i],": ",purity_score(y_true,y))