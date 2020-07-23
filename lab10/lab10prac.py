import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as km
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as gmm
from sklearn import metrics
from sklearn.metrics import homogeneity_score as hs

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0))/np.sum(contingency_matrix)

f = open("2D_points2.txt",'r')
a = f.read()
f.close()
a = a.split()
b1 = []
b2 = []
for i in range(len(a)):
    if i%2==0:
        b1.append(a[i])
    else:
        b2.append(a[i])
a1 = [float(x) for x in b1]
a2 = [float(x) for x in b2]

df = pd.DataFrame()
df['x'] = a1
df['y'] = a2

c = [1,2,3,4,5,6,7,8,9,10]
se = []
for i in c:
    model = km(n_clusters=i)
    y = model.fit_predict(df)
    se.append(model.inertia_)
plt.title("KMeans")
plt.plot(c,se)
plt.show()

ll = []
for i in c:
    mo = gmm(n_components=i)
    y1 = mo.fit_predict(df)
    ll.append(mo.score(df))
plt.title("GMM")
plt.plot(c,ll)
plt.show()

model = km(n_clusters=4)
y = model.fit_predict(df)

print(model.inertia_)
print("Predicted: ",y)
plt.title("KMeans")
plt.scatter(df[y==0]['x'],df[y==0]['y'])
plt.scatter(df[y==1]['x'],df[y==1]['y'])
plt.scatter(df[y==2]['x'],df[y==2]['y'])
plt.scatter(df[y==3]['x'],df[y==3]['y'])
plt.show()
y_true = [0 for i in range(500)]
for i in range(500):
    y_true.append(1)
for i in range(500):
    y_true.append(2)
for i in range(500):
    y_true.append(3)
print("Purity score for KMeans: ",purity_score(y_true,y))
model1 = gmm(n_components=4)
y1 = model1.fit_predict(df)
print("predicted: ",y1)
plt.title("GMM")
plt.scatter(df[y1==0]['x'],df[y1==0]['y'])
plt.scatter(df[y1==1]['x'],df[y1==1]['y'])
plt.scatter(df[y1==2]['x'],df[y1==2]['y'])
plt.scatter(df[y1==3]['x'],df[y1==3]['y'])
plt.show()

print("Purity score for gmm: ",purity_score(y_true,y1))
print("hs for KMeans ",hs(y_true,y))
print("hs for GMM ",hs(y_true,y1))

