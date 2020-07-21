import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


d1=pd.read_csv("winequality-red_miss.csv")
d2=pd.read_csv("winequality-red_original.csv",sep=";")

d=d1.isna().sum(axis=1)
print(d.sum())
print(len(d))
t=[x for x in range(0,13)]
y=[0 for x in range(0,13)]
for i in range(len(d)):
    y[d[i]]+=1
print(y)    
plt.plot(t[1:],y[1:])
plt.show()
c=0
for i in range(len(y)):
    if(i>=6):
        c+=y[i]
print("number of tuples with more than 50% values missing is ",c)
c=[]
for i in range(len(d)):
    if(d[i]>=6):
        c.append(i)
d1=d1.drop(c)
d=d1.isna().sum(axis=1)
print(len(d))
print(c)
e=[]

d4=d1.isna()
q=d4["quality"]
print(q[12])
for i in range(len(q)):
    if(i not in c):
        if(q[i]):
            e.append(i)
d1=d1.drop(e)
d=d1.isna().sum(axis=1)
print(len(d))
print(e)
med=d1.median()
d3=d1.fillna(med)
l=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
for i in range(len(l)):
    m=d3[l[i]]
    o=d2[l[i]]
    print("for attribute",l[i])
    omn=st.mean(o)
    omd=st.median(o)
    ost=st.stdev(o)
    nmn=st.mean(m)
    nmd=st.median(m)
    nst=st.stdev(m)
    print("original mean is ",omn)
    print("new mean is ",nmn)
    print("original median is ",omd)
    print("new median is ",nmd)
    print("original standard deviation is",ost)
    print("new standard deviation is",nst)
    rms = rmse(o,m)
    print("rmse error for mean is ",rms)
    print()


d4=d1.fillna(method="ffill")

print("second method")
for i in range(len(l)):
    m=d4[l[i]]
    o=d2[l[i]]
    print("for attribute",l[i])
    omn=st.mean(o)
    omd=st.median(o)
    ost=st.stdev(o)
    nmn=st.mean(m)
    nmd=st.median(m)
    nst=st.stdev(m)
    print("original mean is ",omn)
    print("new mean is ",nmn)
    print("original median is ",omd)
    print("new median is ",nmd)
    print("original standard deviation is",ost)
    print("new standard deviation is",nst)
    rms = rmse(o,m)
    print("rmse error for mean is ",rms)
    print()
    
d5=d1.interpolate(method='linear', limit_direction='forward', axis=0)
print("third method")
for i in range(len(l)):
    m=d5[l[i]]
    o=d2[l[i]]
    print("for attribute",l[i])
    omn=st.mean(o)
    omd=st.median(o)
    ost=st.stdev(o)
    nmn=st.mean(m)
    nmd=st.median(m)
    nst=st.stdev(m)
    print("original mean is ",omn)
    print("new mean is ",nmn)
    print("original median is ",omd)
    print("new median is ",nmd)
    print("original standard deviation is",ost)
    print("new standard deviation is",nst)
    rms = rmse(o,m)
    print("rmse error for mean is ",rms)
    print()

