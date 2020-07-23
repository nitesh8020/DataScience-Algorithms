#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:27:47 2019

@author: nitesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.ar_model import AR
def load_dataset(a):
    file=pd.read_csv(a)
    return file
a=load_dataset('daily-min-temperatures.csv')
temp=a['Temp']
date=a['Date']
a.plot()
plt.show()
new=[]
new1=[]
for i in range(len(temp)-1):
    new1.append(temp[i])
    new.append(temp[i+1])
corr=np.corrcoef(new,new1)[0][1]
print(corr)
sm.graphics.tsa.plot_acf(temp,lags=20)
plt.show()
train=a[:3643]['Temp']
test=a.tail(7)['Temp']
a3=a[3642:3649]
print(mse(test,a3['Temp'])**(1/2))
model = AR(train)
model_fit = model.fit()
print('Lag:',model_fit.k_ar)
print('Coefficients:', model_fit.params)
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i+3643], test[i+3643]))
error = mse(test, predictions)
print(error**(1/2))
plt.plot(test)
plt.plot(predictions)
plt.show()