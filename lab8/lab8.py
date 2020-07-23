#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:39:00 2019

@author: nitesh
"""

import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np
import statistics as st

def read_data(file_path):
    df=pd.read_csv(file_path)
    return df

def train_testsplit(df):
    df2=df.copy()
    X=np.array(df2.drop(['quality'],1))
    Y=np.array(df2["quality"])
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
    return X_train,X_test,Y_train,Y_test

def linear(x1,x2,y1,y2):
    regressor=LinearRegression()
    regressor.fit(x1,y1)
    y_test_pred=regressor.predict(x2)
    y_train_pred=regressor.predict(x1)
    if(x1.shape[1]==1):
        plt.plot(x1,y_train_pred)
        plt.show()
    elif(x1.shape[1]==2):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(x1[:,0],x1[:,1],y_train_pred)
        plt.show()
    print("Prediction accuracy on training data is ",mse(y1,y_train_pred)**0.5)
    print("Prediction accuracy on test data is ",mse(y2,y_test_pred)**.5)
    plt.title("Actual quality vs predicted quality")
    plt.scatter(y2,y_test_pred)
    plt.show()

def nonlinear(x1,x2,y1,y2):
    plist=[2,3,4,5]
    train_rmse=[]
    test_rmse=[]
    min_error=10**6
    min_val=0
    for p in plist: 
        print("For the value of degree = ",p)
        model=PolynomialFeatures(degree=p)
        x_poly=model.fit_transform(x1)
        regressor = LinearRegression()
        regressor.fit(x_poly,y1)
        y_test_pred=regressor.predict(model.fit_transform(x2))
        y_train_pred=regressor.predict(model.fit_transform(x1))
        if(x1.shape[1]==1):
            plt.scatter(x1,y_train_pred)
            plt.show()
        elif(x1.shape[1]==2):
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_trisurf(x1[:,0],x1[:,1],y_train_pred)
            plt.show()
        rmsevalue_test=mse(y2,y_test_pred)**.5
        train_rmse.append(mse(y1,y_train_pred)**0.5)
        test_rmse.append(rmsevalue_test)
        print("Prediction accuracy on training data is ",mse(y1,y_train_pred)**0.5)
        print("Prediction accuracy on test data is ",mse(y2,y_test_pred)**.5)
        if(min_error>rmsevalue_test):
            min_error=rmsevalue_test
            y_best=y2
            y_best_pred=y_test_pred
            min_val=p
    bars=plt.bar(plist,train_rmse)
    plt.title("Training Data")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, round(yval,5))
    plt.show()
    bars=plt.bar(plist,test_rmse)
    plt.title("Testing Data")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, round(yval,5))
    plt.show()
    print("Best fit is for value = ",min_val )
    plt.scatter(y_best,y_best_pred)
    plt.show()
    
    
    

def main():
    df=read_data("lab8winequality-red.csv")
    x1,x2,y1,y2=train_testsplit(df)
    x1_pH=x1[:,8].reshape(-1,1)
    x2_pH=x2[:,8].reshape(-1,1)
    print("\n Simple Linear \n")
    linear(x1_pH,x2_pH,y1,y2)
    print("\n Simple nonLinear \n")
    nonlinear(x1_pH,x2_pH,y1,y2)
    print("\n Multi Linear \n")
    linear(x1,x2,y1,y2)
    print("\n multi nonLinear \n")
    nonlinear(x1,x2,y1,y2)
    print(df[df.columns[0:]].corr()['quality'][:])
    new_x1=np.vstack((x1[:,1],x1[:,10])).T
    new_x2=np.vstack((x2[:,1],x2[:,10])).T
    print("\n multi Linear using columns alcohol and voltalie acidity\n")
    linear(new_x1,new_x2,y1,y2)
    print("\n multi nonLinear using columns alcohol and voltalie acidity\n")
    nonlinear(new_x1,new_x2,y1,y2)
    
    
if(__name__=="__main__"):
    main()


