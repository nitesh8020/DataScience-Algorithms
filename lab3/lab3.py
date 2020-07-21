import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
l=["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","quality"]
def read_data(path_to_file):
    return pd.read_csv(path_to_file)

def drawbox(col,df):
    l=df[col]
    plt.boxplot(l)
    plt.show()
def replace_outliers(df):
    q1=df.quantile(0.25)
    q2=df.quantile(0.75)
    for i in range(len(l)-1):
        lw=q1[i]
        h=q2[i]
        iqr=h-lw
        ls=df[l[i]]
        c=0
        md=st.median(ls)
        for j in range(len(ls)):
            if((ls[j]<lw-1.5*iqr)or(ls[j]>h+1.5*iqr)):
                c+=1
                df.loc[j,l[i]]=md
        print("the number of outliers in ",l[i]," is ",c )
    return df
            
def ranger(df,col):
    ls=df[col]
    return(min(ls),max(ls))

def min_max_normalization(df,rg):
    df2=df.copy()
    for i in range(len(l)-1):
        ls=df2[l[i]]
        for j in range(len(ls)):
            x=(ls[j]-min(ls))*(rg[1]-rg[0])/(max(ls)-min(ls)) +rg[0]
            df2.loc[j,l[i]]=x
    return df2
            
def standardize(df):
    df2=df.copy()
    for i in range(len(l)-1):
        ls=df2[l[i]]
        mn=st.mean(ls)
        sd=st.stdev(ls)
        for j in range(len(ls)):
            x=(ls[j]-mn)/sd
            df2.loc[j,l[i]]=x
    return df2
    
def main():
    pf="winequality_red_original.csv"
    df=read_data(pf)
    
    for i in range(len(l)-1):
       drawbox(l[i],df)
    df=replace_outliers(df)
    for i in range(len(l)-1):
        print("range of ",l[i]," is ",ranger(df,l[i]))
    df1=min_max_normalization(df,[0,1])
    for i in range(len(l)-1):
        print("range of ",l[i]," is ",ranger(df1,l[i]))
    df2=min_max_normalization(df,[0,20])
    for i in range(len(l)-1):
        print("range of ",l[i]," is ",ranger(df2,l[i]))
    df1=replace_outliers(df1)
    df3=standardize(df)
    df3=df=replace_outliers(df3)
    
    MinMaxScaler(copy=True,feature_range=(0,1))
    scaler=MinMaxScaler()    
    print(scaler.fit(df))
    df4=scaler.transform(df)
    scaler = StandardScaler()
    print(scaler.fit(df))
    df5=scaler.transform(df)
    print(st.mean(df5[0]))
if(__name__=="__main__"):
    main()
