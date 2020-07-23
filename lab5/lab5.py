import pandas as pd
import statistics as st
l=["pregs","plas","pres","skin","test","BMI","pedi","Age","class"]
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    return pd.read_csv(path)

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

def train_testsplit(df):
    df2=df.copy()
    X=np.array(df2.drop(['class'],1))
    Y=np.array(df2["class"])
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
    return X_train,X_test,Y_train,Y_test

def confusion(ytrue,ypred):
    print(confusion_matrix(ytrue,ypred))
    return 0
    
def classifier(xtrain,xtest,ytrain,ytest,k):
    clf=KNeighborsClassifier(n_neighbors=k)
    clf.fit(xtrain,ytrain)
    accuracy=clf.score(xtest,ytest)
    z=clf.predict(xtest)
    print(accuracy,accuracy_score(z,ytest))
    print("Value of k is",k)
    confusion(ytest,z)
    return(accuracy)

def main():
    path="pima-indians-diabetes.csv"
    df=load_data(path)
    df1=min_max_normalization(df,[0,1])
    df2=standardize(df)
    k=[1,3,5,7,9,11,13,15,17,21]
    x1,x2,y1,y2=train_testsplit(df)
    a=[]
    for i in range(len(k)):
        y=classifier(x1,x2,y1,y2,k[i])
        a.append(y)
    plt.scatter(k,a)
    plt.show()
    df.to_csv("1.csv")
    
    
    x1,x2,y1,y2=train_testsplit(df1)
    a=[]
    for i in range(len(k)):
        y=classifier(x1,x2,y1,y2,k[i])
        a.append(y)
    plt.scatter(k,a)
    plt.show()
    
    x1,x2,y1,y2=train_testsplit(df2)
    a=[]
    for i in range(len(k)):
        y=classifier(x1,x2,y1,y2,k[i])
        a.append(y)
    plt.scatter(k,a)
    plt.show()

if(__name__=="__main__"):
    main()
