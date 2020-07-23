
import pandas as pd
import numpy as np
import sklearn.preprocessing as pr
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB
l=["X_Min","X_Max","Y_Min","Y_Max","Pixels_Areas","X_Perimeter","Y_Perimeter","Sum_of_Luminosity","Minimum_of_Luminosity","Maximum_of_Luminosity","Length_of_Conveyer","TypeOfSteel_A300","TypeOfSteel_A400","Steel_Plate_Thickness","Edges_Index ","Empty_Index","Square_Index","Outside_X_Index " ,"Edges_X_Index ","Edges_Y_Index ","Outside_Global_Index " ,"LogOfAreas ","Log_X_Index ","Log_Y_Index ","Orientation_Index ","Luminosity_Index ","SigmoidOfAreas ","Z_Scratch"]
k=[1,3,5,7,9,11,13,15,17,21]

def read_data(file_path):
    df=pd.read_csv(file_path)
    return df

def center(df):
    df2=df.copy()
    for x in range(len(l)-1):
        df2[l[x]]=df2[l[x]]-df2[l[x]].mean()
    return df2

def train_testsplit(df,size):
    df2=df.copy()
    X=np.array(df2.drop(["Z_Scratch"],1))
    Y=np.array(df2["Z_Scratch"])
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=size,random_state=42)
    return X_train,X_test,Y_train,Y_test

def classifier(xtrain,xtest,ytrain,ytest,k):
    clf=KNeighborsClassifier(n_neighbors=k)
    clf.fit(xtrain,ytrain)
    accuracy=clf.score(xtest,ytest)
    z=clf.predict(xtest)
    print("Value of k is",k)
    confusion(ytest,z)
    print("The value of accuracy is",accuracy)
    return(accuracy)

def confusion(ytrue,ypred):
    print(confusion_matrix(ytrue,ypred))
    return 0



def dim_reduction(df):
    df2=df.copy()
    df2=center(df2)
    vals,vecs=np.linalg.eig(df2.T.dot(df2))
    idx =vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:,idx]
    lt=[i for i in range(27)]
    clas=df2["Z_Scratch"]
    df3=pd.DataFrame({"Z_Scratch":clas})
    for i in lt:
        q=vecs[:,i]
        df4=df2.dot(q.T)
        df3.insert(i,l[i],df4,True)
        print("For the value of l = ",i+1)
        print("KNN method is as follows ")
        complete_knn(df3)
        print("Bayes method as follows ")
        gauss(df3)

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
def complete_knn(df):
    df2=df.copy()
    x1,x2,y1,y2=train_testsplit(df2,0.3)
    max=-1
    pos=0
    for i in range(len(k)):
        y=classifier(x1,x2,y1,y2,k[i])
        if(max<y):
            max=y
            pos=i
    print("\n Accuracy is maximum for value of k = ",k[pos] ," which is ",max)

def gauss(df):
    df2=df.copy()
    gnb=GaussianNB()
    x1,x2,y1,y2=train_testsplit(df2,0.3)
    gnb.fit(x1,y1)
    ypred=gnb.predict(x2)
    confusion(y2,ypred)
    print("\n Accuracy of bayes method is ",accuracy_score(y2,ypred))

def gauss1(df):
    clas=df["Z_Scratch"]
    a=[]
    b=[]
    for i in range(len(clas)):
        if(clas[i]==1):
            a.append(i)
        else:
            b.append(i)
    df2=df.copy()
    df2.drop(a,inplace=True)
    df3=df.copy()
    df3.drop(b,inplace=True)
    x1,x2,y1,y2=train_testsplit(df,0.3)
    cov=np.cov(x1.T)
    mn=x1.mean()
    df2=df2.drop(["Z_Scratch"],1)
    df3=df3.drop(["Z_Scratch"],1)
    y3=multivariate_normal.pdf(x2, df2.mean(),df2.cov(),allow_singular=True)
    y4=multivariate_normal.pdf(x2,  df3.mean(),df3.cov(),allow_singular=True)
    print("length of y3",len(y4))
    l1=190
    l2=391
    y5=[]
    for i in range(len(x2)):
        if(l1*y3[i]>l2*y4[i]):
            y5.append(0)
        else:
            y5.append(1)
    print(len(y5))
    print(len(y2))
    print(confusion(y2,y5))
    print("Baye accuracy score is ",accuracy_score(y2,y5))

def main():
    df=read_data("SteelPlateFaults-2class.csv")
    print("For original Data")
    #df=standardize(df)
    #complete_knn(df)
    print("Gaussian Bayes method follows")
    gauss(df)
    #dim_reduction(df)



if(__name__=="__main__"):
    main()
