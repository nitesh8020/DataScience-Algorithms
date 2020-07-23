import pandas as pd
import numpy as np
import sklearn.preprocessing as pr
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import statistics as st
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal as mvn
l=["pregs","plas","pres","skin","test","BMI","pedi","Age","class"]
k=[1,3,5,7,9,11,13,15,17,21]

def read_data(file_path):
    df=pd.read_csv(file_path)
    return df

def center(df):
    df2=df.copy()
    for x in range(len(l)-1):
        df2[l[x]]=df2[l[x]]-df2[l[x]].mean()
    return df2

def train_testsplit(df):
    df2=df.copy()
    X=np.array(df2.drop(['class'],1))
    Y=np.array(df2["class"])
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
    return X_train,X_test,Y_train,Y_test

def classifier(xtrain,xtest,ytrain,ytest,k):
    clf=KNeighborsClassifier(n_neighbors=k)
    clf.fit(xtrain,ytrain)
    accuracy=clf.score(xtest,ytest)
    z=clf.predict(xtest)
    print("Value of k is",k)
    confusion(ytest,z)
    print("The avlue of accuracy is",accuracy)
    return(accuracy)

def confusion(ytrue,ypred):
    print(confusion_matrix(ytrue,ypred))
    
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

def new_reduction(df):
    df2=df.copy()
    features=["pregs","plas","pres","skin","test","BMI","pedi","Age"]
    pca = PCA(n_components=2)
    x = df2.loc[:, features].values
    y = df2.loc[:,['class']].values
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, df[['class']]], axis = 1)
    complete_knn(finalDf)
    gauss(finalDf)
    
def dim_reduction(df):
    df2=df.copy()
    df2=center(df2)
    vals,vecs=np.linalg.eig(df2.T.dot(df2))
    idx =vals.argsort()[::-1]   
    vals = vals[idx]
    vecs = vecs[:,idx]
    l=[0,1,2,3,4,5,6,7]
    clas=df2["class"]
    df3=pd.DataFrame({"class":clas})
    for i in l:
        q=vecs[:,i]
        df4=df2.dot(q.T)
        df3.insert(i,l[i],df4,True)
        print("For the value of l = ",i+1)
        print("KNN method is as follows ")
        #complete_knn(df3)
        print("Bayes method as follows ")
        #gauss(df3)
        bayesian(df3)

def complete_knn(df):
    df2=df.copy()
    x1,x2,y1,y2=train_testsplit(df2)
    a=[]
    for i in range(len(k)):
        y=classifier(x1,x2,y1,y2,k[i])
        a.append(y)
    plt.scatter(k,a)
    plt.show()
    
def gauss(df):
    df2=df.copy()
    gnb=GaussianNB()
    x1,x2,y1,y2=train_testsplit(df2)
    gnb.fit(x1,y1)
    ypred=gnb.predict(x2)
    confusion(y2,ypred)
    print("Accuracy is ",accuracy_score(y2,ypred))

def bayesian(df):
    df2=df.copy()
    x1,x2,y1,y2=train_testsplit(df2)
    idx1=[]
    idx0=[]
    x0=x1
    xt=x1
    list=[1,2,3,4]
    
    for i in range(len(x1)):
        if(y1[i]==0):
            idx0.append(i)
        else:
            idx1.append(i)
    x0=np.delete(x0,idx1,axis=0)
    lx0=len(x0)
    x1=np.delete(x1,idx0,axis=0)
    lx1=len(x1)
    for ln in list:
        gmm1=GaussianMixture(n_components=ln).fit(x1)
        gmm0=GaussianMixture(n_components=ln).fit(x0)
        w0=gmm0.weights_
        w1=gmm1.weights_
        predy=[]
        for j in range(len(x2)):
            s1=0
            s0=0
            for i in range(ln):
               s0=s0+w0[i]*mvn.pdf(x2[j],cov=gmm0.covariances_[i], mean=gmm0.means_[i],allow_singular=True)
               s1+=w1[i]*mvn.pdf(x2[j],cov=gmm1.covariances_[i], mean=gmm1.means_[i],allow_singular=True)
            if(lx0*s0>s1*lx1):
                predy.append(0)
            else:
                predy.append(1)
        print("For n_components = ",ln)
        print(accuracy_score(y2,predy))
        print(confusion(y2,predy))
    

        
    #print(accuracy_score(y2,predy))
    #print(confusion(y2,predy))
    #print(probs)
            

def main():
    df=read_data("pima-indians-diabetes.csv")
    #print("For original Data")
    #complete_knn(df)
    #print("Gaussian Bayes method follows")
    #gauss(df)
    #dim_reduction(df)
    #new_reduction(df)
    bayesian(df)
    #dim_reduction(df)
    


if(__name__=="__main__"):
    main()


