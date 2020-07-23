import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse
def center(x):
    x=x.copy()
    x=x-np.mean(x,axis=0)
    return x

x=np.random.multivariate_normal([0,0],[[7,10],[10,18]],1000)
print(x.shape)
plt.scatter(x[:,0],x[:,1])
plt.show()

x1=center(x)
vals,vecs=np.linalg.eig(x1.T.dot(x))
print(vecs)
q=vecs[:,1]
print(q,vals)
q=q.reshape(2,1)
print(q)
newx=x1.dot(q)
print(newx.shape)
x3=newx.dot(q.T)
xpos=0
ypos=0
print(vecs)
xdirect=vecs[0,0]
ydirect=vecs[0,1]
plt.scatter(x1[:,0],x1[:,1])
plt.quiver(xpos,ypos,xdirect,ydirect,scale=5)
plt.quiver(0,0,vecs[1,0],vecs[1,1],scale=4)
plt.show()
print(round(mse(x1,x3),2))

result = pd.DataFrame(newx, columns=['PC1'])
result['PC1']=x[:,1]
result['y-axis'] = x[:,0]
plt.scatter(x3[:,0],x3[:,1])
plt.show()
sns.lmplot('PC1', 'y-axis', data=result, fit_reg=False)