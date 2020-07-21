# Nitesh
# Lab Assignment1 for Data Visualisation

import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
d=pd.read_csv("winequality-red.csv",sep=";")
s=input("Enter input of column")
l=d[s]
q=d["quality"]
print(st.mean(l))
print(st.median(l))
print(st.mode(l))
print(min(l))
print(max(l))
plt.scatter(l,q)
plt.xlabel(s)
plt.ylabel("quality")
plt.show()
print("The value of coefficient is",pearsonr(l,q))
plt.hist(l,bins=50)
plt.show()
q1=d.groupby("quality")
p=q1.get_group(5)
ph=p["pH"]
plt.hist(ph)
plt.show()
#d.boxplot(column = s)
plt.boxplot(l)
plt.show()
plt.boxplot(q)
plt.show()