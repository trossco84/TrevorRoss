import pandas as pd 
import matplotlib.pyplot as plt 
df3 = pd.read_csv('df3-Copy1')

%pip install deap
import deap

df3.info() 
df3.head() 

df3.plot.scatter(x='a',y='b',figsize=(12,3),c='red')

df3.plot(kind='scatter',x='a',y='b',figsize=(12,3),c='red')

df3['a'].hist(bins=30)

df3[['a','b']].plot.box()

df3['d'].plot.kde()

df3['d'].plot.kde(ls='--',lw=2)

#must be run together
df3[0:31].plot.area(alpha=.5,legend=False)
plt.legend(loc='center left',bbox_to_anchor=(1.0,0.5))