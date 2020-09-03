#import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import cufflinks as cf 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
plt.show 
cf.go_offline()
sns.set_style('whitegrid')
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#print(confusion_matrix(y_test,psvc))
#print(classification_report(y_test,psvc))

#data
college = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/Refactored_Py_DS_ML_Bootcamp-master/17-K-Means-Clustering/College_Data',index_col=0)
college.head()
college.info()
college.describe()

sns.scatterplot(x='Room.Board',y='Grad.Rate',data=college,hue='Private')
sns.scatterplot(x='Outstate',y='F.Undergrad',data=college,hue='Private')

g=sns.FacetGrid(college,hue='Private',palette='seismic',size=6,aspect=2)
g=g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

h=sns.FacetGrid(college,hue='Private',palette='seismic',size=6,aspect=2)
h=h.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

#replacing a specific value in the dataframe
college[college['Grad.Rate']>100]
college.loc['Cazenovia College']['Grad.Rate']
college.replace(to_replace=college.loc['Cazenovia College']['Grad.Rate'],value=100,inplace=True)


#kmeans
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)
km.fit(college.drop('Private',axis=1))
km.cluster_centers_
km.labels_

#evaluating
college['Cluster'] = college['Private'].apply(lambda x: 1 if x=='Yes' else 0)
college.head()

print(confusion_matrix(college['Cluster'],km.labels_))
print(classification_report(college['Cluster'],km.labels_))
