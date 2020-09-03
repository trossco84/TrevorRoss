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

#Creating Data (artificial)
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=101)
data    
#actually a tuple
data[0]
#an item in the tuple is an array
plt.scatter(data[0][:,0],data[0][:,1],c=data[1])

#Build it
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4)
#have to know number of clusters before
km.fit(data[0])
km.cluster_centers_
km.labels_
#if youre working with real data and you didnt have the labels you would be done at this stage

fig , (ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(10,6))
ax1.set_title('K  Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=km.labels_,cmap='rainbow')
ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
