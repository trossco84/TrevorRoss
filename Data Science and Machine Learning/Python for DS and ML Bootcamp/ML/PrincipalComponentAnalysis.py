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
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
#cancer acts as a dictionary
cancer.keys()
print(cancer['DESCR'])
cancer['target_names']
#find the most important components

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()

from sklearn.preprocessing import StandardScaler
#scale data so each feature has a single unit variance
scaler = StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)

#data frame is just the features so you can just use df
#Intantiate a PCA object
#find the principal components using the fit method
#apply the rotation and dimensionality reduction by calling transform
#can also specify how many components we want to keep when creating the PCA object

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
plt.figure(figsize=(8,6))
#all the rows from column 0 and all the rows from column 1
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
#this essentially turns the 30 different dimensions of data into two (compression algorithm)
#this doesnt relate 1:1 to any of the features, but the components relate to the combinations of features in the dataset
#the components themselves are stored as an attribute of the pca object

pca.components_
#each row represents a principal component
#each column relates back to the features
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')

#feed in x_pca into a classification algrithm (logistic regression), support vector machines would be a good use for this data and its split

