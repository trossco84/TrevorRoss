#import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import cufflinks as cf 
plt.show 
cf.go_offline()
sns.set_style('whitegrid')

#Data
kdata = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/KNN_Project_Data')
kdata.head()
sns.pairplot(kdata,hue='TARGET CLASS',diag_kind='hist')

#Standardize the Variables
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(kdata.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(kdata.drop('TARGET CLASS',axis=1))
kfeat = pd.DataFrame(scaled_features,columns = kdata.columns[:-1])

#Build the Model
from sklearn.model_selection import train_test_split
X = kfeat
y = kdata['TARGET CLASS']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

#Predictions and Evaluations
predicts = knn.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predicts))
print('\n')
print(classification_report(y_test,predicts))

#Elbow method to choose a k value
error_rate=[]
for i in range(1,75):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    p_i = knn.predict(X_test)
    error_rate.append(np.mean(p_i != y_test))

plt.figure(figsize=(12,8))
plt.plot(range(1,75),error_rate,color='blue',ls='--',marker='o',markerfacecolor='red',markersize=10)
plt.title=('Error Rate vs K size')
plt.xlabel('K')
plt.ylabel('Error Rate')

#Optimal k = 14
knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(X_train,y_train)
pk14 = knn.predict(X_test)
print(confusion_matrix(y_test,pk14))
print('\n')
print(classification_report(y_test,pk14))