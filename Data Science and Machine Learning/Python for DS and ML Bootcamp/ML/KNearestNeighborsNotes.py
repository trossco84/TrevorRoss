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
df = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/Refactored_Py_DS_ML_Bootcamp-master/14-K-Nearest-Neighbors/Classified Data',index_col=0)
df.head() 

#Using KNN so the variables need to be standardized so everything is to the same scale
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

df_feat = pd.DataFrame(scaled_features,columns = df.columns[:-1])
#the [:-1] is everything but the last one
df_feat.head()

#Build the model
from sklearn.model_selection import train_test_split
X=df_feat
y=df['TARGET CLASS']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

#predictions and evaluation
predicts = knn.predict(X_test)
predicts
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predicts))
print(classification_report(y_test,predicts))

#Elbow method to determine optimal k value
error_rate=[]
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pre_i = knn.predict(X_test)
    error_rate.append(np.mean(pre_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',ls='--',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

#great error rates, 17 is v low in k value and error rate so we'll pick that
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
predicts = knn.predict(X_test)
print(confusion_matrix(y_test,predicts))
print('\n')
print(classification_report(y_test,predicts))