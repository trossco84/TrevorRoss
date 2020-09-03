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

#data is built in breatcancer from sklearn
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])
#Want to predict if the tumor is malignant or benign

df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.head()
df_feat.info()
cancer['target']
cancer['target_names']

#skipping visualization because no domain experience in tumor cells

#build the model
X=df_feat
y=cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
psvc = model.predict(X_test)

print(confusion_matrix(y_test,psvc))
print(classification_report(y_test,psvc))

#If there is an error, we can do grid search
#this happens when the model needs its parameters adjusted
#search for the best parameters using a grid search
#grid search allows you to find the right parameters such as what C or gamma values to use
#usually finding these parameters is a tricky task, but we can be lazy and just try a bunch of combinations and see what works best
#create  a grid of parameters is trying out all the best combinations
from sklearn.model_selection import GridSearchCV
#grid search takes in a dictionary that describes the parameters to be tried and the model to train
#the grid of parameters is defined as a dictionary where the keys are the parameters and the values is basically a list of settings to be tested
#in SVC, 'C' controls the cost of missclassification on the training data
#a large C value gives you low bias and high variance, low bias because you penalize the cost of missclassification
#and reverse for a low C value
#the gamma parameter has to do with the free parameter of the gaussian radial basis function
#you can see the kernel is the radial basis function 'rbf'
#a small gamma means a gaussian with a large variance
#a large gamma value leads to high bias and low variance
# if gamma is large then variance is small implying that the support vecor does not have a widespread influence
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=3)
#gridsearch is a meta estimator, it takes an estimator just like we did with SVC, that support vector classifier, and creates a new estimator thats exactly the same
#verbose is the description while going, lets you know its doing something
grid.fit(X_train,y_train)
#fit runs the same loop with cross validation to find the best parameter combination 
#once it has the best combination it runs fit again on all data passsed to that fit without cross validation
#builds a new model using the best parameter setting
#grab best 
grid.best_params_
grid.best_estimator_
grid.best_score_

pgrid = grid.predict(X_test)
print(confusion_matrix(y_test,pgrid))
print(classification_report(y_test,pgrid))
