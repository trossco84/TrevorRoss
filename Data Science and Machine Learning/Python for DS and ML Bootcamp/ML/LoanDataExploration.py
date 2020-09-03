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

#Data
loans = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random-Forests/loan_data.csv')
loans.head()
loans.info()
loans.describe()

sns.distplot(loans[loans['credit.policy']==1]['fico'],kde=False,color='blue',label='Credit Policy=1')
sns.distplot(loans[loans['credit.policy']==0]['fico'],kde=False,color='red',label='Credit Policy=0')
plt.legend()

sns.distplot(loans[loans['not.fully.paid']==1]['fico'],kde=False,color='blue',label='Not Fully Paid=1')
sns.distplot(loans[loans['not.fully.paid']==0]['fico'],kde=False,color='red',label='Not Fully Paid=0')
plt.legend()

sns.countplot(x='purpose',data=loans,hue='not.fully.paid')
sns.jointplot(x='fico',y='int.rate',data=loans)

sns.lmplot(x='fico',y='int.rate', col='not.fully.paid',data=loans,hue='credit.policy')

loans.info()

#The 'purpose' variable is a category, we need to make that into a dummy variable
cat_feats=['purpose']
l2 = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
#from the dataframe loans, make dummy variables for the 'purpose' category, and drop the first dummy variable (essentially an else statement)

#Build Decision Tree Model and Evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

X = l2.drop('not.fully.paid',axis=1)
y = l2['not.fully.paid']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pdt = dtree.predict(X_test)
print(confusion_matrix(y_test,pdt))
print(classification_report(y_test,pdt))

#Build Random Forest Model and Evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

X = l2.drop('not.fully.paid',axis=1)
y = l2['not.fully.paid']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

rforest = RandomForestClassifier(n_estimators=600)
rforest.fit(X_train,y_train)
prf = rforest.predict(X_test)
print(confusion_matrix(y_test,prf))
print(classification_report(y_test,prf))

