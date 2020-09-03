import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.show

train = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/titanic_train.csv')
train.head()

#Exploratory Data Analysis
#returns a boolean dataframe
train.isnull()
#make a heatmap of this 
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Age and Cabin are data points(variables) with a lot of null values
#age we might be able to fix but cabin is too unknown
sns.set_style('whitegrid')

#look at a ratio of the target label (who survived and who didnt)
sns.countplot(x='Survived',data=train)
#survival with hue of sex
sns.countplot(x='Survived',data=train,hue='Sex',palette='RdBu_r')
#this shows a trend of survival between sexex (males did not survive as much)
#keep exploring
sns.countplot(x='Survived',data=train,hue='Pclass')
#non survival is popular amongst those in lowest class
#age distribution
sns.distplot(train['Age'].dropna(),kde=False,bins=30)
train['Age'].plot.hist(bins=35)
train.info()

#siblings spouses data
sns.countplot(x='SibSp',data=train)
#most people traveled alone, second is most likely spouse

#fair data
train['Fare'].hist(bins=40,figsize=(10,4))
#most are cheaper tickets, this makes sense as most passengers as 3rd class

import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=50)

#End of Exploratory Data Analysis

#Data Cleaning Section
#Dealing with Missing Values
#imputation, filling in the empty age values with the mean age of the population
#check average age per passenger class
plt.figure(figsize=(12,10))
sns.boxplot(x='Pclass',y='Age',data=train)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37
        elif Pclass == 3:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#handle the cabin column, we can either create a new feature of whether or not cabin is available
#in this case we'll just drop the column
train.drop('Cabin',axis=1,inplace=True)
#drop the rest of NA
train.dropna(inplace=True)

#Deal with Categorical Features
#Convert Categorical features into dummy variables using pandas
#Otherwise the machine learning algorithm wont be able to directly tak in those features as inputs
pd.get_dummies(train['Sex'])
#one column predicts the other column perfectly (known as multi colinearity)
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
#remove Name, Ticket Embarked, and Sex, passenger id (just an index)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.drop('PassengerId',axis=1,inplace=True)
train.head()
#Pclass is also a categorical column as well, so first we'll look at keeping it in the df like this
#then recreate the model but with Pclass as a dummy variable
#treating train data frame as if it is all our data

#creating logistic regression model
X=train.drop('Survived',axis=1)
y=train['Survived']
import sklearn
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)

#evaluating model
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

#Exploring other ways to increase precision,
#Repeat cleaning process for training and testing data sets and train on all of training data set
#feature engineering, grabbing the title of the 'Name' column as a dummy variable, cabin letter or ticket information