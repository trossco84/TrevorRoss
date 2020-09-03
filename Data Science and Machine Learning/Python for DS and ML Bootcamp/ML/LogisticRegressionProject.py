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
ad_data = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()

#Exploratory Data Analysis
#age histogram
ad_data['Age'].hist(bins=30)
#area income vs age jointplot
sns.jointplot(x='Age',y='Area Income',data=ad_data)
#jointplot of the kde distributions of daily time spent on site vs age
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red')
#jointplot daily time on site vs daily internet usage
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')
#pairplot with hue of clicked on ad
sns.pairplot(ad_data,hue = 'Clicked on Ad',diag_kind='hist')

#Logistic Regression
ad_data.head()
ad_data['Ad Topic Line'].value_counts()
ad_data['Ad Topic Line'].nunique()
#ad topic line can be dropped as each entry is a unique value
ad_data.drop('Ad Topic Line',axis=1,inplace=True)
ad_data['City'].nunique()
#city can be dropped
ad_data.drop('City',axis=1,inplace=True)
ad_data['Country'].nunique()
adnc = ad_data.drop('Country',axis=1)
adnc.head()
ad_data.drop('Timestamp',axis=1,inplace=True)
adnc.head()
sns.heatmap(adnc.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#no missing data and columns have been selected, time to build model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
y = adnc['Clicked on Ad']
X = adnc.drop('Clicked on Ad', axis = 1)
X_train,X_Test,y_train,y_test = train_test_split(X,y,test_size=0.35)
logmod = LogisticRegression()
logmod.fit(X_train,y_train)

#predict the test set and evaluate results
predictions = logmod.predict(X_Test)

print(classification_report(y_test,predictions,))

