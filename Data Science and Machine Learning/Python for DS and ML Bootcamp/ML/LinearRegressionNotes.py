# %pip install scikit-learn
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)

import numpy as np 

from sklearn.model_selection import train_test_split
X,y=np.arange(10).reshape((5,2)),range(5)
X
list(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
X_train 
y_train 
X_test 
y_test 

#training supervised model
model.fit(X_train,y_train)

predictions = model.predict(X_test)
predictions
y_test 
model.score(X_test,y_test)

##LINEAR REGRESSION##
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
plt.show 
df = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv')
#checkout the dataframe
df.head()
df.info() 
df.describe() 
df.columns 

sns.pairplot(df) 
sns.distplot(df['Price'])
sns.heatmap(df.corr())

from sklearn.model_selection import train_test_split 

x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df[['Price']]

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)

print(lm.intercept_)
lm.coef_

cdf = pd.DataFrame(lm.coef_.transpose(),X_train.columns,columns=['Coeff'])
cdf

#boston dataset
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()

#Predictions
predictions = lm.predict(X_test)
predictions 
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions))

#Evaluating
from sklearn import metrics 
metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions) 
np.sqrt(metrics.mean_squared_error(y_test,predictions)) 