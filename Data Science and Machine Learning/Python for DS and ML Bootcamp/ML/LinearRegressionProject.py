import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
plt.show
import sklearn 

customers = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/Ecommerce Customers')
customers.head()
customers.info()
customers.describe() 

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y='Length of Membership',data=customers,kind='hex')
sns.pairplot(customers) 

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

x=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y=customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)

predicts = lm.predict(X_test)

sns.scatterplot(x=y_test,y=predicts) 

from sklearn import metrics 
mae = metrics.mean_absolute_error(y_test,predicts)
mse = metrics.mean_squared_error(y_test,predicts)
rmse = np.sqrt(metrics.mean_squared_error(y_test,predicts))

sns.distplot((y_test - predicts),bins=50)

cdf = pd.DataFrame(lm.coef_.transpose(),X_train.columns,columns=['Coeff'])
cdf

#Although the coefficients point to the length of membership being the most relevant feature in the increase in Yearly Amount Spent, however
#length of membership also goes in units of 1 year (roughly) which may be hard to create revenue from given the amount of effor that needs to be devoted to each additional year of membership (die out due to age increase?)
#thus, the time on the App should be given the most effor, with membership length in a close second (this is assuming the cost of increasing the time on the app to the amount of yearly increase in revenue is a lesser ratio than that of cost of increasing membership length)
