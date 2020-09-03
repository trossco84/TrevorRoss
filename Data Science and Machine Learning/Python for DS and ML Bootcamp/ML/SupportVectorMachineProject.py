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

from IPython.display import Image

#Iris Setosa
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url=url,width=300,height=300)

#Iris Versicolor
url2 = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url=url2,width=300,height=300)

#Iris Virginica
url3 = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url=url3,width=300,height=300)

#Data
#three classes of Iris (n=50 for each),4 features in cm: sepal length and width, petal length and width
iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species',diag_kind='hist')
sns.kdeplot(iris[iris['species']=='setosa']['sepal_width'],iris[iris['species']=='setosa']['sepal_length'],shade=True,shade_lowest=False)

#build the model
iris.head()
X=iris.drop('species',axis=1)
y=iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.svm import SVC
yuh=SVC()
yuh.fit(X_train,y_train)
pyuh = yuh.predict(X_test)

print(confusion_matrix(y_test,pyuh))
print(classification_report(y_test,pyuh))

#Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.01,0.1,1,10,100,1000],'gamma':[1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=5)
grid.fit(X_train,y_train)
grid.best_params_
#best is c 100 gamma 0.01, i'm gonna play around those ranges now
crange = np.arange(80,121)
grange = np.arange(0.008,0.012,0.0001)
p2 = {'C':crange,'gamma':grange}
grid2 = GridSearchCV(SVC(),p2,verbose=3)
grid2.fit(X_train,y_train)
grid2.best_params_
#more searching
cr2=np.arange(75,83)
gr2=np.arange(0.01,0.013,0.001)
p3={'C':cr2,'gamma':gr2}
grid3=GridSearchCV(SVC(),p3,verbose=3)
grid3.fit(X_train,y_train)
grid3.best_params_

pg3=grid3.predict(X_test)
print(confusion_matrix(y_test,pg3))
print(classification_report(y_test,pg3))
#in this case the model performs just as well
