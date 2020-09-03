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

#Data
#%pip install html5lib
#%pip install bs4
# import html5lib
# from bs4 import BeautifulSoup
#df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
#pd.read_excel('Excel_Sample.xlsx',sheetname='Sheet1')
#%pip install xlrd
import xlrd
#Regular Season Data
traeyoung = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/ML/MM2017.csv',index_col=0)
traeyoung.head()
traeyoung = traeyoung.sort_index(ascending=True)
traeyoung.head()
tackofall = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/ML/MM2018.csv',index_col=0)
tackofall = tackofall.sort_index(ascending=True)
tackofall.head()

#Post Season Data
results2017 = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/ML/MMR2k17.csv',index_col=0)
results2017.head()
results2018 = pd.read_csv('/Users/tross/Desktop/Coding/Python for DS and ML Bootcamp/ML/MMR2k18.csv',index_col=0)
# results = pd.concat([results2017,results2018])
# results.dropna(inplace = True)
# results['target'] = results['REB'] + results['AST'] + results['PTS']
# len(results)

ty = traeyoung.dropna()
ty.set_index('Player')
r217 = results2017.dropna()
r217['target'] = r217['REB'] + r217['AST'] + r217['PTS']
r217.set_index('Player')
r217['GPNCAA'] = r217['GP']
r7=r217[['Player','target']]

tf = tackofall.dropna()
tf.set_index('Player')
r218 = results2018.dropna()
r218['target'] = r218['REB'] + r218['AST'] + r218['PTS']
r218.set_index('Player')
r8=r218[['Player','target']]

len(r8)
len(tf)

one = pd.merge(ty,r8,on='Player')
one.set_index('Player')

one['Team'].value_counts()
len(one)

X=one.drop('target',axis=1)
y=one['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.linear_model import LinearRegression
take1 = LinearRegression()
take1.fit(X_train,y_train)













#now we need team data to kick out all the players who didnt make the tournament (by team abrv)
#we also want team data on each team's ranking so as to help be a predicting factor
#need to create a train set of tournament data for the target variable
#targetvariable is the sum of Points, Rebounds, and Assists
#results.columns
#results['target'] = results['REB'] + results['AST'] + results['PTS']

luka = pd.concat([traeyoung,tackofall])
luka.dropna(inplace=True)
len(luka)

#restar = results[['Player', 'target']]
#restar.set_index('Player',inplace=True)
luka.set_index('Player',inplace=True)

#dababy = luka.join(restar,how='inner')
#dababy.head()
