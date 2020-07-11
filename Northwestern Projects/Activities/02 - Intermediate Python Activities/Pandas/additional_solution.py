import numpy as np
import pandas as pd
import datetime as dt 
import pathlib as Path 
import matplotlib.pyplot as plt 
plt.show()

##Data and Data Cleaning
whale = pd.read_csv('/Users/tross/WILD/Northwestern/NWREPO/FinTech-Lesson-Plans/02-Homework/04-Pandas/Instructions/Starter_Code/Resources/whale_returns.csv',index_col="Date", infer_datetime_format=True, parse_dates=True)
whale.set_index('Date',inplace=True)
whale.head()
whale.sort_index(inplace=True)
whale.isnull().sum()
whale.dropna(inplace=True)
whale.isnull().sum()

algo = pd.read_csv('/Users/tross/WILD/Northwestern/NWREPO/FinTech-Lesson-Plans/02-Homework/04-Pandas/Instructions/Starter_Code/Resources/algo_returns.csv',index_col="Date", infer_datetime_format=True, parse_dates=True)
algo.sort_index(inplace=True)
algo.dropna(inplace=True)

sp = pd.read_csv('/Users/tross/WILD/Northwestern/NWREPO/FinTech-Lesson-Plans/02-Homework/04-Pandas/Instructions/Starter_Code/Resources/sp500_history.csv',index_col="Date", infer_datetime_format=True, parse_dates=True)
sp.head()
sp.sort_index(inplace=True)
sp.dropna(inplace=True)
#format the prices column
sp['Close'] = sp.apply(lambda x: x['Close'].strip('$'),axis=1)
sp['Close'] = sp.apply(lambda x: float(x['Close']),axis=1)
#calculate daily returns
dailyreturns = sp.pct_change(1)
dailyreturns.head()
dailyreturns.dropna(inplace=True)

#rename column
dailyreturns.rename(columns={'Close':'S&P 500'},inplace=True)


#concatenate all dataframes
returnsdf = pd.concat([whale,algo,dailyreturns],join='inner',axis=1)
returnsdf = returnsdf.pct_change()
returnsdf
#plotting
fig = plt.figure(figsize=(12,8))
plt.plot(returnsdf)

fig = plt.figure(figsize=(12,8))
plt.plot(returnsdf.cumprod())

import math
#risk
returnsdf.plot(kind='box')
rdf_std = returnsdf.std()
rdf_std>rdf_std[6]
returnsdf.head(-5)
rdf_std_standardized=rdf_std*math.sqrt(252)
rdf_std_standardized


#rolling
rollstd = returnsdf.rolling(window=21).std()
rollstd.plot()

corrtable = returnsdf.corr()
corrnosp = corrtable.drop(['S&P 500'],axis=0)
corrnosp['S&P 500'].idxmin()

#beta of algo 1
# calc covariance of algo 1 to the s&p
covalgo1 = returnsdf['Algo 1'].cov(returnsdf['S&P 500'])
covalgo1
# calc variance of s&p
varsp = returnsdf['S&P 500'].var()
varsp 
# calc beta of algo1
beta_algo1 = covalgo1/varsp
beta_algo1
# rolling variance plot
rollcovalgo1 = returnsdf['Algo 1'].rolling(window=21).cov(returnsdf['S&P 500'])
rollvarsp = returnsdf['S&P 500'].rolling(window=21).var()
rollbeta1 = rollcovalgo1/rollvarsp
rollbeta1.plot(figsize=(12,6),title="yooo")

#Exponentially weighted average
ewmbb = returnsdf.ewm(halflife=21).std()
ewmbb.plot(figsize=(12,6))

#Sharpe Ratios
sharpe_ratios = (returnsdf.mean() * 252)/(returnsdf.std() * np.sqrt(252))
sharpe_ratios
sharpe_ratios.plot(kind='bar',title = 'sharpeeeeboyyy')

#Portfolio Returns
appl = pd.read_csv('/Users/tross/WILD/Northwestern/NWREPO/FinTech-Lesson-Plans/02-Homework/04-Pandas/Instructions/Starter_Code/Resources/aapl_historical.csv',index_col="Trade DATE", infer_datetime_format=True, parse_dates=True)
appl.drop(['Symbol'],axis=1,inplace=True)
appl.columns=['AAPL']

aapl = pd.read_csv('/Users/tross/WILD/Northwestern/NWREPO/FinTech-Lesson-Plans/02-Homework/04-Pandas/Instructions/Starter_Code/Resources/aapl_historical.csv',index_col="Trade DATE", infer_datetime_format=True, parse_dates=True)
goog = pd.read_csv('/Users/tross/WILD/Northwestern/NWREPO/FinTech-Lesson-Plans/02-Homework/04-Pandas/Instructions/Starter_Code/Resources/goog_historical.csv',index_col="Trade DATE", infer_datetime_format=True, parse_dates=True)
cost = pd.read_csv('/Users/tross/WILD/Northwestern/NWREPO/FinTech-Lesson-Plans/02-Homework/04-Pandas/Instructions/Starter_Code/Resources/cost_historical.csv',index_col="Trade DATE", infer_datetime_format=True, parse_dates=True)

myport = pd.concat([aapl,goog,cost])
myport.reset_index(inplace=True)

#Pivotting the df:
myport = myport.pivot_table(values='NOCP',index='Trade DATE',columns='Symbol')
myport.head()

#calculating weighted returns
wts = [1/3,1/3,1/3]
myreturns = myport.pct_change()
myrw = myreturns@wts
myrw
returnsdf['myreturns'] = myrw
returnsdf.head()
returnsdf.sort_index(ascending=False)

returnsoleary = returnsdf.dropna()
returnsoleary.sort_index(ascending=False,inplace=True)

roleary = returnsoleary.rolling(window=21).std()
roleary.plot(figsize=(12,8))
quandlapi = 'g1ZZYXFT6m8tFsx11i7J'

corrport = returnsoleary.corr()
corrport