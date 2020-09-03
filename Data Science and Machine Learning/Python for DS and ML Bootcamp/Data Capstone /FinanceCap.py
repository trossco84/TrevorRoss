#from pandas_datareader import data, wb 
#from pandas import data,wb
#%pip install pandas_datareader 

import pandas_datareader as pdr 
from pandas_datareader import data as web, wb
import pandas as pd 
import os 
import numpy as np 
from datetime import datetime 
import matplotlib.pyplot as plt 
import seaborn as sns
plt.show 

##Exploratory data analysis of Stock Prices
#Focusing on bank stocks and seeing how they progressed throughout the financail crisis -> 2016
#pandas datareader allows you to read stock information directly from the internet

#Data
#figure out how to get the stock data from 1/1/2006 to 1/1/2016 for the following banks:
# Bank of America (BAC)
# CitiGroup (C)
# Goldman Sachs (GS)
# JPMorgan Chase (JPM)
# Morgan Stanley (MS)
# Wells Fargo (WFC)

#set each bank to  a separate dataframe with the variable name being its ticker symbol

#Step 1, setting start and end dates
st = datetime(year=2006, month=1, day=1)
en = datetime(year=2016, month=1, day=1)

#Step 2, Figure out the ticker symbol for each bank
#not sure if this needs to be coded? all i did was google

#Step 3, use datareader to grab info on the stock using yahoo
tickers = 'BAC C GS JPM MS WFC'.split()
BAC = web.DataReader('BAC','yahoo',st,en)
C = web.DataReader('C','yahoo',st,en)
GS = web.DataReader('GS','yahoo',st,en)
JPM = web.DataReader('JPM','yahoo',st,en)
MS = web.DataReader('MS','yahoo',st,en)
WFC = web.DataReader('WFC','yahoo',st,en)

#Step 3.1, using tingo because yahoo was down
# heres the link: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
#you need to create an account
# BAC = pdr.get_data_tiingo('BAC',api_key=os.getenv('TINGO API KEY'))

# #Step 3.2, using alpha vantage (preferred data provider), documentation :https://www.alphavantage.co/documentation/
# #data is loaded in as 'web' in this examples
 apiki = 'ZE7E66ACGTISDOFE'
# BAC = web.DataReader('BAC','av-daily',start=st,end=en,api_key=apiki)
# C = web.DataReader('C','av-daily',start=st,end=en,api_key=apiki)
# GS = web.DataReader('GS','av-daily',start=st,end=en,api_key=apiki)
# JPM = web.DataReader('JPM','av-daily',start=st,end=en,api_key=apiki)
# MS = web.DataReader('MS','av-daily',start=st,end=en,api_key=apiki)
# WFC = web.DataReader('WFC','av-daily',start=st,end=en,api_key=apiki)

#Concatenate the Dataframes
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], keys = tickers,axis=1)

#set the column name levels and check the head
bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']
bank_stocks.head()

#Time to explore the data
#What is the max close price for each bank's stock price throughout the time period
bank_stocks.xs(key = 'Close',level='Stock Info',axis=1).max()

#New returns dataframe
returns = pd.DataFrame()
# bank_stocks.index[0]
# pd.pct_change()
# len(bank_stocks.index)
# ti = 'BAC'
# bank_stocks.xs(key = 'Close',level='Stock Info',axis=1)[ti][1]
# len(bank_stocks.columns.levels[0])

for i in range(0,len(bank_stocks.columns.levels[0])):
    tick = bank_stocks.columns.levels[0][i]
    tickreturns = [np.nan]
    dates = [bank_stocks.index[0]]
    for j in range(1,len(bank_stocks.index)):
        dates.append(bank_stocks.index[j])
        close1 = bank_stocks.xs(key = 'Close',level='Stock Info',axis=1)[tick][j-1]
        close2 = bank_stocks.xs(key = 'Close',level='Stock Info',axis=1)[tick][j]
        tr = (close2/close1)-1
        tickreturns.append(tr)
    
    if (i == 0):
        returns['Dates'] = dates
    tickname = tick + " Return"
    returns[tickname] = tickreturns
    if (i == 5):
        returns.set_index('Dates',inplace=True)

# returns.set_index('Dates',inplace=True)
returns.head()

#Solution 
returns2=pd.DataFrame()
for tick in tickers:
    returns2[tick+'Return']=bank_stocks[tick]['Close'].pct_change()
returns2.head()    
sns.pairplot(returns2)

#pairplot
sns.pairplot(returns[1:])

#Citigroup's data seems to more normalized than the solution, this may have come from a correction
#also could be due to different data source (google no longer applicable)

#figure out the dates each bank stock had the best and worst single day returns
#Worst
returns.idxmin()

#Best
returns.idxmax()

#Standard Deviation of returns
returns.std()
#riskiest stock is CitiGroup because they have largest stadard deviation
#std of returns from 2015
returns.ix['2015-01-01':'2015-12-31'].std()
returns.loc['2015-01-01':'2015-12-31'].std()

#Create a seaborn dist plot of the 2015 returns for morgan stanley
sns.distplot(returns['MS Return'].loc['2015-01-01':'2015-12-31'],bins=100,color='green')

#create a distplot for 2008 returns for Citigroup
sns.distplot(returns['C Return'].loc['2008-01-01':'2008-12-31'],bins=100,color='red')


##VISUALIZATION SECTION##
sns.set_style('whitegrid')
import plotly
import cufflinks as cf 
cf.go_offline()

#create a line plot showing the close price for each bank for the entire index of time
sns.lineplot(data = bank_stocks.xs(key='Close',level='Stock Info',axis=1))
for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()

#Moving Averages
#plot the 30 day average against the close price for bank of americas stock for the year 2008
bank_stocks.head()
plt.figure(figsize=(12,6))
BAC['Close'].loc['2008-01-01':'2008-12-31'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].loc['2008-01-01':'2008-12-31'].plot(label='BAC Close')
plt.legend()

#heatmap showing correlation between stocks close price, also clustermap
sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


#extras
BAC[['Open', 'High', 'Low', 'Close']].ix['2015-01-01':'2016-01-01'].iplot(kind='candle')
MS['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')
BAC['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='boll')

