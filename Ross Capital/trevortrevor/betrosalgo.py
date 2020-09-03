#Importing in necessary libraries
import pandas_datareader as pdr 
from pandas_datareader import data as web, wb
import pandas as pd 
import os 
import numpy as np 
from datetime import datetime 
import matplotlib.pyplot as plt 
import seaborn as sns

#Setting initial settings
plt.show 
from datetime import datetime
from datetime import timedelta

## This is a data acquisition and analysis pipeline to help analyze a stock buying strategy proposed by Trevor Betros. The strategy is simple in nature: if a stock goes down by 15% or greater on any given day, then, after a 4 day waiting period, purchase the stock the next time its returns are negative on two consecutive trading days. The initial analysis and pipeline creation was performed over the period January 1st 2018 - January 1st 2020. The selected stocks were the 100 highest volume stocks on a randomly selected day in February 2020. Given that the stocks were selected in 2020, some stocks do not have data going back two years (reducing the number of buying opportunities)

#Useful functions for viewing data. df means dataframe, replace this value with whatever dataset you'd like to view
#shows the dataframe
df = pd.DataFrame()
df 
#shows the first 5 records of the dataframe, putting any number between the () will show that many records
df.head()
#shows the length (number of records) of the dataframe
len(df)
#copies dataframe to clipboard (copy + paste)
df.to_clipboard()

####SETTING UP THE PIPELINE####
# setting the date and time range
st = datetime(2018, 1, 1)
en = datetime(2020, 1, 1)

# setting the tickers to view, some of these are unavailable for periods at a time
bticks = 'SPCE AMD F NIO FCEL S GE AAPL BAC PLUG SNAP I AUY GNPX BLPH GOLD MSFT NOK VALE RIG MU ROKU TEVA VEON UBER PBR ABEV WFC XOM TSLA KHC INO T KR BBD ZNGA CSCO LK BBBY KGC WORK FCX ITUB FB AGRX NVDA CTL MRO BB BMY PFE FLR TWTR IQ BBVA HL BABA M SIRI ET BILI HMY INTC WMT JD QD MDT SLB CMCSA PINS VZ RESI VG CAG TSM BTG NLY AMAT IMGN HSBC CPE SAN GFI BYND RF VOD TME LYFT GM EBAY BP C MOS OXY PAVM SPWR MRK SBGL CGC DIS'.split()

#Create a DataFrame for each stock, for the defined time period.
#Data pulled: High, Low, Open, Close, Volume, and Adjusted Close
for i in bticks:
    vars()[i] = web.DataReader(i,'yahoo',st,en)

#Create a Dataframe that combines all of the separate stock's dataframes, this is the master dataset
stockdata = pd.concat([SPCE ,AMD ,F ,NIO ,FCEL ,S ,GE ,AAPL ,BAC ,PLUG ,SNAP ,I ,AUY ,GNPX ,BLPH ,GOLD ,MSFT ,NOK ,VALE ,RIG ,MU ,ROKU ,TEVA ,VEON ,UBER ,PBR ,ABEV ,WFC ,XOM ,TSLA ,KHC ,INO ,T ,KR ,BBD ,ZNGA ,CSCO ,LK ,BBBY ,KGC ,WORK ,FCX ,ITUB ,FB ,AGRX ,NVDA ,CTL ,MRO ,BB ,BMY ,PFE ,FLR ,TWTR ,IQ ,BBVA ,HL ,BABA ,M ,SIRI ,ET ,BILI ,HMY ,INTC ,WMT ,JD ,QD ,MDT ,SLB ,CMCSA ,PINS ,VZ ,RESI ,VG ,CAG ,TSM ,BTG ,NLY ,AMAT ,IMGN ,HSBC ,CPE ,SAN ,GFI ,BYND ,RF ,VOD ,TME ,LYFT ,GM ,EBAY ,BP ,C ,MOS ,OXY ,PAVM ,SPWR ,MRK ,SBGL ,CGC ,DIS],axis=1,keys=bticks)
stockdata.columns.names = ['Ticker', 'Stock Info']

#Creates a DataFrame of each stocks returns over the given time period
returns = pd.DataFrame()
for tick in bticks:
    returns[tick+' Return'] = stockdata[tick]['Close'].pct_change()


#Creating a 'buy matrix' that determines the stocks to buy on what dates, the output matrix of this code snippet gives the coordinates in the stock dataframe 
buyx1 = []
buyy1 = []

for j in range(0,100):
    for i in range(0,504):
        if (returns.iloc[i,j] <= (-0.15)):
            z = i + 4
            for y in range(z,503):
                if (returns.iloc[y,j] < 0):
                    if (returns.iloc[y+1,j]<0):
                        h = y+1
                        buyx1.append(h)
                        buyy1.append(j)
                        break
                        
#Creates  a dictionary for stock coordinates and their associated name
stockdict = {}
i = 0
for tick in bticks:
    stockdict.update({i:tick})
    i=i+1

for j in range(0,93):
    stocknum = buyy1[j]
    buyy1[j] = stockdict.get(stocknum)

#creates a dictionary for date coordinates and associated date
datedict = {}
for l in range(0,504):
    rowdate = stockdata.index[l]
    datedict.update({i:rowdate})
    i = i+1

for c in range(0,93):
    datenum = buyx1[c]
    buyx1[c] = datedict.get(datenum)

#Creates the matrix for when to buy what stocks
buymatrix = pd.DataFrame({'BuyDate':buyx1,'Stock':buyy1})

#drops the duplicated rows (still debugging why they are added in the first place)
buymatrix.drop_duplicates(inplace=True)
#buymatrix should have a length of 83 after this, going forward "83" will have to be dynamic

#creates columns in the buymatrix that will be inputted with closing price data for each date 30 days out from buying date
for i in range(1,31):
    columntitle = 'ClosingPrice' + str(i)
    buymatrix[columntitle] = range(0,82)
    buymatrix[columntitle] = buymatrix[columntitle].astype(float)

#NEED TO DROP NA DATES THAT WERE PULLED
buymatrix.dropna(inplace=True)
#pulls data from yahoo to input into each column the closing price at each date
for p in range(0,81):
    daybought = buymatrix.iloc[p,0]
    stockname = buymatrix.iloc[p,1]
    daystop = daybought + timedelta(days=60)
    stocknamedata = web.DataReader(stockname,'yahoo',daybought,daystop)
    for t in range(0,30):
        closingprice = stocknamedata['Close'][t]
        buymatrix.iloc[p,2+t] = closingprice

buymatrix.head(30)
#Poor Data Visualization, will come back to this later
sns.heatmap(buymatrix.drop(['BuyDate','Stock'],axis=1),col)
fig = plt.figure(figsize=(25,25))
sns.boxplot(data = buymatrix.drop(['BuyDate','Stock'],axis=1))
sns.clustermap(buymatrix.drop(['BuyDate','Stock'],axis=1))

#Test if this works
buymatrix['TradeID'] = buymatrix.apply((str(buymatrix['BuyDate']) + buymatrix['Stock']),axis=1)



#After a few months of learning and discovering, I will now reattempt this strategy proposed by Trev.
import pandas as pd 
import numpy as np 
from fastquant import get_stock_data

#Data Reading and Cleaning
okherewego = pd.read_csv('/Users/tross/WILD/RossCo/MyRepo/TrevorRoss/Ross Capital/trevortrevor/stocks2.csv')
ok2 = okherewego.copy()
yearlist = []
ok3 = ok2.dropna()
for yo in ok3['EndDate']:
    yearlist.append(int(yo[-2:]))
ok3['yo'] = yearlist
ok4 = ok3[ok3['yo'] <= 20]
ok5 = ok4[ok4['yo'] >= 10]
stklist = list(ok5['Symbol'])
stklist[-1:]
#Creating a DF of all stock prices
fulldf = pd.DataFrame()
for stk in stklist:
    new_column = stk
    print(stk)
    try:
        fulldf[new_column] = get_stock_data(stk,"2010-1-1", "2010-08-01")['close']
    except:
        pass

fulldf.to_csv('Stocks_and_ClosingPrices_Last10years.csv')

fd2 = pd.read_csv('/Users/tross/WILD/RossCo/MyRepo/TrevorRoss/Ross Capital/trevortrevor/Stocks_and_ClosingPrices_Last10years.csv')
fd3 = fd2.copy()
fd3.head()
fd3['date'] = 
fd3.head()
fd4 = fd3[fd3>=2]
fd3.dropna(axis=1,inplace=True)
fd4=fd3.copy()
fd4.set_index('dt',inplace=True)
fd4.head()
returnsdf = pd.DataFrame()
for stk in fd4.columns:
    returnsdf[stk] = fd3[stk].pct_change()
returnsdf.set_index(fd4.index,inplace=True)
lastday = len(returnsdf)

d3slist4 = []
for stk in returnsdf.columns:
    for dropday in range(0,lastday):
        if (returnsdf[stk][dropday] <= (-0.15)):
            tradeableday = dropday+4
            if (tradeableday+210) > lastday:
                stopday = lastday - 2
            else:
                stopday = tradeableday+200
            for day in range(tradeableday,stopday):
                if (returnsdf[stk][day] < 0):
                    day2 = day+1
                    if (returnsdf[stk][day2] < 0):
                        #stryear = str(returnsdf.index[day2].date().year)
                        #strmonth = str(returnsdf.index[day2].date().month)
                        #strday = str(returnsdf.index[day2].date().day)
                        #buydate = stryear + "-" + strmonth+ "-" + strday
                        buydate = returnsdf.index[day2]
                        tup = (buydate,stk)
                        d3slist4.append(tup)
                        day = stopday+1
                        dropday = stopday+1
                        break

d3slist4[0:30]
dates_and_stocks.to_csv('/Users/tross/WILD/RossCo/MyRepo/TrevorRoss/Ross Capital/trevortrevor/buydates_and_stocks')

dates_and_stocks = pd.DataFrame(d3slist4,columns=['date','stock'])
dates_and_stocks.drop_duplicates(inplace=True)
len(dates_and_stocks)
dates_and_stocks.head()

ds2 = dates_and_stocks.copy()

type(ds2['date'][0])
from datetime import datetime
realdate1 = pd.to_datetime(ds2['date'])
ds2['rd1'] = realdate1
ds2.head()

from datetime import date
from datetime import timedelta
realdate2 = [(day + timedelta(days=400)) for day in realdate1]
realdate2[0].date().year
ds2.head(6)
ds2['date2'] = realdate2
ds2['date2'][5] < date.today()

ds2['filter'] = [1 if dayboy < date.today() else 0 for dayboy in ds2['date2']]
ds3 = ds2[ds2['filter']==1]

ds4 = ds3.drop(['filter'],axis=1)
len(ds4)
ds4['date2']
ds4.head()
ds5 = ds4.reset_index(inplace=True)
ds4.drop(['index'],axis=1,inplace=True)

ds5 = ds4.copy()
ds5['date2'][2].day


end_year_strings = [str(day.year) for day in ds5['date2']]
end_month_strings = [str(day.month) for day in ds5['date2']]
end_day_strings = [str(day.day) for day in ds5['date2']]

stop = len(end_year_strings)
end_dates = []
for y in range(0,stop):
    strday = end_year_strings[y]+"-"+end_month_strings[y]+"-"+end_day_strings[y]
    end_dates.append(strday)
realdate1[0:5]
end_dates[0]
ds2['rd1'][0] 
date.today()- timedelta(400)
ds3 = ds2[ds2['date']]
len(d3slist3)
t = len(d3slist3)
returnsdf.shape 
b = 2663*4419

ds5['end_date'] = end_dates
uuid_list = ds5['stock']+"__"+ds5['date']
end_df = pd.DataFrame()
ds5['UUID'] =uuid_list


final_df = pd.DataFrame()
county=0
tot = len(uuid_list)
for uuid in uuid_list:
    perc = round((county/tot)*100,4)
    perc
    print(f'total progress: {perc}%')
    new_column = uuid
    stk = ds5['stock'][county]
    print(stk)
    d1 = ds5['date'][county]
    d2 = ds5['end_date'][county]
    try:
        newarray = get_stock_data(stk,d1, d2)['close']
        newlist = list(newarray.values)
        final_df[new_column] = newlist
    except:
        pass
    county+=1
final_df.head()

final_df.shape
final_df[uuid].pct_change()

returnsdf = pd.DataFrame()
for uuid in uuid_list:
    try:
        newreturn = final_df[uuid].pct_change()
        returnsdf[uuid] = newreturn
    except:
        pass


import seaborn as sns
sns.lineplot(x=returnsdf.index,y=returnsdf)

sns.s


returns30 = returnsdf[0:30]
returns60 = returnsdf[0:60]
returns90 = returnsdf[0:90]
returns120 = returnsdf[0:120]
returns150 = returnsdf[0:150]
returns180 = returnsdf[0:180]
returns210 = returnsdf[0:210]
returns240 = returnsdf[0:240]

returns1518 = returnsdf[150:180]

returns30.mean().mean()
returns60.mean().mean()
returns90.mean().mean()
returns120.mean().mean()
returns150.mean().mean()
returns180.mean().mean()
returns210.mean().mean()

returns1518.head()

returnsdf.to_csv('returns dataframe.csv')
final_df.to_csv('final closing prices.csv')


r10 = returnsdf.columns[0:10]
r11 = returnsdf[r10]
r11.head()

plt.figure(figsize=(20,12))
plot_title = "Returns over the next 276 Trading Days"
r11.plot(legend=None, title=plot_title,kind='line',linewidth=0.5)
plt.ylim(-.5,.5)

returnsdf.head()
final_df.shape
fulldf.head()
t/b
stk
yo5 = returnsdf.columns.values
yo5
len(yo5)
c = 0
for i in yo5:
    c = c+1
    if i == 'ACUR':
        z = c

finaldf.head()


len(d3slist)

buyx1 = []
buyy1 = []

for j in range(0,100):
    for i in range(0,504):
        if (returns.iloc[i,j] <= (-0.15)):
            z = i + 4
            for y in range(z,503):
                if (returns.iloc[y,j] < 0):
                    if (returns.iloc[y+1,j]<0):
                        h = y+1
                        buyx1.append(h)
                        buyy1.append(j)
                        break
                        
#Creates  a dictionary for stock coordinates and their associated name
stockdict = {}
i = 0
for tick in bticks:
    stockdict.update({i:tick})
    i=i+1

for j in range(0,93):
    stocknum = buyy1[j]
    buyy1[j] = stockdict.get(stocknum)

#creates a dictionary for date coordinates and associated date
datedict = {}
for l in range(0,504):
    rowdate = stockdata.index[l]
    datedict.update({i:rowdate})
    i = i+1

for c in range(0,93):
    datenum = buyx1[c]
    buyx1[c] = datedict.get(datenum)

#Creates the matrix for when to buy what stocks
buymatrix = pd.DataFrame({'BuyDate':buyx1,'Stock':buyy1})

#drops the duplicated rows (still debugging why they are added in the first place)
buymatrix.drop_duplicates(inplace=True)
#buymatrix should have a length of 83 after this, going forward "83" will have to be dynamic

#creates columns in the buymatrix that will be inputted with closing price data for each date 30 days out from buying date
for i in range(1,31):
    columntitle = 'ClosingPrice' + str(i)
    buymatrix[columntitle] = range(0,82)
    buymatrix[columntitle] = buymatrix[columntitle].astype(float)

#NEED TO DROP NA DATES THAT WERE PULLED
buymatrix.dropna(inplace=True)
#pulls data from yahoo to input into each column the closing price at each date
for p in range(0,81):
    daybought = buymatrix.iloc[p,0]
    stockname = buymatrix.iloc[p,1]
    daystop = daybought + timedelta(days=60)
    stocknamedata = web.DataReader(stockname,'yahoo',daybought,daystop)
    for t in range(0,30):
        closingprice = stocknamedata['Close'][t]
        buymatrix.iloc[p,2+t] = closingprice