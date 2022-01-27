import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
plt.show() 

df = pd.read_csv('911.csv')

df.info() 
df.head() 

####BASICS####
##What are the top 5 zipcodes for 911 calls
df['zip'].value_counts().head(5)

##What are the top 5 townships (twp) for 911 calls
df['twp'].value_counts().head(5)

##How many unique title codes are there
df['title'].nunique()

####Creating New Features####
##create a new column that is the Reason string value as dictated in the title column
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])

##What is the most common reason for a 911 call
df['Reason'].value_counts()

##Use Seaborn to create a countplot of 911 calls by Reason
sns.countplot(x='Reason',data=df)

##what is the datatyp of the objects in the timeStamp column
type(df['timeStamp'].iloc[0])

#creating three new columns based off of data in timestamp column
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['timeStamp'][0]
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['timeStamp'].apply(lambda x: dmap[x.dayofweek])
df[['Hour','Month','Day of Week']].head(10)

#seaborn countplot of the day of the week column with the hue based off of the reason column
sns.countplot(x='Day of Week',data=df,hue='Reason')
#moves the legend outside of the graph, must be run simultaneously
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)

#same plot for month
sns.countplot(x='Month',data=df,hue='Reason')
#moves the legend outside of the graph, must be run simultaneously
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)

#a few months are missing, need to fill in this information another way
#create a groupby object called byMonth where the df is grouped by the month column 
# and the count() method is used for aggregatins
byMonth = df.groupby('Month').count()
byMonth.head()

#create a simple plot indicating the count of calls per month
byMonth['Day of Week'].plot()

#create a seaborn lm plot to create a linear fit on the number of calls per month
#need to reset the index
byMonth.reset_index(inplace=True)
sns.lmplot(x='Month',y='twp',data=byMonth)

#create a new column called 'Date' that contains the date from timeStamp
df['Date']=df['timeStamp'].apply(lambda x: x.date())

#groupby date column and create plot of counts
byDate = df.groupby('Date').count()
byDate['Day of Week'].plot()

#create the same plot but 3 separate plots with each plot representing a Reason
df[(df['Reason']=='Traffic')].groupby('Date')['Day of Week'].count().plot(title='Traffic')
df[(df['Reason']=='Fire')].groupby('Date')['Day of Week'].count().plot(title='Fire')
df[(df['Reason']=='EMS')].groupby('Date')['Day of Week'].count().plot(title='EMS')

#heatmaps with seaborn and data
#restructure the data frame so that the columns become the cours and the index becomes the day of the week
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()

#heatmap
sns.heatmap(dayHour)

#clustermap
sns.clustermap(dayHour)

#do the same with month
monthHour = df.groupby(by=['Month','Hour']).count()['Reason'].unstack()
monthHour.head(2)

sns.heatmap(monthHour)
sns.clustermap(monthHour)

