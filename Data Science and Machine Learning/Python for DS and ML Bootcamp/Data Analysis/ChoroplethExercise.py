import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode, iplot 
init_notebook_mode(connected=True)

import pandas as pd 

df = pd.read_csv('2014_World_Power_Consumption-Copy1')

df.head()
len(df)
data = dict(type='choropleth',locations=df['Country'],text=df['Text'],z=df['Power Consumption KWH'],colorbar={'title':'Power Consumption for Countries'},locationmode='country names')
layout1 = dict(title='Power Consumption by KWH', geo=dict(showframe=True,projection={'type':'natural earth'}))

choromap1 = go.Figure(data=[data],layout=layout1)

iplot(choromap1,validate=False)

df2 = pd.read_csv('2012_Election_Data-Copy1')
df2.head()

data2 = dict(type='choropleth',locations=df2['State Abv'],locationmode='USA-states',text=df2['State'],z=df2['Voting-Age Population (VAP)'],colorbar={'title':'Voting Age Population'})
layout2=dict(title='Voting Age Population by State',geo={'scope':'usa'})

choromap2 = go.Figure(data=[data2],layout=layout2)
iplot(choromap2,validate=False)