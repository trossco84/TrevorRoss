%pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd 
%pip install lxml
import lxml
plt.show

titanic = pd.read_html('https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv')
sns.set_style('whitegrid')
titanic = sns.load_dataset('titanic')
titanic.head()

sns.jointplot(x='fare',y='age',data=titanic)

sns.distplot(titanic['fare'],kde=False,bins=30)

sns.boxplot(x='class',y='age',data=titanic)

sns.swarmplot(x='class',y='age',data= titanic)

sns.countplot(x='sex',data=titanic)

tc = titanic.corr()

sns.heatmap(tc,annot=False,cmap='coolwarm')

g = sns.FacetGrid(data=titanic,col='sex',row='age')

tm = titanic[titanic['sex']=='male']
tf = titanic[titanic['sex']=='female']
fig,ax = plt.subplots(nrows=1,ncols=2)
ax[1].plot(sns.distplot(tm['age'],kde=False))

sns.pairplot(titanic,vars=['age','sex'],x_vars='age')
tt=titanic[['sex','age']]

l = sns.FacetGrid(data=titanic,col='sex')
l.map(sns.distplot,titanic['age'])