#Read and format the data
####import libraries
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

####client data input and transformation
rifco = pd.read_csv('/Users/tross/Desktop/Clients/Potential/Rifco/RifcoDataOnly.csv')
rifco.head()

#stripping client's extra fields that won't be used
rifco_imp = rifco.drop(['UUID','p_new','good','new_score','y','P_1'],axis=1)
rifco_imp.head()

#seeing the # of target variable records
rifco_imp['bad1'].value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#small percentage of bad loans, to increase these we will oversample the data set, but first we must split the data
X = rifco_imp.drop(['bad1'],axis=1)
y = rifco_imp['bad1']


#creating variables only dataset

#it seems like there's a lot of missing values
# number of null values per variable
rifco_imp.isnull().sum()
# total number of null values
tot_null = rifco_imp.isnull().sum().sum()
# percentage of null values
perc_null = pertot_null/(rifco_imp.shape[0]*rifco_imp.shape[1])
rifco_imp.shape[1]

rifco_onlyvars.drop(['missingsum'],axis=1,inplace=True)
rifcosmall = rifco_onlyvars.dropna(thresh=18)
len(rifcosmall)
len(rifcosmall.columns)
rifcosmallimputed = rifcosmall
rifcosmall.head()
rifcosmallimputed = rifcosmallimputed.apply(lambda x: x.fillna(x.median()),axis=0)

rifcovarsimputed = rifco_onlyvars.drop(['bad1'],axis=1).apply(lambda x: x.fillna(x.median()),axis=0)

rifcosmallimputed.head()
for i in range(0,31):
    rifcosmallimputed[i].fill


rifcosmall.to_csv('rifcolownull',index=False)
####data exploration
sns.pairplot(rifco_onlyvars)

##Variable by Variable
#A2MH017 - Minimum pay rate on revolving trades in the past 6 months
a2mh017 = rifco['A2MH017']
a2mh017

#TCTE023
tcte023 = rifco['TCTE023']
sum(tcte023.isnull())
#imputing data with mean value of the range

#Logistic Regression
from sklearn.linear_model import LogisticRegression
X = rifcosmallimputed.drop(['bad1'],axis=1)
y = rifcosmallimputed['bad1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
logmod = LogisticRegression()
logmod.fit(X_train,y_train)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report
predictions = logmod.predict(X_test)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions,))
sns.distplot((X_test-predictions))

#Standardize the Variables
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(rifcosmallimputed.drop('bad1',axis=1))
scaled_features = scaler.transform((rifcosmallimputed.drop('bad1',axis=1)))
rifcofeat = pd.DataFrame(scaled_features,columns = rifcosmallimputed.columns[:-1])
rifcofeat.head()

rifcobad = rifco[['bad1','new_score']]
rifcobad.head()

rbadscores = rifcobad[rifcobad['bad1']==1,]

sns.barplot(y=pd.qcut(rifcobad['new_score'],10,labels=False),x='bad1',data=rifcobad)

decile = pd.qcut(rifcobad['new_score'],10,labels=False)
pd.qc
pd.qcut(investment_df['investment'], 10, labels=True)
decile

rifcobad['decile'] = decile

countd1 = rifcobad[(rifcobad['bad1']==1) & (rifcobad['decile']==1)].sum()
countd1

sns.countplot(x=decile,data=rifcoterrible)

sns.barplot(x = rifcoterrible['decile'].value_counts())

rifcoterrible = rifcobad[rifcobad['bad1']==1]
rifcoterrible.head()
sns.pairplot(rifcoterrible)

sns.barplot(x=rifcoterrible['decile'],y='bad1',data=rifcoterrible)
sns.countplot(x='decile',data=rifcoterrible,palette='seismic')
rifcoterrible['decile'].value_counts().plot(kind='bar',c='green')
sns.distplot(rifcoterrible['decile'].value_counts(),kde=False,bins=10,color='green',ax)
# ADDED: Extract axes.
fig, ax = plt.subplots(1, 1, figsize = (12, 8), dpi=300)
sns.countplot(x='decile',data=rifcoterrible,palette='coolwarm',saturation=.98,)
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')

rifcopca = rifco_onlyvars.drop(['bad1'],axis=1)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(rifcopca)

rifcopca = scaler.transform(rifcopca)
rifcopca = pd.DataFrame(rifcopca,columns = rifco_onlyvars.columns[:-1])
from sklearn.decomposition import PCA

len(rifcopca.columns)
ronehot = pd.read_csv('/Users/tross/Desktop/Clients/Potential/Rifco/rifcoonehot.csv')
ronehot.head()
rimputed = ronehot.apply(lambda x: x.fillna(-1,axis=0))
for i in range(0,30):
len(rimputed.columns)
rifcopca.to_clipboard()
rifcosmallimputed = rifcosmallimputed.apply(lambda x: x.fillna(x.median()),axis=0)

pca = PCA(n_components=3)
rimputed.
len(rifcoscaled)
pca.fit(rimputed)
newarray = pca.transform(rimputed)
len(newarray)
zeet = pd.DataFrame(newarray)
zeet.head()

combined = pd.concat([rimputed,zeet],axis=1)
combined['target'] = rifco_onlyvars['bad1'][0:30001]
rimputed.head()


combined = pd.read_csv('/Users/tross/Desktop/Clients/Potential/Rifco/rifcocombined.csv')
rimputed.drop(rimputed.tail(1).index,inplace=True)
df_tryagain = pd.DataFrame(pca.components_)
df_tryagain.head(12)
df_tryagain['target']=rifco_onlyvars['bad1']
len(df_tryagain)



X = combined.drop(['target'],axis=1)
y = combined['target']

combined = pd.read_csv('')
combined.to_csv('rifcocombined.csv',index=False)
rtest = rifco_onlyvars.apply(lambda x: x.fillna(x.mean()),axis=0)

X = rtest.drop(['bad1'],axis=1)
y = rtest['bad1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
logmod = LogisticRegression()
logmod.fit(X_train,y_train)
from sklearn.metrics import classification_report
from sklearn.metrics import auc
from sklearn import metrics
fpr,tpr,thresholds = metrics.roc_curve(y_test,preds_proba)
roc_auc = metrics.auc(fpr,tpr)
roc_auc.plot()
#import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#lookup cell blocks

from sklearn.metrics import confusion_matrix, classification_report
predictions = logmod.predict_proba(X_test)
preds = logmod.predict(X_test)
preds_proba = predictions[:,1]
print(confusion_matrix(y_test,preds)) 
print(classification_report(y_test,preds,))

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,preds)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train,y_train)

rfcpred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfcpred))
print(classification_report(y_test,rfcpred))

print(pca.explained_variance_)
print(pca.)


pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

np.cumsum(pca.explained_variance_ratio_[0:30])




from sklearn.model_selection import K










sns.set_style('white')
fig,ax = plt.figure()

sns.(data=finalrifco['Ensemble'],data2=rifco['p_new'])
import matplotlib.patches as mpatches
gpatch = mpatches.Patch(color='green',label='Bad Loans')
bpatch = mpatches.Patch(color='blue',label='All Loans')
sns.distplot(finalrifco['Ensemble'],color='blue',axlabel='Probability of Default')
sns.distplot(finalrifco[finalrifco['bad1']==1]['Ensemble'],color='green',axlabel='Probability of Default')
sns.despine()
plt.xlim(0,1)
plt.ylim(0,8)
plt.legend(handles=[bpatch,gpatch])

sns.distplot(rifco['p_new'],color='blue',axlabel='Probability of Default',bins=60)
sns.distplot(rifco[finalrifco['bad1']==1]['p_new'],color='green',axlabel='Probability of Default',bins=60)
sns.despine()
plt.xlim(0,1)
plt.ylim(0,)
plt.legend(handles=[bpatch,gpatch])

sns.lmplot(x='bad1',y='Ensemble',data=finalrifco)


sns.distplot(finalrifco['bad1'],color='green')

sns.distplot(rifco['p_new'],color='orange')





finalrifco = pd.read_csv('/Users/tross/Desktop/Clients/Potential/Rifco/prediction_final.csv',index_col='UUID')

finalrifco['Good'] = 1 - finalrifco['Ensemble']

finalrifco['score'] = (1-finalrifco['Ensemble'])*900
finalrifco['score'] = finalrifco['score'].apply(lambda x: int(x))
finalrifco.head()

easydatadf = pd.DataFrame()
easydatadf['oldtot'] = oldtotals
oldtotals = [bucket1,bucket2,bucket3,bucket4,bucket5,bucket6] 
bucket1 = sum(rifco['new_score']<=650)
bucket2 = sum(rifco['new_score']<=700) - bucket1
bucket3 = sum(rifco['new_score']<=750) - bucket1 - bucket2
bucket4 = sum(rifco['new_score']<=800) - bucket1 - bucket2 - bucket3
bucket5 = sum(rifco['new_score']<=850) - bucket1 - bucket2 - bucket3 - bucket4
bucket6 = sum(rifco['new_score']<=900) - bucket1 - bucket2 - bucket3 - bucket4 - bucket5
easydatadf

sns.countplot(easydatadf['oldbadtot'])

orb = rifco[rifco['bad1']==1]
nrb = finalrifco[finalrifco['bad1']==1]

easydatadf.drop('Bucket1',axis=1,inplace=True)

buck1 = sum(orb['new_score']<=650)
buck2 = sum(orb['new_score']<=700) - buck1
buck3 = sum(orb['new_score']<=750) - buck1 - buck2
buck4 = sum(orb['new_score']<=800) - buck1 - buck2 - buck3
buck5 = sum(orb['new_score']<=850) - buck1 - buck2 - buck3 - buck4
buck6 = sum(orb['new_score']<=900) - buck1 - buck2 - buck3 - buck4 - buck5
ob = [buck1,buck2,buck3,buck4,buck5,buck6]
nb = [ob1,ob2,ob3,ob4,ob5,ob6]

easydatadf['oldbadtot'] = ob
easydatadf['newbadtot'] = nb

b1 = sum(finalrifco['score']<=650)
b2 = sum(finalrifco['score']<=700) - b1
b3 = sum(finalrifco['score']<=750) - b1 - b2
b4 = sum(finalrifco['score']<=800) - b1 - b2 - b3
b5 = sum(finalrifco['score']<=850) - b1 - b2 - b3 - b4
b6 = sum(finalrifco['score']<=900) - b1 - b2 - b3 - b4 - b5
newtotals = [b1,b2,b3,b4,b5,b6]
easydatadf['newtot'] = newtotals


ob1 = sum(nrb['score']<=650)
ob2 = sum(nrb['score']<=700) - ob1
ob3 = sum(nrb['score']<=750) - ob1 - ob2
ob4 = sum(nrb['score']<=800) - ob1 - ob2 - ob3
ob5 = sum(nrb['score']<=850) - ob1 - ob2 - ob3 - ob4
ob6 = sum(nrb['score']<=900) - ob1 - ob2 - ob3 - ob4 - ob5

fig, ax = plt.subplots(1, 1, figsize = (12, 8), dpi=300)
sns.countplot(x='oldbadtot',data=easydatadf,palette='coolwarm',saturation=.98,)
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')

easydatadf.to_clipboard()