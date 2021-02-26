import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

final_features_normalized = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/final_features/final_features_normalized.csv')
final_features_standardized = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/final_features/final_features_standardized.csv')


ffn2 = final_features_normalized.copy()
ffs2 = final_features_standardized.copy()

ffn2.set_index('customer_id',inplace=True)
ffs2.set_index('customer_id',inplace=True)

def variance(x):
    return(ffn2[x].std())

def highlynull(x):
    return(ffn2[x].isnull().sum()/len(ffn2[x]))

def blankrecord(x):
    return(ffn3.iloc[x].isnull().sum()/len(ffn3.iloc[x]))

ffn3.shape
ffn3 = ffn2.astype('float64')
ffn3.fillna(float(0),inplace=True)

ffs3 = ffs2.astype('float64')
ffs3.fillna(float(0),inplace=True)

checkhighlynull = [x for x in ffn2.columns if highlynull(x)>0.0]
checklowvariance = [x for x in ffn2.columns if variance(x)<0.05]
checkblankrecords = [x for x in range(0,len(ffn3)) if blankrecord(x) > 0.5]


from sklearn.decomposition import PCA
pca = PCA(n_components=50)
principalComponents = pca.fit_transform(ffs3)

pca_names = [f'pc{x}' for x in range(0,50)]
principalDf = pd.DataFrame(data = principalComponents,columns=pca_names)

pdf = principalDf.copy()

plt.plot(pca.explained_variance_ratio_.cumsum())

pdf2 = principalDf[principalDf.columns[0:5]]

ffn3.head()
from sklearn.ensemble import IsolationForest

isof = IsolationForest(max_samples=12000,n_estimators=205,bootstrap=True)
isof.fit(ffn3)

import pickle
model_name = '/Users/tross/Desktop/Analytics/FRAUD/Final/final_isoforest_model.sav'

pickle.dump(isof, open(model_name, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)



customers = list(ffn3.index.values)
ffn3.loc[32084].to_clipboard()
ffn3.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Final/annoying.csv')
ffn3.drop(['anomaly'],inplace=True,axis=1)
isoforestdf = pd.DataFrame(customers,columns=['customer_id'])
isoforestdf['scores'] = isof.decision_function(ffn3)
isoforestdf['anomaly'] = isof.predict(ffn3)

isoforestdf[isoforestdf.anomaly==-1].head()
isoforestdf.anomaly.value_counts()

anomaly_ids = [isoforestdf.iloc[x].customer_id for x in isoforestdf.index if isoforestdf.iloc[x].anomaly == -1]

anomalydf = ffn3.loc[anomaly_ids]

ffn3['anomaly'] = list(isoforestdf.anomaly)

pdf2['anomaly'] = isoforestdf.anomaly

isof.decision_function(pdf2)

import seaborn as sns
sns.pairplot(pdf2,hue='anomaly',palette='seismic',diag_kind='hist')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import auc

rfor = RandomForestClassifier()

X = ffn3.drop(['anomaly'],axis=1)
y = ffn3['anomaly']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

rf = RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train,y_train)

predictions = rf.predict(X_test)
pred_probs = rf.predict_proba(X_test)

confusion_matrix(y_test,predictions)
roc_value = roc_auc_score(predictions,y_test)
roc_curve(y_test,predictions)

preds = pred_probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

import imblearn

oversampling = imblearn.over_sampling

randovrsamp =  oversampling.RandomOverSampler(sampling_strategy = 'minority')

X_res,y_res = randovrsamp.fit_resample(X,y)

X_res.shape


#re-running random forest
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.25)
rf = RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train,y_train)

predictions = rf.predict(X_test)
pred_probs = rf.predict_proba(X_test)
y_res.value_counts()
confusion_matrix(y_test,predictions)
roc_value = roc_auc_score(predictions,y_test)
roc_curve(y_test,predictions)
roc_value
preds = pred_probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

feature_list = list(X_res.columns)
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

features_sorted = [x[0] for x in feature_importances]
importances_sorted = [x[1] for x in feature_importances]

fidf = pd.DataFrame(features_sorted,columns=['feature'])
fidf['importance'] = importances_sorted
fidf.to_clipboard()
fidf[0:25]
#looking at the distributions of the top features and how the anomalies are located within those distributions
import seaborn as sns
import matplotlib.pyplot as plt

#CreditReport.ID95_V1.IDSCORE named IDA ID Score 95
ffn3['CreditReport.ID95_V1.IDSCORE'].plot(kind='kde',title='IDA ID Score 95')
sns.displot(ffn3,x="CreditReport.ID95_V1.IDSCORE",hue='anomaly',palette='Pastel1',stat='probability',common_norm=False)
sns.displot(ffn3,x="CreditReport.ID95_V1.IDSCORE",hue='anomaly',palette='Pastel1',kind='kde',common_norm=False)

#reformatting dataframes for visualization purposes
visdf = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/process/numerical_features.csv')
visdf = visdf[~visdf['customer_id'].isin(['test_decision_application'])]
visdf.set_index('customer_id',inplace=True)
visdf_corr = visdf.corr().abs()
upper_tri = visdf_corr.where(np.triu(np.ones(visdf_corr.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.925)]
vdf_r = visdf.drop(to_drop,axis=1)
visanom = ['0' if x == 1 else '1' for x in isoforestdf.anomaly]
vdf_r['anomaly'] = visanom
top25 = [x for x in fidf.iloc[0:25]['feature']]
top25.append('anomaly')
cat_feats = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/process/cleaned_categorical_features.csv')
cat_feats.set_index('customer_id',inplace=True)
cf2 = cat_feats.copy()
cf2 = pd.get_dummies(cat_feats,drop_first=True)
cf2_corr = cf2.corr().abs()
upper_tri3 = cf2_corr.where(np.triu(np.ones(cf2_corr.shape),k=1).astype(np.bool))
to_drop3 = [column for column in upper_tri3.columns if any(upper_tri3[column] > 0.925)]
cf2_reduced = cf2.drop(to_drop3,axis=1)
bin_feats = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/process/cleaned_binary_features.csv')
bin_feats.set_index('customer_id',inplace=True)
joindfs = [cf2_reduced,bin_feats]
vdf2 = vdf_r.join(joindfs)
vdf3 = pd.DataFrame(vdf2[top25])
top25.append('CreditReport.CMPLY_V1.ITEM_1_GRADE')
top25.drop('CreditReport.CMPLY_V1.ITEM_1_GRADE_F'
vdf3.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/final_features/top_25.csv')
vdf3 = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/final_features/top_25.csv')
vdf2.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Final/processed_data.csv')
vdf2 = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Final/processed_data.csv')
#now checking features
def display_statistics(feature):
    indexvals=['Mean','St. Dev.', 'Range']
    vdf3_a = vdf3[vdf3.anomaly=='1']
    vdf3_n = vdf3[vdf3.anomaly=='0']
    disp = pd.DataFrame(indexvals,columns=['Statistics'])
    disp['Total Population'] = list((vdf3[feature].mean(),vdf3[feature].std(),f'[{vdf3[feature].min()},{vdf3[feature].max()}]'))
    disp['Anomalies'] = list((vdf3_a[feature].mean(),vdf3_a[feature].std(),f'[{vdf3_a[feature].min()},{vdf3_a[feature].max()}]'))
    disp['Non-Anomalies'] = list((vdf3_n[feature].mean(),vdf3_n[feature].std(),f'[{vdf3_n[feature].min()},{vdf3_n[feature].max()}]'))
    disp.set_index('Statistics',inplace=True)
    disp.to_clipboard()
    return disp.head()

#CreditReport.ID95_V1.IDSCORE, IDA ID Score 95
sns.displot(vdf3,x="CreditReport.ID95_V1.IDSCORE",hue='anomaly',palette='RdBu_r',stat='probability',common_norm=False,legend=False)
plt.xlabel('IDA ID Score 95')
sns.displot(vdf3,x="CreditReport.ID95_V1.IDSCORE",hue='anomaly',palette='RdBu_r',kind='kde',common_norm=False,fill=True,cut=0)
plt.xlabel('IDA ID Score 95')
display_statistics('CreditReport.ID95_V1.IDSCORE')
vdf3['CreditReport.ID95_V1.IDSCORE'].quantile(.9974)
542 + 2*130

#CreditReport.IDN_V1.ITEMAA3_ALLATTS_NAMESSNDOBCONFIRMED, IDA Name SSN DOB Confirmed
sns.catplot(x='CreditReport.IDN_V1.ITEMAA3_ALLATTS_NAMESSNDOBCONFIRMED',kind='count',data=vdf3,hue='anomaly')
#proportion
def anomaly_binary_distribution(feature,name):
    nsdconf = ['Confirmed' if x == 1 else "Not Confirmed" if x==0 else 'N/A' for x in vdf3['CreditReport.IDN_V1.ITEMAA3_ALLATTS_NAMESSNDOBCONFIRMED']]
    newdf1 = pd.DataFrame()
    newdf1[f'{feature}'] = nsdconf
    newdf1.set_index(vdf3.index,inplace=True)
    newdf1['anomaly'] = vdf3['anomaly']
    sns.catplot(x=f'{feature}',data=newdf1[newdf1.anomaly=='1'],hue='anomaly',kind='count',palette='Pastel1',legend=False)
    plt.title(name)
    plt.xlabel('Anomalies per Category')
    plt.ylabel('Total')
    
    
#newdf1[newdf1.anomaly=='1']['IDA Name SSN DOB Confirmed'].value_counts()

#CreditReport.IDN_V1.ITEMAA3_ALLATTS_NAMESSNDOBADDRCONFIRMED, IDA Verification 1
names = 'IDA Verification 0,IDA Verification 1,IDA Verification 2,IDA Verification 3,IDA Verification 4,IDA Verification 5,IDA Verification 6,IDA Verification 7,IDA Verification 8,IDA Verification 9, IDA Verification 10,IDA Verification 11'.split(',')
actuals = 'CreditReport.IDN_V1.ITEMAA3_ALLATTS_NAMESSNDOBCONFIRMED,CreditReport.IDN_V1.ITEMAA3_ALLATTS_NAMESSNDOBADDRCONFIRMED,CreditReport.IDN_V1.ITEMAA3_ALLATTS_NAMESSNDOB6ADDRCONFIRMED,CreditReport.IDN_V1.ITEMAA3_ALLATTS_NAMESSNDOB6CONFIRMED,CreditReport.IDN_V1.ITEMAA3_ALLATTS_LASTNAMESSNDOBADDRCONFIRMED,CreditReport.IDN_V1.ITEMAA3_ALLATTS_LASTNAMESSNDOBCONFIRMED,CreditReport.IDN_V1.ITEMAA3_ALLATTS_LASTNAMESSNDOB6CONFIRMED,CreditReport.IDN_V1.ITEMAA3_ALLATTS_LASTNAMESSNDOB6ADDRCONFIRMED,CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR2YEARS,CreditReport.IDN_V1.ITEMAA3_ALLATTS_NAMESSNDOBPRIMPHONECONFIRMED,CreditReport.IDN_V1.ITEMAA3_ALLATTS_CONSISTENCYSSNDOB,CreditReport.IDN_V1.ITEMAA3_ALLATTS_NAMESSNDOB6PRIMPHONECONFIRMED'.split(',')
fundict1 = {actuals[i]:names[i] for i in range(0,len(names))}
#plotting distributions for each variable
for item in fundict1.items():
    anomaly_binary_distribution(item[0],item[1])


#CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS, IDA Applications in last 3 Months by Address
#null values ended up being negative integers, adjusted these to be np.nan's
impute1 = [np.nan if x < 0 else x for x in vdf3['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS']]
vdf3['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS'] = impute1
vdf3[vdf3['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS'] <1]['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS'].sum()
ida3month = sns.displot(vdf3,x='CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS',hue='anomaly',palette='prism',common_norm=False,legend=False,kind='kde',cut=0,fill=True,bw_adjust=1)
plt.xlabel('IDA Applications in last 3 Months by Address')
display_statistics('CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS')

vdf3_a[vdf3_a['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS']>8].anomaly.value_counts()
#i want to know the probabilty of a record being an anomaly as a function of this variable
#as x increases, how does it's probability of being an anomaly increase?
sns.displot(vdf3_a,x='CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS',stat='probability')
testdf2 = pd.DataFrame()
xlist = []
plist=[]
for x in range(1,175):
    xlist.append(x)
    p_n = len(vdf3_n[vdf3_n['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS']==x])/len(vdf3_n)
    p_a = len(vdf3_a[vdf3_a['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS']==x])/len(vdf3_a)
    prob = (1-p_n)
    plist.append(prob)

testdf2['Number of Applications in Last 3 Months'] = xlist
testdf2['Probability of Anomaly'] = plist
testdf2.plot('Number of Applications in Last 3 Months',cmap='prism')

#CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR1YEAR
impute1 = [np.nan if x < 0 else x for x in vdf3['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR1YEAR']]
vdf3['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR1YEAR'] = impute1
vdf3[vdf3['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR1YEAR'] <1]['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR1YEAR'].sum()
ida3month = sns.displot(vdf3,x='CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR1YEAR',hue='anomaly',palette='prism',common_norm=False,legend=False,kind='kde',cut=0,fill=True,bw_adjust=1)
plt.xlabel('IDA Applications in last 1 Year by Address')
display_statistics('CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR1YEAR')

vdf3_a[vdf3_a['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR3MONTHS']>8].anomaly.value_counts()
#i want to know the probabilty of a record being an anomaly as a function of this variable
#as x increases, how does it's probability of being an anomaly increase?
sns.displot(vdf3_a,x='CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR1YEAR',stat='probability')
testdf2 = pd.DataFrame()
xlist = []
plist=[]
for x in range(1,175):
    xlist.append(x)
    p_n = len(vdf3_n[vdf3_n['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR1YEAR']==x])/len(vdf3_n)
    p_a = len(vdf3_a[vdf3_a['CreditReport.IDN_V1.ITEMAA3_ALLATTS_TIMESAPPLIEDBYADDR1YEAR']==x])/len(vdf3_a)
    prob = (1-p_n)
    plist.append(prob)

testdf2['Number of Applications in Last 1 Year'] = xlist
testdf2['Probability of Anomaly'] = plist
testdf2.plot('Number of Applications in Last 1 Year',cmap='prism')


vdf3[vdf3['CreditReport.CMPLY_V1.ITEM_1_GRADE']==1].anomaly.value_counts()
sns.countplot(vdf3['CreditReport.CMPLY_V1.ITEM_1_GRADE'],hue=vdf3.anomaly)
sns.catplot(x='CreditReport.CMPLY_V1.ITEM_1_GRADE',data=vdf3,hue='anomaly',kind='count',palette='Pastel1')


x1 = 'CreditReport.CMPLY_V1.ITEM_1_GRADE'
y1 = 'Proportion'
hue1 = 'anomaly'

(vdf3[x1]
 .groupby(vdf3[hue1])
 .value_counts(normalize=True)
 .rename(y1)
 .reset_index()
 .pipe((sns.barplot, "data"), x=x1, y=y1, hue=hue1,palette='Pastel2',order=['A','B','C','D','F','I']))
plt.xlabel('IDA Comply360 Grade')


sns.catplot(x=f'{feature}',data=newdf1[newdf1.anomaly=='1'],hue='anomaly',kind='count',palette='Pastel1',estimator=lambda x: sum(x==0)*100.0/len(x)))

vdf3.anomaly.value_counts()


#CreditReport.IDN_V1.ITEMAA3_ALLATTS_VELOCITYBASEDRISK, IDA Velocity Based Risk
sns.displot(vdf3,x='CreditReport.IDN_V1.ITEMAA3_ALLATTS_VELOCITYBASEDRISK',stat='probability',hue='anomaly',common_norm=False,palette = 'seismic')
sns.displot(vdf3,x='CreditReport.IDN_V1.ITEMAA3_ALLATTS_VELOCITYBASEDRISK',hue='a_score',palette='seismic',kind='kde',common_norm=False,fill=False,cut=0)
sns.distplot(vdf3['a_score'])

plt.xlabel('IDA Velocity Based Risk')
vdf3['CreditReport.IDN_V1.ITEMAA3_ALLATTS_VELOCITYBASEDRISK'].quantile(0.)
vdf3['a_score'] = isoforestdf['scores']

sns.displot(vdf3,x='CreditReport.ID95_V1.IDSCORE',hue='CreditReport.CMPLY_V1.ITEM_1_GRADE',palette='seismic',kind='kde',common_norm=False,fill=False,cut=0)
sns.catplot(vdf3,x='CreditReport.ID95_V1.IDSCORE',y='CreditReport.CMPLY_V1.ITEM_1_GRADE',palette='seismic')


vdf3['CreditReport.IDN_V1.ITEMAA3_ALLATTS_VELOCITYBASEDRISK'].describe()
vdf3.columns
fdf1 = vdf3.copy()
final_features = ['CreditReport.ID95_V1.IDSCORE','CreditReport.CMPLY_V1.ITEM_1_GRADE','CreditReport.IDN_V1.ITEMAA3_ALLATTS_VELOCITYBASEDRISK']
fdf2=fdf1[final_features]
fdf1.head()
fdf1['CreditReport.CMPLY_V1.ITEM_1_GRADE'][0:5]
fdf2.rename(columns={'CreditReport.ID95_V1.IDSCORE':'IDA ID Score 95','CreditReport.CMPLY_V1.ITEM_1_GRADE':'IDA Comply360 Grade','CreditReport.IDN_V1.ITEMAA3_ALLATTS_VELOCITYBASEDRISK':'IDA Velocity Based Risk'},inplace=True)

#create a score for each feature and combo features, and a final score that includes isoforest
#times applied composite
timesapplied = 'TIMESAPPLIED'
tryit = [x for x in vdf2.columns if timesapplied in x]
tryit

year2 = '2YEARS'
tryit2 = [x for x in tryit if year2 in x]
tad = pd.DataFrame(vdf2[tryit2])
tad.head()

tad3.composite.quantile(.9545)


tad2 = tad.copy()
mask2 = tad2 < 0
tad3 = tad2.mask(tad2<0,0)
tad3['composite'] = tad3.iloc[:].sum(axis=1)
tad3['anomaly']=vdf2.anomaly



tad3.composite.quantile(.9545)
tad3.composite.mean()+2*np.sqrt(tad3.composite.mean())
len(tad3)
len(tad3[tad3.composite>210])/len(tad3)

sns.displot(tad3,x='composite',kind='kde',palette = 'seismic',fill=True)

plt.xlabel('Times Applied Total')

tad3['Times Applied Composite Score'] = [(x/750) if x<750 else 1 for x in tad3.composite]

sns.displot(tad3,x='Times Applied Composite Score',kind='kde',hue='anomaly',common_norm=False,palette = 'seismic',fill=True,cut=0)
fdf2['IDA Times Applied Composite Score'] = tad3['Times Applied Composite Score']

#confirmed pii
conf = 'CONFIRMED'
confcols = [x for x in vdf2.columns if conf in x]
len(confcols)

confdf = pd.DataFrame(vdf2[confcols])
confdf[confdf.columns[]].describe()
confdf2 = confdf.mask(confdf.isnull(),0)
confdf2['composite'] = confdf2.iloc[:].sum(axis=1)
confdf2['anomaly'] = vdf2.anomaly

confdf2.composite.quantile(0.9545)
import seaborn as sns
sns.displot(confdf2,x='composite',kind='kde',hue='anomaly',common_norm=False,palette = 'seismic',fill=True,cut=0)
confdf2['Confirmed PII Composite Score'] = [(x/35) for x in confdf2.composite]

sns.displot(confdf2,x='Confirmed PII Composite Score',kind='kde',hue='anomaly',common_norm=False,palette = 'seismic',fill=True,cut=0)
fdf2['IDA Confirmed PII Composite Score'] = confdf2['Confirmed PII Composite Score']

#Times Fraud Reported
fraudstring = 'FRAUD'
frcols = [x for x in vdf2.columns if fraudstring in x]

frdf = pd.DataFrame(vdf2[frcols])
frdf2 = frdf.copy()
frdf2.drop(['CreditReport.EM_V1.ITEM_FRAUDRISK.CATEGORY'],inplace=True,axis=1)
frdf2['composite'] = frdf2.iloc[:].sum(axis=1)
frdf2['anomaly'] = vdf2.anomaly

frdf2.composite.quantile(.95)
sns.displot(frdf2,x='composite',kind='kde',hue='anomaly',common_norm=False,palette = 'seismic',fill=True,cut=0)
plt.xlim(0,20)
plt.ylim(0,1)
plt.xticks(range(0,21))

frcat = []
for x in frdf2.composite:
    if x == 0:
        frcat.append("None")
    elif 0<x<2:
        frcat.append("Low")
    elif 2<x<5:
        frcat.append("Moderate")
    else:
        frcat.append("High")

frdf2['fraud_activity'] = frcat
#frdf2.plot('fraud_activity',kind='bar')

frdf2['fraud_activity']
sns.countplot(frdf2['fraud_activity'],hue=frdf2['anomaly'])
sns.catplot(x='fraud_activity',y='composite',data = frdf2,hue='anomaly',kind='box',dodge=False)
plt.ylim(0,30)

frdf2['fraud_activity'].value_counts().plot(kind='bar',hue='anomaly')
frdf2[frdf2.anomaly=='0']['fraud_activity'].value_counts().plot(kind='bar')
sns.catplot(x='fraud_activity',data=frdf2,hue='anomaly',kind='count')
plt.ylim(0,200)

x1 = 'fraud_activity'
y1 = 'Density'
hue1 = 'anomaly'

(frdf2[x1]
 .groupby(frdf2[hue1])
 .value_counts(normalize=True)
 .rename(y1)
 .reset_index()
 .pipe((sns.barplot, "data"), x=x1, y=y1, hue=hue1,palette='Pastel1',order=['None','Low','Moderate','High']))
plt.xlabel('Reported Fraud Activity')


fdf2['IDA Reported Fraud Activity'] = frdf2['fraud_activity']

#recent fraud binary
recent = '6MONTHS'
recentfraud = [x for x in frcols if recent in x ]
len(recentfraud)
frdf3 = pd.DataFrame(vdf2[recentfraud])

frecent = frdf3.copy()
frecent.head()
frecent['composite'] = frecent.iloc[:].sum(axis=1)
frecent['anomaly'] = vdf2.anomaly

sns.displot(frecent,x='composite',kind='kde',hue='anomaly',common_norm=False,palette = 'seismic',fill=True,cut=0)
plt.ylim(0,2)
plt.xlim(0,4)

frecent['Recently Reported Fraud'] = [1 if x>0 else 0 for x in frecent.composite]

fdf2['IDA Recently Reported Fraud'] = frecent['Recently Reported Fraud']

fdf2.head()

fdf2.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Final/decision_features.csv')

final = fdf2.copy()
final.head()
#create decision rules and a final composite score


list_of_palettes = ['Accent', 'Accent_r', 'Blues',\
     'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
     'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r','Dark2', 
     'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 
     'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 
     'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 
     'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 
     'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 
     'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 
     'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 
     'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 
     'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 
     'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 
     'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 
     'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 
     'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 
     'afmhot_r', 'autumn', 'autumn_r', 'binary', 
     'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 
     'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 
     'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 
     'copper_r', 'crest', 'crest_r', 'cubehelix', 
     'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 
     'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 
     'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 
     'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 
     'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 
     'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 
     'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 
     'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 
     'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 
     'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 
     'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 
     'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 
     'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 
     'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 
     'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 
     'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']