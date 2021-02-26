import pandas as pd 
import numpy as np

csc_fraud = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Data/csc_cleaned_fraud_features.csv')

binary = []
cat = []
num = []
unsure=[]
dropcols = []

for colu in csc_fraud.columns:
    if csc_fraud[colu].dtype == 'float64':
        num.append(colu)
    elif csc_fraud[colu].dtype == 'bool':
        binary.append(colu)
    elif csc_fraud[colu].dtype == 'str':
        cat.append(colu)
    else:
        unsure.append(colu)

for colu in unsure:
    print()
    print(colu)
    print(csc_fraud[colu].describe())
    if input('want to see the first few?') == 'yes':
        print(csc_fraud[colu][0:8])
        placement = input('where should this go?')
    else:
        placement = input('where should this go?')
    if placement == 'binary':
        binary.append(colu)
    elif placement == 'cat':
        cat.append(colu)
    elif placement == 'num':
        num.append(colu)
    elif placement == 'drop':
        dropcols.append(colu)
    else:
        unsure.append(colu)

more_drops = ['CreditReport.CMPLY_V1.ITEM_1_RESULTCODE2','CreditReport.CMPLY_V1.ITEM_1_RESULTCODE3','CreditReport.CMPLY_V1.ITEM_1_RESULTCODE4','CreditReport.CMPLY_V1.ITEM_1_CONTRARYIDENTITY_SSN','CreditReport.CMPLY_V1.ITEM_1_CONTRARYIDENTITY_ZIP','CreditReport.CMPLY_V1.ITEM_1_CONTRARYIDENTITY_PHONE','CreditReport.ID95_V1.IDSCORERESULTCODE2','CreditReport.ID95_V1.IDSCORERESULTCODE3','CreditReport.EM_V1.ITEM_EAADVICEID','CreditReport.EM_V1.ITEM_EAREASONID','CreditReport.EM_V1.ITEM_EASTATUSID','CreditReport.EM_V1.ITEM_EARISKBANDID','CreditReport.EM_V1.ITEM_DOMAINRISKLEVELID','CreditReport.IDN_V1.ITEMAA3_ALLATTS_LIKELYAGEATSSNISSUANCE']

num = [x for x in num if x not in more_drops]
dropcols.pop(-1)
for i in more_drops:
    dropcols.append(i)

binary_features = csc_fraud[binary]
binary_features['customer_id'] = csc_fraud['customer_id']
binary_features.set_index('customer_id',inplace=True)
binary_features.reset_index(inplace=True)

categorical_features = csc_fraud[cat]
categorical_features['customer_id'] = csc_fraud['customer_id']
categorical_features.set_index('customer_id',inplace=True)
categorical_features.reset_index(inplace=True)

numerical_features = csc_fraud[num]
numerical_features['customer_id'] = csc_fraud['customer_id']
numerical_features.set_index('customer_id',inplace=True)
numerical_features.reset_index(inplace=True)

drop_features = csc_fraud[dropcols]
drop_features['customer_id'] = csc_fraud['customer_id']
drop_features.set_index('customer_id',inplace=True)
drop_features.reset_index(inplace=True)

binary_features.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/binary_features.csv',index=False)
categorical_features.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/categorical_features.csv',index=False)
numerical_features.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/numerical_features.csv',index=False)
drop_features.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/bad_features.csv',index=False)

#spaced out all of the variables into category (needs work, some have a lot), numeric (needs to be normalized), and binary (needs to be adjusted to 1's and 0's)

#Converting all binary variables to 1's and 0's, see replacements for reference
bdf = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/process/binary_features.csv')
bdf.head()
bdf.shape
b_other = csc_fraud.select_dtypes(include='bool')

what_else = [x for x in bdf.columns if x not in b_other.columns]
what_else
bdf2 = bdf.copy()

bdf2.replace(False,float(0),inplace=True)
bdf2.replace(True,float(1),inplace=True)
bdf2.replace('Green',float(1),inplace=True)
bdf2.replace('Red',float(0),inplace=True)
bdf2.replace('Yes',float(1),inplace=True)
bdf2.replace('No',float(0),inplace=True)
bdf2.replace('N',float(0),inplace=True)
bdf2.replace('NN',float(1),inplace=True)
bdf2.replace('M',float(1),inplace=True)
bdf2.describe()
bdf2.head()

nonbin = [x for x in bdf2.columns if bdf2[x][4] not in [float(0),float(1)]]
nonbin

bdf3 = bdf2[~bdf2['customer_id'].isin(['test_decision_application'])]
bdf3.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/cleaned_binary_features.csv',index=False)

#Next up is Numerical and Categorical
cdf = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/process/categorical_features.csv')
cdf.head()
cdf.shape

cdf2 = cdf.copy()

#keep as is
cdf2[cdf2.columns[1]].describe()
cdf2[cdf2.columns[2]].describe()
cdf2[cdf2.columns[4]].describe()
cdf2[cdf2.columns[6]].describe()
cdf2[cdf2.columns[7]].describe()
cdf2[cdf2.columns[10]].describe()

#adjustments
#adjust countries that only have one value count as other
cdf2[cdf2.columns[3]].describe()
cdf2[cdf2.columns[3]].value_counts()
keep_countries = 'US CA CH FR DE ES GB'.split()
country_codes = [x if x in keep_countries else "other" for x in cdf2['CreditReport.EM_V1.ITEM_COUNTRY']]
cdf2['CreditReport.EM_V1.ITEM_COUNTRY'] = country_codes

#adjust fraud buckets to better values
cdf2[cdf2.columns[5]].value_counts()
cdf2.replace('Fraud Score 1 to 100','FraudBucket1',inplace=True)
cdf2.replace('Fraud Score 101 to 300','FraudBucket2',inplace=True)
cdf2.replace('Fraud Score 301 to 600','FraudBucket3',inplace=True)
cdf2.replace('Fraud Score 601 to 799','FraudBucket4',inplace=True)
cdf2.replace('Fraud Score 800 to 899','FraudBucket5',inplace=True)
cdf2.replace('Fraud Score 900 to 999','FraudBucket6',inplace=True)

#dropped domain country name and item domain category
cdf2.drop(['CreditReport.EM_V1.ITEM_DOMAINCOUNTRYNAME'],axis=1,inplace=True)
cdf2.drop(['CreditReport.EM_V1.ITEM_DOMAINCATEGORY'],axis=1,inplace=True)

#created two separate columns for Fraud Risk, numeric (fraud score), and categorical (fraud category)
fraudscore = [int(x[0:3]) for x in cdf2['CreditReport.EM_V1.ITEM_FRAUDRISK']]
fraudcategory = [x[4:] for x in cdf2['CreditReport.EM_V1.ITEM_FRAUDRISK']]
cdf2.drop(['CreditReport.EM_V1.ITEM_FRAUDRISK'],axis=1,inplace=True)
cdf2['CreditReport.EM_V1.ITEM_FRAUDRISK.SCORE'] = fraudscore
cdf2['CreditReport.EM_V1.ITEM_FRAUDRISK.CATEGORY'] = fraudcategory

#not sure what to do with emails
cdf2[cdf2.columns[8]].describe()

cdf2.drop(['CreditReport.EM_V1.ITEM_FRAUDRISK.SCORE'],axis=1,inplace=True)
cdf2.drop(['CreditReport.EM_V1.ITEM_DOMAINNAME'],axis=1,inplace=True)

cdf3 = cdf2[~cdf2['customer_id'].isin(['test_decision_application'])]
cdf3.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/cleaned_categorical_features.csv',index=False)
cat7 = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/process/cleaned_categorical_features.csv')
cat7.columns
#make fraudscore numerical
fraudscore
'CreditReport.EM_V1.ITEM_FRAUDRISK.SCORE'

ndf = cdf = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/process/numerical_features.csv')
nummys = csc_fraud.select_dtypes(include='float64')
nummys.shape
ndf.head()
ndf.shape
what_else2 = [x for x in ndf.columns if x not in nummys.columns]
what_else3 = [x for x in nummys.columns if x not in ndf.columns]
what_else
ndf['CreditReport.EM_V1.ITEM_FRAUDRISK.SCORE'] = fraudscore

ndf2 = ndf.copy()
ndf3 = ndf2[~ndf2['customer_id'].isin(['test_decision_application'])]
ndf3.set_index('customer_id',inplace=True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

norm = MinMaxScaler().fit(ndf3)
norm_numerical = pd.DataFrame(norm.transform(ndf3),columns=ndf3.columns)
norm_numerical.set_index(ndf3.index,inplace=True)

standard = StandardScaler().fit_transform(ndf3)
stand_numerical = pd.DataFrame(standard,columns=ndf3.columns)
stand_numerical.set_index(ndf3.index,inplace=True)

norm_numerical.reset_index(inplace=True)
stand_numerical.reset_index(inplace=True)

norm_numerical.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/cleaned_normalized_numerical_features.csv',index=False)
stand_numerical.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Initial/features/cleaned_standardized_numerical_features.csv',index=False)