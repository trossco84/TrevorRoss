import pandas as pd
import numpy as np
sal = pd.read_csv('Salaries-Copy1.csv')
sal.head()
sal.info()
sal['BasePay'].mean()
sal['OvertimePay'].max()
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['JobTitle']
sal[sal['EmployeeName']=='JOSEPH DRISCOLL']['TotalPayBenefits']

sal[sal['TotalPayBenefits']==sal['TotalPayBenefits'].max()]['EmployeeName']
sal.loc[sal['TotalPayBenefits'].idxmax()]
sal[sal['TotalPayBenefits']==sal['TotalPayBenefits'].min()]['EmployeeName']

sal.groupby('Year').mean()['BasePay']

sal['JobTitle'].nunique()

sal['JobTitle'].value_counts().nlargest(5)
sal['JobTitle'].value_counts().head(5)

sum(sal[sal['Year']==2013]['JobTitle'].value_counts()==1)

sal['JobTitle']
def cstring(title):
    if 'chief' in title.lower().split():
        return True
    else:
        return False

sum(sal['JobTitle'].apply(lambda x: cstring(x)))

sal[sal['JobTitle'].str.contains('Chief', case=False)]['EmployeeName'].nunique()

sal['title_len'] = sal['JobTitle'].apply(len)
sal[['title_len','TotalPayBenefits']].corr()


##Ecommerce
ecom = pd.read_csv('Ecommerce Purchases-Copy1')
ecom.head()
len(ecom)
ecom.info()

ecom['Purchase Price'].mean()
ecom['Purchase Price'].max()
ecom['Purchase Price'].min()

len(ecom[ecom['Language']=='en'])

len(ecom[ecom['Job']=='Lawyer'])

ecom['AM or PM'].value_counts()

ecom['Job'].value_counts().head(5)

ecom[ecom['Lot']=='90 WT']['Purchase Price']

ecom[ecom['Credit Card']==4926535242672853]['Email']

len(ecom[(ecom['CC Provider']=='American Express') & (ecom['Purchase Price']>95)])

ecom.head()

ecom['yrexp'] = ecom['CC Exp Date'.rjust(2)]

len(ecom[ecom['CC Exp Date'].str.contains('25')])

sum(ecom['CC Exp Date'].apply(lambda exp: exp[3:]=='25'))
ecom[ecom['CC Exp Date'].apply(lambda exp: exp[3:]=='25')].count()

ecom['emailstr'] = str(ecom['Email'])

ecom['Email']

ecom['Email'].apply(lambda email: email.split('@')[1]).value_counts().head(5)