import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_apps_feature(df):
    apps_string = 'TIMESAPPLIED'
    year2_string = '2YEARS'
    apps_feats = [x for x in df.columns if apps_string in x]
    apps_feats = [x for x in apps_feats if year2_string in x]
    apps_df = pd.DataFrame(df.set_index('customer_id')[apps_feats])
    apps_df = apps_df.mask(apps_df<0,0)
    apps_totals = apps_df.iloc[:].sum(axis=1)
    return apps_scores

def create_verification_feature(df):
    verif_string = 'CONFIRMED'
    verif_cols = [x for x in df.columns if verif_string in x]
    vdf = pd.DataFrame(df[verif_cols])
    vdf = vdf.mask(vdf.isnull(),0)
    verif_totals = vdf.iloc[:].sum(axis=1)
    verif_scores = [(x/35) for x in verif_totals]
    return verif_scores

def create_reported_feature(df):
    fraud_string = 'FRAUD'
    frcols = [x for x in df.columns if fraud_string in x]
    frdf = pd.DataFrame(df[frcols])
    frdf.drop(['CreditReport.EM_V1.ITEM_FRAUDRISK'],axis=1,inplace=True)
    reported_totals = frdf.iloc[:].sum(axis=1)
    frcat = []
    for x in reported_totals:
        if x == 0:
            frcat.append("None")
        elif 0<x<2:
            frcat.append("Low")
        elif 2<x<5:
            frcat.append("Moderate")
        else:
            frcat.append("High")
    return frcat

def create_recent_feature(df):
    fraud_string = 'FRAUD'
    frcols = [x for x in df.columns if fraud_string in x]
    recent = '6MONTHS'
    recentfraud = [x for x in frcols if recent in x ]
    recentdf = pd.DataFrame(df[recentfraud])
    recent_totals = recentdf.iloc[:].sum(axis=1)
    recent_scores = [1 if x>0 else 0 for x in recent_totals]
    return recent_scores

def process_for_kickout(df):
    removals = [x for x in df.customer_id if 'test' in str(x)]
    df = df[~df.customer_id.isin(removals)]
    base_features = ['CreditReport.ID95_V1.IDSCORE','CreditReport.CMPLY_V1.ITEM_1_GRADE','CreditReport.IDN_V1.ITEMAA3_ALLATTS_VELOCITYBASEDRISK']
    df_kickout = df.set_index('customer_id')[base_features]
    df_kickout.rename(columns={'CreditReport.ID95_V1.IDSCORE':'IDA ID Score 95','CreditReport.CMPLY_V1.ITEM_1_GRADE':'IDA Comply360 Grade','CreditReport.IDN_V1.ITEMAA3_ALLATTS_VELOCITYBASEDRISK':'IDA Velocity Based Risk'},inplace=True)
    df_kickout['IDA Times Applied Composite Score'] = create_apps_feature(df)
    df_kickout['IDA Confirmed PII Composite Score'] = create_verification_feature(df)
    df_kickout['IDA Reported Fraud Activity'] = create_reported_feature(df)
    df_kickout['IDA Recently Reported Fraud'] = create_recent_feature(df)
    return df_kickout


def retrieve_binaries(df):
    binaries = df.select_dtypes(include='bool')
    extras = ['CreditReport.IDN_V1.ITEMAA3_ALLATTS_OFACMATCHCODE','CreditReport.IDN_V1.ITEMAA3_ALLATTS_ADDRTYPEGENERALLYHIGHRISK','CreditReport.CMPLY_V1.ITEM_1_IDENTITYVERIFICATION_ADDRESS','CreditReport.CMPLY_V1.ITEM_1_IDENTITYVERIFICATION_LASTNAME','CreditReport.CMPLY_V1.ITEM_1_IDENTITYVERIFICATION_FIRSTNAME','CreditReport.EM_V1.ITEM_DOMAINCORPORATE','CreditReport.CMPLY_V1.ITEM_1_IDENTITYVERIFICATION_PHONE']
    extra_df = df[extras]
    binaries = binaries.join(extra_df)
    return binaries

def process_binaries(df):
    #replacing all No's and Yes's with 0's and 1's
    df.replace(False,float(0),inplace=True)
    df.replace(True,float(1),inplace=True)
    df.replace('Green',float(1),inplace=True)
    df.replace('Red',float(0),inplace=True)
    df.replace('Yes',float(1),inplace=True)
    df.replace('No',float(0),inplace=True)
    df.replace('N',float(0),inplace=True)
    df.replace('NN',float(1),inplace=True)
    df.replace('M',float(1),inplace=True)
    return df

def retrieve_cats(df):
    cat_cols = ['CreditReport.CMPLY_V1.ITEM_1_GRADE','CreditReport.EM_V1.ITEM_STATUS', 'CreditReport.EM_V1.ITEM_COUNTRY','CreditReport.EM_V1.ITEM_EAADVICE','CreditReport.EM_V1.ITEM_EARISKBAND','CreditReport.EM_V1.ITEM_EMAILEXISTS','CreditReport.EM_V1.ITEM_DOMAINRISKLEVEL','CreditReport.EM_V1.ITEM_FRAUDRISK']
    cat_initial = df[cat_cols]
    return cat_initial

def process_cats(df):
    #reducing categories in the countries column
    keep_countries = 'US CA CH FR DE ES GB'.split()
    country_codes = [x if x in keep_countries else "other" for x in df['CreditReport.EM_V1.ITEM_COUNTRY']]
    df['CreditReport.EM_V1.ITEM_COUNTRY'] = country_codes
    #cleaning up the EA risk band
    fraud_dict = {'Fraud Score 1 to 100':'FraudBucket1','Fraud Score 101 to 300':'FraudBucket2','Fraud Score 301 to 600':'FraudBucket3','Fraud Score 601 to 799':'FraudBucket4','Fraud Score 800 to 899':'FraudBucket5','Fraud Score 900 to 999':'FraudBucket6'}
    df['CreditReport.EM_V1.ITEM_EARISKBAND'] = [fraud_dict[x] for x in df['CreditReport.EM_V1.ITEM_EARISKBAND']]
    #processing fraud category
    fraudcategory = [x[4:] for x in df['CreditReport.EM_V1.ITEM_FRAUDRISK']]
    df['CreditReport.EM_V1.ITEM_FRAUDRISK.CATEGORY'] = fraudcategory
    df.drop(['CreditReport.EM_V1.ITEM_FRAUDRISK'],axis=1,inplace=True)
    #one hot encoding and removing the correlated columns
    cf2 = pd.get_dummies(df,drop_first=True)
    cf2_corr = cf2.corr().abs()
    upper_tri3 = cf2_corr.where(np.triu(np.ones(cf2_corr.shape),k=1).astype(np.bool))
    to_drop3 = [column for column in upper_tri3.columns if any(upper_tri3[column] > 0.925)]
    cf2_reduced = cf2.drop(to_drop3,axis=1)
    return cf2_reduced

def retrieve_numeric(df):
    nummys = df.select_dtypes(include='float64')
    drop_these = ['CreditReport.IDN_V1.ITEMAA3_ALLATTS_LIKELYAGEATSSNISSUANCE','CreditReport.CMPLY_V1.ITEM_1_RESULTCODE2','CreditReport.CMPLY_V1.ITEM_1_RESULTCODE3','CreditReport.CMPLY_V1.ITEM_1_RESULTCODE4','CreditReport.CMPLY_V1.ITEM_1_CONTRARYIDENTITY_SSN','CreditReport.CMPLY_V1.ITEM_1_CONTRARYIDENTITY_ZIP','CreditReport.CMPLY_V1.ITEM_1_CONTRARYIDENTITY_PHONE','CreditReport.ID95_V1.IDSCORERESULTCODE2','CreditReport.ID95_V1.IDSCORERESULTCODE3']
    add_these = ['CreditReport.IDN_V1.ITEMAA3_ALLATTS_CONSISTENCYSNAPD','CreditReport.IDN_V1.ITEMAA3_ALLATTS_CONSISTENCYSSNDOB','CreditReport.IDN_V1.ITEMAA3_ALLATTS_PRIMPHONETYPECODE','CreditReport.IDN_V1.ITEMAA3_ALLATTS_NUMCHARREPEATEMAIL','CreditReport.IDN_V1.ITEMAA3_ALLATTS_DISTANCEZIPPRIMPHONE','CreditReport.IDN_V1.ITEMAA3_ALLATTS_CONSISTENCYSSNOTHERELEMENTS','CreditReport.ID95_V1.IDSCORE','CreditReport.EM_V1.ITEM_EASCORE']
    num_df = nummys.drop(drop_these,axis=1)
    num_df = num_df.join(df[add_these])
    fraudscore = [float(x[0:3]) for x in df['CreditReport.EM_V1.ITEM_FRAUDRISK']]
    num_df['CreditReport.EM_V1.ITEM_FRAUDRISK.SCORE'] = fraudscore
    return num_df

def process_numeric(df):
    normalize = MinMaxScaler().fit(df)
    num_df = pd.DataFrame(normalize.transform(df),columns=df.columns)
    num_df.set_index(df.index,inplace=True)
    num_norm_corr = num_df.corr().abs()
    upper_tri = num_norm_corr.where(np.triu(np.ones(num_norm_corr.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.925)]
    num_reduced = num_df.drop(to_drop,axis=1)
    num_reduced = num_reduced.astype('float64')
    num_reduced.fillna(float(0),inplace=True)
    return num_reduced

def process_for_iso(df):
    #preliminary processing of raw data
    removals = [x for x in df.customer_id if 'test' in str(x)]
    df = df[~df.customer_id.isin(removals)]
    df.set_index('customer_id',inplace=True)
    
    #separating and processing the variables accordingly
    binary_df = retrieve_binaries(df)
    binary_df = process_binaries(binary_df)
    cat_df = retrieve_cats(df)
    cat_df = process_cats(cat_df)
    num_df = retrieve_numeric(df)
    num_df = process_numeric(num_df)

    #re-joining the separated dataframes
    joindfs = [cat_df,binary_df]
    processed_df = num_df.join(joindfs)
    return processed_df