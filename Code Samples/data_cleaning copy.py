#libraries
import pandas as pd
import numpy as np 
import json
from pandas.io.json._normalize import nested_to_record    

#csc data
cscheader1 = ['Client','raw_text','request_id','applicant_uuid','timestmp','raw_request_id','response_id','decision_uuid','decision_name']
csc_edm_data = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Data/CSC/fraudproject_csc_edm_datapull.csv',names=cscheader1)

cnames = ['raw_id','customer_id','raw_uuid','credit_report_name','parsed_report','date']
credit_reports_file = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Data/CSC/csc_reports_data.csv',names=cnames)


def process_credit_report(text):
    j = json.loads(text)
    return nested_to_record(j)

list(credit_reports_file.credit_report_name.unique())

fraud_reports = ['idanalytics_network_attributes','idanalytics_comply360','idanalytics_id_score','emailage','whitepages_pro','equifax_eid_compare']

fraud_report_df = credit_reports_file[credit_reports_file['credit_report_name'].isin(fraud_reports)]

#create separate dataframes
ida_net_attr = fraud_report_df[fraud_report_df['credit_report_name']=='idanalytics_network_attributes']
ida_comply360 = fraud_report_df[fraud_report_df['credit_report_name']=='idanalytics_comply360']
ida_id_score = fraud_report_df[fraud_report_df['credit_report_name']=='idanalytics_id_score']
emailage = fraud_report_df[fraud_report_df['credit_report_name']=='emailage']

#create processed credit reports, format df's, and create individual feature df's

ida_net_attr['processed_credit_report'] = ida_net_attr.parsed_report.apply(lambda x: process_credit_report(x))
ida_comply360['processed_credit_report'] = ida_comply360.parsed_report.apply(lambda x: process_credit_report(x))
ida_id_score['processed_credit_report'] = ida_id_score.parsed_report.apply(lambda x: process_credit_report(x))
emailage['processed_credit_report'] = emailage.parsed_report.apply(lambda x: process_credit_report(x))

ida_net_attr.reset_index(inplace=True)
ina_features = pd.DataFrame.from_records(ida_net_attr['processed_credit_report'],index=ida_net_attr['raw_uuid'])
ina_features.reset_index(inplace=True)

ida_comply360.reset_index(inplace=True)
ic3_features = pd.DataFrame.from_records(ida_comply360['processed_credit_report'],index=ida_comply360['raw_uuid'])
ic3_features.reset_index(inplace=True)

ida_id_score.reset_index(inplace=True)
iis_features = pd.DataFrame.from_records(ida_id_score['processed_credit_report'],index=ida_id_score['raw_uuid'])
iis_features.reset_index(inplace=True)

emailage.reset_index(inplace=True)
ema_features = pd.DataFrame.from_records(emailage['processed_credit_report'],index=emailage['raw_uuid'])
ema_features.reset_index(inplace=True)


#merging with originals to add customer id's
ina2 = ina_features.merge(ida_net_attr[['customer_id','raw_uuid']], how='left',on='raw_uuid')
ina2['raw_ina2'] = ina2['raw_uuid']
ic32 = ic3_features.merge(ida_comply360[['customer_id','raw_uuid']], how='left',on='raw_uuid')
ic32['raw_ic32'] = ic32['raw_uuid']
iis2 = iis_features.merge(ida_id_score[['customer_id','raw_uuid']], how='left',on='raw_uuid')
iis2['raw_iis2'] = iis2['raw_uuid']
ema2 = ema_features.merge(emailage[['customer_id','raw_uuid']], how='left',on='raw_uuid')
ema2['raw_ema2'] = ema2['raw_uuid']

#merging the dataframes
feature_dfs_list = [ina2,ic32,iis2,ema2]

for df in feature_dfs_list:
    df.drop(['raw_uuid'],inplace=True,axis=1)
    df = df[df['customer_id']!='test_decision_application']
    df.set_index('customer_id',inplace=True)
    df.fillna(np.nan,inplace=True)

feature_dfs_list = [ina2,ic32,iis2,ema2]

#creating only features that have more than one value dataframes
i = 0
for df in feature_dfs_list:
    unilist = []
    df_features = list(df.columns)
    for cl in df_features:
        if df[cl].nunique()>1:
            unilist.append(cl)
    if i==0:
        red_ina2 = ina2[unilist]
    elif i==1:
        red_ic32 = ic32[unilist]
    elif i==2:
        red_iis2 = iis2[unilist]
    elif i==3:
        red_ema2 = ema2[unilist]
    else:
        print("RED FLAG")
    i = i+1

reduced_dfs_list = [red_ina2,red_ic32,red_iis2,red_ema2]

for df in reduced_dfs_list:
    df.drop_duplicates(subset='customer_id',inplace=True,keep='last')
    df.set_index('customer_id',inplace=True)

for df in feature_dfs_list:
    df.drop_duplicates(subset='customer_id',keep='last')
    df.set_index('customer_id',inplace=True)

reduced_dfs_list = [red_ina2,red_ic32,red_iis2,red_ema2]

fraud_features_df = pd.concat(reduced_dfs_list,axis=1,join='inner')
fraud_features_df.head()
fraud_features_df.shape
fraud_features_df.fillna(np.nan,inplace=True)
fraud_features_df.reset_index(inplace=True)

#removing irrelevant info
dropcols = ['CreditReport.IDN_V1.REQUESTID','CreditReport.IDN_V1.ITEMAA3_APPID','CreditReport.IDN_V1.ITEMAA3_IDASEQUENCE','CreditReport.IDN_V1.ITEMAA3_IDATIMESTAMP','CreditReport.IDN_V1.ITEMAA3_ALLATTS_APPLICANTAGE','CreditReport.IDN_V1.ITEMAA3_ALLATTS_LIKELYGENDER','raw_ina2','CreditReport.CMPLY_V1.REQUESTID','CreditReport.CMPLY_V1.ITEM_1_APPID','raw_ic32','CreditReport.ID95_V1.APPID','CreditReport.ID95_V1.REQUESTID','CreditReport.ID95_V1.IDASEQUENCE','CreditReport.ID95_V1.IDATIMESTAMP','raw_iis2','CreditReport.EM_V1.ITEM_GENDER','raw_ema2','CreditReport.EM_V1.ITEM_LASTVERIFICATIONDATE','CreditReport.EM_V1.ITEM_FIRSTVERIFICATIONDATE']
ff2 = fraud_features_df.drop(dropcols,axis=1)
datecols = [col for col in list(ff2.columns) if 'date' in col.lower()]
ff3 = ff2.drop(datecols,axis=1)
ff3.rename(columns={'index':'customer_id'},inplace=True)
ff3.to_csv('/Users/tross/Desktop/Analytics/FRAUD/Data/csc_cleaned_fraud_features.csv',index=False)

ff3.head()
ff3.shape


## Additional Unused Cleaning Code ##
#From John 
# import json
# import xml.etree.ElementTree as ET
# def process_raw_LNRV5(text):
#     ns = {'RV':'http://webservices.seisint.com/WsRiskView'}
#     #Load string to JSON
#     j = json.loads(text)
#     #Find and parse raw XML
#     try:
#         ln = j['data']['attributes']['decision_result']['extended_data']['raw_reports']['lexis_nexis_risk_view']
#         lnx = ET.ElementTree(ET.fromstring(ln)).getroot()
#     except:
#         return "No LN Data"
#     #Find attributes, extract into key:value pairs
#     raw = {}
#     for attribute in lnx.findall(".//RV:AttributesGroup//RV:Attribute",ns):
#         attribute_name = attribute.find('RV:Name',ns).text
#         attribute_value = attribute.find('RV:Value',ns).text
#         raw[attribute_name] = attribute_value
#     #return key:value pairs
#     return raw
# import pandas as pd
# df = pd.read_csv('/Users/tross/Desktop/csc_edmdata_100.csv')
# df['LNRV5'] = df.raw_text.apply(lambda x: process_raw_LNRV5(x))
# df.head()

# LNFeatures = pd.DataFrame(df['LNRV5'][0],index=df['applicant_uuid'])
# LNFeatures.head()
# len(LNFeatures)

# from pandas.io.json._normalize import nested_to_record    
# def process_ida(text):
#     j = json.loads(text)
#     try:
#         ida_json = j['data']['attributes']['decision_result']['extended_data']['ida_rules']
#         ida_record = nested_to_record(ida_json)
#         return ida_record
#     except:
#         return "No IDA"

# df['IDA'] = df.raw_text.apply(lambda x: process_ida(x))
# df.head()

# #manually identifying key features
# def manually_select():
#     all_fraud_features = list(fraud_features_df.columns)
#     keeplist=[]
#     droplist=[]
#     for cl in all_fraud_features:
#         print()
#         print(f'{cl}:')
#         print(ina2[cl].describe())
#         print(ina2[cl][0:5])
#         if input("Keep?") == "yes":
#             keeplist.append(cl)
#         else:
#             droplist.append(cl)