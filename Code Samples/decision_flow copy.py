#necessary libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessing
from sklearn.ensemble import IsolationForest

#importing current dataframe
#dfeats = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Final/decision_features.csv',index_col=0)
raw_data = pd.read_csv('/Users/tross/Desktop/Analytics/FRAUD/Data/csc_cleaned_fraud_features.csv')
dfeats = preprocessing.process_for_kickout(raw_data)
diso = preprocessing.process_for_iso(raw_data)

#importing the model
model_path = '/Users/tross/Desktop/Analytics/FRAUD/Final/final_isoforest_model.sav'
anomaly_detector = pickle.load(open(model_path, 'rb'))

#creating the decision logic functions
def idscore_logic(df):
    idscore_declines=list(df[df['IDA ID Score 95']>=925].index)
    idscore_continues=list(df[df['IDA ID Score 95']<925].index)
    return idscore_declines, idscore_continues

def comply360_logic(df):
    declining_grades = ['F','I']
    comply360_declines = list(df[df['IDA Comply360 Grade'].isin(declining_grades)].index)
    comply360_continues = list(df[~df['IDA Comply360 Grade'].isin(declining_grades)].index)
    return comply360_declines, comply360_continues

def velocity_logic(df):
    velocity_declines = list(df[df['IDA Velocity Based Risk']>0.45].index)
    velocity_continues = list(df[df['IDA Velocity Based Risk']<=0.45].index)
    return velocity_declines, velocity_continues

def applications_logic(df):
    app_declines = list(df[df['IDA Times Applied Composite Score']>260].index)
    app_continues = list(df[df['IDA Times Applied Composite Score']<=260].index)
    return app_declines,app_continues

def fraudulent_logic(df):
    fraud_declines = list(df[df['IDA Reported Fraud Activity']=='High'].index)
    fraud_continues = list(df[df['IDA Reported Fraud Activity']!='High'].index)
    return fraud_declines, fraud_continues

def recent_logic(df):
    recent_declines = list(df[df['IDA Recently Reported Fraud']==1].index)
    recent_continues = list(df[df['IDA Recently Reported Fraud']!=1].index)
    return recent_declines,recent_continues

#processing the applicants throught the anomaly detection layer
def detect_anomalies(df):
    customers = list(df.index.values)
    isoforestdf = pd.DataFrame(customers,columns=['customer_id'])
    isoforestdf['scores'] = anomaly_detector.decision_function(df)
    isoforestdf['anomaly'] = anomaly_detector.predict(df)
    return isoforestdf

def retrieve_iso(df):
    df = iso2.copy()
    df = df.astype('float64')
    df.fillna(float(0),inplace=True)
    isoforestdf = detect_anomalies(df)
    anomaly_scores = list(isoforestdf.scores)
    return anomaly_scores

def run_em_through(df):
    #id score decision logic
    kickout1, keepem1 = idscore_logic(df)
    df2 = df.loc[keepem1]
    #comply360 decision logic
    kickout2,keepem2 = comply360_logic(df2)
    df3 = df2.loc[keepem2]
    #velocity decision logic
    kickout3,keepem3 = velocity_logic(df3)
    df4 = df3.loc[keepem3]
    #applications decision logic
    kickout4,keepem4 = applications_logic(df4)
    df5 = df4.loc[keepem4]
    #fraudulent decision logic
    kickout6,keepem6 = fraudulent_logic(df5)
    df6 = df5.loc[keepem6]
    #recent decision logic
    kickout7,keepem7 = recent_logic(df6)
    df7 = df6.loc[keepem7]
    #send results back
    declines = {"ID Score":kickout1,"Comply360":kickout2,"Velocity":kickout3,"Applications":kickout4,"Fraudulent":kickout6,"Recent":kickout7}
    remaining = df7.copy()
    return declines,remaining

def retrieve_ross(df):
    #variable one: id score on scale from 0->1, closer to 1 is higher risk of fraud
    v1 = lambda x: (int(df.loc[x]['IDA ID Score 95'])/1000)
    #variable two: a mapped comply360 grade, 1 being D, 0 being A
    grade_dict = {'A':0.125,'B':0.375, 'C':0.625,'D':0.875,np.nan : 0.5}
    v2 = lambda x: grade_dict[df.loc[x]['IDA Comply360 Grade']]
    #variable three: simply the velocity risk score
    v3 = lambda x: df.loc[x]['IDA Velocity Based Risk']
    ross_scores = [(v1(x)/4) + (v2(x)/4) + v3(x) for x in df.index]
    ross_scores = [float(str(x)[0:5]) for x in ross_scores]
    return ross_scores

#decision flow to run the applicants through, it returns a list of declined, remediated, and approved applicants
def flow(processed_data,iso_data):
    #initiate variables
    p2 = processed_data.copy()

    #process the data through the decision logic
    immediate_declines,p3 = run_em_through(p2)

    #retrieve ross scores
    p3['Preliminary Score'] = retrieve_ross(p3)

    #retrieve anomaly scores
    iso2 = iso_data.loc[p3.index]
    p3['IsoForest Score'] = retrieve_iso(iso2)

    return immediate_declines, p3

declines,processed_applicants = flow(dfeats,diso)
processed_applicants.head()
#declined_applicants,remediate_applicants,allclear_applicants = flow(dfeats)
