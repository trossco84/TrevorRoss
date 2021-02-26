#Required Libraries
import pandas as pd
import numpy as np 
from datetime import datetime
import json
import os
import ast
from pandas.io.json._normalize import nested_to_record
import time
import app2


#initial data
milliman_initial_data = pd.read_csv('/Users/tross/Desktop/Clients/Current/Milliman/Retro Study Q1 2021/ENOVA_ATO_feed_exported_on_2021-01-21_data_period_2020-12-02_to_2021-01-20.csv')
md3 = milliman_initial_data.copy()

#parsing dates and retrieving desired range
md3['dates'] = [datetime.date(datetime.strptime(x,'%m/%d/%Y %I:%M:%S %p')) for x in md3['Event Time']]
md3['dates'] = pd.to_datetime(md3['dates'])
start_date = datetime(2021,1,1,0,0,0,0)
end_date = md3['dates'].max()
mask = (md3['dates'] >= start_date) & (md3['dates'] <= end_date)
md4 = md3.loc[mask]
md4.reset_index(inplace=True)
md4.drop(['index'],axis=1,inplace=True)

#MLK Day: md5 = [x for x in md4['dates'] if x.day !=18]

#Creating the Dataframe for the Request Documents
mdc = md4.copy()
#test 5 records to start
mdc2 = mdc[70001:]
mdc2 = mdc_extras.copy()
len(mdc2)
mdc18 = list(mdc[0:]['Event ID'])

mdc3 = pd.DataFrame()
mdc3['eid'] = mdc2['Event ID']
mdc3['FirstName'] = mdc2['First Name']
mdc3['LastName'] = mdc2['Last Name']
mdc3['Phone'] = [np.nan if pd.isnull(x) else str(x)[1:11] for x in mdc2['Phone 1']]
mdc3['EmailAddress'] = mdc2['Email 1']
mdc3['IPAddress'] = mdc2['Client IP']
mdc3['Address1'] = mdc2['Address 1']
mdc3['Address2'] = mdc2['Address 2']
mdc3['City'] = mdc2['City']
mdc3['State'] = mdc2['State']
mdc3['Zip'] = np.nan
mdc3['CountryCode'] = mdc2['Country']
mdc3['TransactionType'] = 'Retro'

# skipped_ids = mdc3.eid
# skipped_ids[0:12]
#Creating the Request Document,
#mdc3.to_json('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/milliman_requests.json',orient='records')
#replace string "}," with "}},{"Request":" in the document, add a starting Request and an ending }


#creating a loop to handle all of the requests and responses from EDC
to_dicts = lambda x: ast.literal_eval(x)
edcresp = pd.read_csv('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/current_dataframe.csv',index_col=0,converters={'EmailageEmailRisk':to_dicts,'WPPro':to_dicts,'Request':to_dicts,'Tracing':to_dicts})
bad_calls = []
for rc in range(0,len(mdc3)):
    if rc == 0:
        strttime = datetime.now().time()
        tok = app2.get_a_token()
        print(datetime.now().time())
        
    print(f'processing {rc+1} of {len(mdc3)}')
    #creates request document for app2
    req = '[{"Request":' + str(mdc3.iloc[rc].to_json()) + '}]'
    with open('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/milliman_request.json', 'w') as json_file:
        json.dump(json.JSONDecoder().decode(req),json_file)
    
    #runs app2 with the request
    try:
        app2.do_the_thing(tok)
        # if rc == 0:
        # #    edcresp = pd.read_json('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/current_response.json')
        # else:
        edcresp = edcresp.append(pd.read_json('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/current_response.json'),ignore_index=True)
    except:
        bad_calls.append(mdc3.iloc[rc].eid)
    if rc % 500 == 0:
        edcresp.to_csv('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/current_dataframe.csv')
    #time.sleep(0.25)
edcresp.to_csv('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/current_dataframe.csv')

# pd.read_json('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/current_dataframe.csv',orient='records')
# pd.read_json()
# len(bad_calls)
# edcresp.head()
# len(edcresp)

#processing the responce data
#targets the data we're interested in
wpdata = pd.DataFrame(edcresp.WPPro)

def process_wppro(text):
    try:
        wppro = text['Response']
        w_record = nested_to_record(wppro)
        return w_record
    except:
        return "No Ekata"

parsedf = pd.DataFrame()
parsedf['eid'] = [x['eid'] for x in edcresp['Request']]
parsedf['EkataNested'] = wpdata.WPPro.apply(lambda x: process_wppro(x))

milliman_final = pd.DataFrame.from_records(parsedf.EkataNested,index=parsedf.eid)
milliman_final.to_csv('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/final_processed_ekata.csv')
milliman_final.head()
milliman_final.shape

#checks
last_eid = int(edcresp.Request[len(edcresp)-1]['eid'])
md4[md4['Event ID']==last_eid]

mdc20 = [str(x) for x in mdc18]
len(mdc20)

blue66 = list(milliman_final[milliman_final.index.isin(mdc20)].index)
green88 = [x for x in mdc20 if x not in blue66]
len(green88)
green88[0:5]
green99 = [int(x) for x in green88]

mdc_extras = md4.set_index('Event ID').loc[green99]
mdc_extras.reset_index(inplace=True)
mdc_extras.to_csv('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/un_processed.csv')

milliman_final.columns

test9 = pd.read_csv('/Users/tross/Desktop/EDC/EDC SDK/PythonDecisionServiceExample-2.0.0/Milliman Retro Q12021/processed_30k.csv')
test9.head()