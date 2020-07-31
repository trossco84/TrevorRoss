import requests

url = "https://paper-api.alpaca.markets/v2/account"

payload = {}
headers = {
  'APCA-API-KEY-ID': 'PKMAT61YC9Q73264FPLN',
  'APCA-API-SECRET-KEY': '8aEpEfazEaxL3DA6RREYb5uf0B9m/sCRz55av01/'
}
response = requests.request("GET", url, headers=headers, data = payload)
print(response.text.encode('utf8'))
pprint(response.json())
import json 
pretty_json = json.loads(response.text.encode('utf8'))
print(json.dumps(pretty_json,indent=2))

acct = pretty_json
acct
buying_power = acct['buying_power']
buying_power

assets = api.list_assets()
assets = [asset for asset in assets if asset.tradable ]
a
all_assets_df = pd.DataFrame()
all_assets_df['Ticker'] = [a.symbol for a in assets]
all_assets_df['']
all_assets_df.head()

#--Second way is more direct and uses alpaca created library
#importing alpaca specific librarys
import alpaca_trade_api as tradeapi
# import alpaca_robinhood_volume as robbyV
# import alpha_vantage as dv8

#importing a library to retrieve environment variables
from decouple import config

#setting environment variables
key_id = 'PKMAT61YC9Q73264FPLN'
secret_key = '8aEpEfazEaxL3DA6RREYb5uf0B9m/sCRz55av01/'
key_id
#creat api call (uses REST)
api = tradeapi.REST(key_id,secret_key)
tradeapi.REST(key_id='PKMAT61YC9Q73264FPLN',secret_key='8aEpEfazEaxL3DA6RREYb5uf0B9m/sCRz55av01/').get_account()

# #specific calls p
# #getting the account
account = api.get_account(     )
#listing current positions
api.list_positions()
#submits an order
# api.submit_order(
#     symbol='SPY',
#     side='buy',
#     type='market',
#     qty='100',
#     time_in_force='day',
#     order_class='bracket',
#     take_profit=dict(
#         limit_price='305.0',
#     ),
#     stop_loss=dict(
#         stop_price='295.5',
#         limit_price='295.5',
#     )
# )

#### TIME FOR FUN ####
import pandas as pd
import statistics
import sys
import time

from datetime import datetime, timedelta
from pytz import timezone