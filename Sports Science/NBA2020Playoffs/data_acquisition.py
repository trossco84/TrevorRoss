import numpy as np
import pandas as pd
import nba_api.stats as nba
from nba_api.stats.endpoints import commonplayerinfo
import nba_py as nba2
import requests

#generic player info request
player_info = commonplayerinfo.CommonPlayerInfo(player_id=2544)

#initialize the request for lebron james
lebronjames = commonplayerinfo.CommonPlayerInfo(player_id=2544)

#custom headers required to legitimately retrieve the info
custom_headers = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Only available after v1.1.0
# Proxy Support, Custom Headers Support, Timeout Support (in seconds)
import time
time.sleep(10)
lebronjames = commonplayerinfo.CommonPlayerInfo(player_id=2544, proxy='127.0.0.1:80', headers=custom_headers, timeout=100)

import requests

url = "https://api-nba-v1.p.rapidapi.com/seasons/"

headers = {
    'x-rapidapi-host': "api-nba-v1.p.rapidapi.com",
    'x-rapidapi-key': "c2377c792cmsh7d08f5d18db8104p176e54jsn1e2db3b0e2fe"
    }

response = requests.request("GET", url, headers=headers)
response.json()
print(response.text)

seasons = 