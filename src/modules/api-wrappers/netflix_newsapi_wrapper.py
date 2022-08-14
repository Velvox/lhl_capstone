from newsapi import NewsApiClient
import os
import datetime as dt
import time
import json

api_key = os.environ["NEWS_API_KEY"]

# establish initial variables
start = dt.datetime(2022, 8, 13)
day = dt.timedelta(1,0)

# go back one day
start = start - day

# one day ends after delta
end = start + day

# number of free daily tokens
tokens = 10

# Init Client
newsapi = NewsApiClient(api_key=api_key)

response_dict = {}
try:
    for token in range(tokens):
        print(start)
        response = newsapi.get_everything(qintitle='Netflix company',
                                            from_param=start,
                                            to=end,
                                            language='en',
                                            sort_by='relevancy')
        # store response if successful
        response_dict[start.strftime('%Y-%m-%d')] = response
        start = start - day
        end = end - day
        time.sleep(1)

except:
    pass

with open('../../../data/news/raw/news_aug_13.json', 'w') as fp:
    json.dump(response_dict, fp, sort_keys=True, indent=4)