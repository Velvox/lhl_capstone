import os
import requests as re

def gNewsSearch(date_start, date_end, query='netflix company', lang='en'):
    """
    Takes a query and timeframe to return a set of articles,
    requires api key stores as GNEWS_API_KEY in OS var.
    Returns a response object.
    
    query (str)
    date_start (string) YYYY-MM-DDTHH:MM:SSZ
    date_end (string) YYYY-MM-DDTHH:MM:SSZ
    lang (str)
    """
    
    api_key = os.environ["GNEWS_API_KEY"]

    url = 'https://gnews.io/api/v4/search'

    params = {
        'token': api_key,
        'q': query,
        'from': date_start,
        'to': date_end,
        'lang': lang
    }

    response = (re.get(url, params=params))
    return response