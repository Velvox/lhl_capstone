import tweepy
import requests as re
import os
import json
import pandas as pd

## instantiate os variables

api_key = os.environ['TWITTER_API_KEY']
api_secret = os.environ['TWITTER_API_KEY_SECRET']
access_token = os.environ['TWITTER_ACCESS_TOKEN']
access_token_secret = os.environ['TWITTER_ACCESS_TOKEN_SECRET']

## Set up auth request

auth_url = "https://api.twitter.com/oauth2/token?grant_type=client_credentials"

auth_data = {
    'username' : api_key,
    'password' : api_secret
}

auth_auth = (api_key, api_secret)

auth_params = {
    'grant_type': "client_credentials"
}

## get the response
auth_response = re.post(auth_url, data=auth_data, auth=auth_auth)
auth_response.json()

## assign the response
token_type = auth_response.json()['token_type']
access_token = auth_response.json()['access_token']

## set up filename

filename = "netflix_tweets_v1.json"

## store data from response

class StatusListener(tweepy.StreamingClient):
        
    def on_tweet(self, tweet):
        # with open("output.txt", 'a') as f:
        #     f.write(json.dumps(tweet.text))
        #     f.write('\n')
        #     print(tweet)

        tweet_dict = {
            'tweet_id': str(tweet.id),
            'author_id': str(tweet.author_id),
            'created_at': str(tweet.created_at),
            'tweet_text': str(tweet),
        }

        with open(filename, 'a') as f:
            f.write(json.dumps(tweet_dict))
            f.write('\n')
            print(tweet.author_id)
            print(tweet.created_at)
            print(tweet)
    
# instantiate streaming client

main = StatusListener(access_token)

# run client

main.filter(expansions=['author_id'], tweet_fields=['created_at'], user_fields='location')