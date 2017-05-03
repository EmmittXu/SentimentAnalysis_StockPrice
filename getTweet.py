#This is a pure python script that streams tweets using Twitter API and tweepy
#Keywords can be specified in the main function
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import time, os
import json
import random
import re

class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = "utlxZzQmSyEFyV7bkvGiUL0kz"
        consumer_secret = "V8SfcipgV10qVhYRYNa2SYDa59t9AhlQVdGtthAKSXrtyG2S9u"
        access_token = "3018213812-Q3pcr2H3tricjOhRDBiIVEjLWgKGeQX3Yf7rAsv"
        access_token_secret = "R85oKLMeaxFpflxQHRTSFOfB0p0LAwWpHLmFoit5AXoQ2"

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)

        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) \
                                    |(\w+:\/\/\S+)", " ", tweet).split())

    def remove_emoji(self, text):
        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        return (emoji_pattern.sub(r'', text))

    def get_tweets(self, query, count):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []
        parsed_tweets=[]
        c = tweepy.Cursor(self.api.search, q=query).items(count)
        directory=make_dir(query)
        save_file = open(os.path.join(directory,'{}.json'.format(query)), 'a')
        for tweet in c:
            if(tweet.lang=='en'):
                parsedTweet=self.clean_tweet(self.remove_emoji(tweet.text.encode('utf-8')))
                tweets.append(parsedTweet)
                if tweet.retweet_count > 0:
                    if parsedTweet not in tweets:
                        tweets.append(parsedTweet)
                    else:
                        tweets.append(parsedTweet)
        return tweets



def make_dir(query):
    DIR="tweets"
    #Create main directory if necessary
    if not os.path.exists(DIR):
           os.mkdir(DIR)
    DIR = os.path.join(DIR, query)
    #Make sub-directory if necessary
    if not os.path.exists(DIR):
           os.mkdir(DIR)
    return DIR

def on_data(query, tweet, dir):
    save_file = open(os.path.join(dir,'{}.json'.format(query)), 'a')
    for i in tweet:
        save_file.write(json.dumps(i._json))
    print("*")*60
    print tweet
    print("*")*60
    save_file.write(str(tweet))

def main():
    # creating object of TwitterClient Class
    api = TwitterClient()
    # calling function to get tweets
    tweets = api.get_tweets(query ="Apple", count =100)
    directory=make_dir(search_input)
    on_data(query="Apple", tweet=tweets,dir=directory)

if __name__ == "__main__":
    # calling main function
    main()
