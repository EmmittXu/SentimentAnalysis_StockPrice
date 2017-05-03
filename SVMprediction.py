from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import getTweet
from textblob import TextBlob
import re

def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) \
                                |(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    '''
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    '''
    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet).decode('utf-8'))
    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 1.0
    elif analysis.sentiment.polarity == 0:
        return 0.0
    else:
        return -1.0


def vectorize_feature(training):
    hashingTF = HashingTF()
    tf_training = training.map(lambda tup: hashingTF.transform(tup[1]))
    idf_training = IDF().fit(tf_training)
    tfidf_training = idf_training.transform(tf_training)
    return tfidf_training

conf = SparkConf().setAppName("appName").setMaster("local")
conf.set("spark.executor.memory", "2g")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

output_dir = '/home/emmittxu/Desktop/Stock-Sentiment-alalysis/mySVMModel'
print("Loading model.......")
model = SVMModel.load(sc, output_dir)

query="UA apologized"
api=getTweet.TwitterClient()
tweets = api.get_tweets(query = query, count = 200)
# directory=api.make_dir(query)
#
# api.on_data(query, tweets, directory)
tweets_rdd=sc.parallelize(tweets)
tweets_feature=vectorize_feature(tweets_rdd)
negative=0.0
positive=0.0

api_negative=0.0
api_positive=0.0

for tweet, feature in zip(tweets_rdd.collect(), tweets_feature.collect()):
    print(tweet, "label:", model.predict(feature))
    if(model.predict(feature)==0.0):
        negative+=1.0
    if(model.predict(feature)==1.0):
        positive+=1.0
    if(get_tweet_sentiment(tweet)==-1.0):
        api_negative+=1.0
    if(get_tweet_sentiment(tweet)==1.0):
        api_positive+=1.0

total=negative+positive
print("negative: ", negative/total)
print("positive: ", positive/total)
print("*")*50
api_total=api_positive+api_negative
print("api negative: ", api_negative/api_total)
print("api positive ", api_positive/api_total)
