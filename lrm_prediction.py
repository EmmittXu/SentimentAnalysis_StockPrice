from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import getTweet

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

output_dir = '/home/emmittxu/Desktop/DataStreamProj/test/myLogisticRegressionModel'
print("Loading model.......")
model = LogisticRegressionModel.load(sc, output_dir)


api=getTweet.TwitterClient()
tweets = api.get_tweets(query = "United airline", count = 30)
tweets_rdd=sc.parallelize(tweets)
tweets_feature=vectorize_feature(tweets_rdd)
negative=0.0
neural=0.0
positive=0.0

for tweet, feature in zip(tweets_rdd.collect(), tweets_feature.collect()):
    print(tweet, "label:", model.predict(feature))
    if(model.predict(feature)==0.0):
        negative+=1.0
    if(model.predict(feature)==2.0):
        neural+=1.0
    if(model.predict(feature)==4.0):
        positive+=1.0

total=negative+neural+positive
print("negative: ", negative/total)
print("neural: ", neural/total)
print("positive: ", positive/total)
