from pyspark.streaming import StreamingContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import time as tm
from threading import Thread
import numpy as np

conf = SparkConf().setAppName("appName").setMaster("local")
conf.set("spark.executor.memory", "2g")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

#Load pretrained models
output_dir1 = '/home/emmittxu/Desktop/Stock-Sentiment-alalysis/Models/myNaiveBayesModel'
output_dir2 = '/home/emmittxu/Desktop/Stock-Sentiment-alalysis/Models/sent_stockModel'
print("Loading model.......")
model1 = NaiveBayesModel.load(sc, output_dir1)
model2 = NaiveBayesModel.load(sc, output_dir2)
print("Models successfully loaded......")


#Global variables to record the number of positive and negative sentiments
negative=0.0
neutral=0.0
positive=0.0

#Do feature extraction using TF-IDF and feed feature vectors to the sentiment classifier
def vectorize_feature(training):
	try:
		global positive
		global negative
		positive=0
		negative=0
		#Do TF-IDF
		hashingTF = HashingTF()
		tf_training = training.map(lambda tup: hashingTF.transform(tup[1]))
		idf_training = IDF().fit(tf_training)
		tfidf_training = idf_training.transform(tf_training)
		for tweet, feature in zip(training.collect(), tfidf_training.collect()):
			label=model1.predict(feature)#Feed feature vector to the sentiment classifier
			if(label==4):
				positive+=1
			elif(label==2):
				neutral+=1
			else:
				negative+=1
	except:
		pass

#Consumer thread that listens on the port and processes data consumed
def spark_thread():
	spark = SparkSession.builder.master("local[2]")\
	.appName("sentiment-stock")\
	.config("spark.some.config.option", "some-value") \
	.getOrCreate()
	sc = spark.sparkContext
	ssc = StreamingContext(sc, 5)
	tweets_ds = ssc.socketTextStream("localhost", 9999)#Get data from the port
	try:
		#Feed each rdd in the tweets_feature to the function vectorize_feature()
		tweets_feature=tweets_ds.foreachRDD(vectorize_feature)
	except:
		pass
	ssc.start()
	ssc.awaitTermination()

#Timer is a seperate thread that counts the number of positive/negative sentiments in each interval
def timer():
	global negative
	global positive
	while True:
		print("timer")
		pos = 0
		neg = 0
		for i in range(1,101):
			tm.sleep(0.1)
			if i%50 == 0:
				pos += positive
				neg += negative
				print("*")*100
				print("Last 10 seconds positive/negative: ", positive, negative)
				print(model2.predict(np.array([positive, negative])))

if __name__ == '__main__':
	timer_1 = Thread(target = timer, args = ())
	timer_1.setDaemon(True)
	timer_1.start()
	spark_thread()
