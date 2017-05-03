from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import Row
from pandas.tools.plotting import scatter_matrix

last_stock_price = 69.54
def myFunc(s):
	global last_stock_price
	change = 0
	p = s.split(',')
	change = last_stock_price - float(p[6])
	last_stock_price = float(p[6])

	# return stock data with its movement(change)
	return (p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5]),change)

def trend(s):
	if s.Change > 0:
		return 1
	else:
		return 0

if __name__ == '__main__':
	# https://spark.apache.org/docs/2.0.1/api/java/org/apache/spark/sql/SparkSession.html
	spark = SparkSession.builder.master("local")\
	.appName("sentiment-stock")\
	.config("spark.some.config.option", "some-value") \
	.getOrCreate()
	sc = spark.sparkContext


	#
	# input historical stock data file directory and name below
	stocks = sc.textFile('aapl.txt').map(myFunc)
	#
	#

	# Construct stock's dataframe
	schemaString_stock = "Date Open High Low Close Volume Change"
	schema_stock = StructType([StructField('Date', StringType(), True),\
		StructField('Open', FloatType(), True),\
		StructField('High', FloatType(), True),\
		StructField('Low', FloatType(), True),\
		StructField('Close', FloatType(), True),\
		StructField('Volume', IntegerType(), True),
		StructField('Change', FloatType(), True)])

	schema_stock = spark.createDataFrame(stocks, schema_stock).select(col('Date'), col('Change'))



	#
	# input historical sentiments data file directory and name below
	parts = sc.textFile('news_sentiment.txt').map(lambda l: l.split(' '))
	#
	#
	
	# Construct sentiments' dataframe
	sentiments = parts.map(lambda p: (p[0], int(p[1]), int(p[2])))
	schema_sentiments = StructType([StructField('Date_2', StringType(), True),\
		StructField('Pos_score', IntegerType(), True),\
		StructField('Neg_score', IntegerType(), True)])
	schema_sentiments = spark.createDataFrame(sentiments, schema_sentiments)

	# union of sentiments and stocks dataframes
	sentiments_stocks = schema_stock.join(schema_sentiments, schema_sentiments.Date_2 == schema_stock.Date).select(col('Date'), col('Change'), col('Pos_score'), col('Neg_score'))
	sentiments_stocks = sentiments_stocks.withColumn('increasing', sentiments_stocks.Change > 0)
	final = sentiments_stocks.select(col('Pos_score'), col('Neg_score'), col('increasing'))

	#
	# input output csv file name below
	final.toPandas().to_csv('aapl_news_training.csv')
	#
	#





