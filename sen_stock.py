#This script trains the stock price classifier and runs testing on 30% of the training data
#Also save the model to disk
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import os,shutil

conf = SparkConf().setAppName("appName").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

def parseData(line):
    parts = line.split(',')
    label = float(parts[4])
    point1=float(parts[1])
    point2=float(parts[2])
    return LabeledPoint(label,[point1/(point1+point2), point2/(point1+point2)])

def train():
    print("Training..........")
    script_dir = os.path.dirname(__file__)
    filename_training="aapl_training.csv"
    training_file = os.path.join(script_dir, filename_training)
    allData = sc.textFile(training_file)
    training, testing=allData.randomSplit([0.7, 0.3], seed=1)
    training= training.map(parseData)
    testing=testing.map(parseData)
    print("Training Naive Bayes model....")
    nb_model = NaiveBayes.train(training, 1.0)
    print("Done training")
    nb_output_dir = '/home/emmittxu/Desktop/Stock-Sentiment-alalysis/Models/sent_stockModel'
    shutil.rmtree(nb_output_dir, ignore_errors=True)
    nb_model.save(sc, nb_output_dir)
    print("Done saving model")
    print("Testing accuracy:")
    predictionAndLabel = testing.map(lambda p : (nb_model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / testing.count()
    print accuracy

def main():
    train()

if __name__ == "__main__":
    # calling main function
    main()
