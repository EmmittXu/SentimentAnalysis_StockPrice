from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import os, shutil
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF


conf = SparkConf().setAppName("appName").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

def parseTweet(line):
    parts = line.split(',')
    label = float(parts[0][1])
    tweet = parts[5]
    words = tweet.strip().split(" ")
    return (label, words)

def vectorize(training):
    hashingTF = HashingTF()
    tf_training = training.map(lambda tup: hashingTF.transform(tup[1])).persist()
    idf_training = IDF().fit(tf_training)
    tfidf_training = idf_training.transform(tf_training)
    tfidf_idx = tfidf_training.zipWithIndex()
    training_idx = training.zipWithIndex()
    idx_training = training_idx.map(lambda line: (line[1], line[0]))
    idx_tfidf = tfidf_idx.map(lambda l: (l[1], l[0]))
    joined_tfidf_training = idx_training.join(idx_tfidf)
    training_labeled = joined_tfidf_training.map(lambda tup: tup[1])
    labeled_training_data = training_labeled.map(lambda k: LabeledPoint(k[0][0], k[1]))
    return labeled_training_data


def train():
    print("Training..........")
    script_dir = os.path.dirname(__file__)
    filename_training="training_neutral_removed.csv"
    training_file = os.path.join(script_dir, filename_training)
    allData = sc.textFile(training_file).sample(False,0.1)
    training = allData.map(parseTweet)
    labeled_training_data=vectorize(training)
    print("Training logistic regression model....")
    # Build the model
    svm_model = SVMWithSGD.train(labeled_training_data, iterations=300)
    print("Done training")
    svm_output_dir = '/home/emmittxu/Desktop/Stock-Sentiment-alalysis/mySVMModel'
    shutil.rmtree(svm_output_dir, ignore_errors=True)
    svm_model.save(sc, svm_output_dir)
    print("Done saving model")
    return svm_model, svm_output_dir


def main():
    train()

if __name__ == "__main__":
    # calling main function
    main()
