from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import os,shutil

conf = SparkConf().setAppName("appName").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

def vectorize(training):
    hashingTF = HashingTF()
    tf_training = training.map(lambda tup: hashingTF.transform(tup[1]))
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

def parseTweet(line):
    parts = line.split(',')
    label = float(parts[0][1])
    tweet = parts[5]
    words = tweet.strip().split(" ")
    return (label, words)

def testing():
    filename_training="training_neutral_removed.csv"
    script_dir = os.path.dirname(__file__)
    training_file = os.path.join(script_dir, filename_training)
    allData = sc.textFile(training_file)
    header = allData.first()
    training = allData.filter(lambda x: x != header).map(parseTweet)
    labeled_training_data, labeled_testing_data=vectorize(training).randomSplit([0.7, 0.3], seed=0)
    print("Training SVM model....")
    model = SVMWithSGD.train(labeled_training_data, iterations=100)
    print("Done training")

    nb_output_dir = '/home/emmittxu/Desktop/Stock-Sentiment-alalysis/mySVMModel'
    shutil.rmtree(nb_output_dir, ignore_errors=True)
    model.save(sc, nb_output_dir)
    print("Done saving model")
    print("Testing..........")
    predictionAndLabel = labeled_testing_data.map(lambda p : (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / labeled_testing_data.count()
    print accuracy


def main():
    testing()

if __name__ == "__main__":
    # calling main function
    main()
