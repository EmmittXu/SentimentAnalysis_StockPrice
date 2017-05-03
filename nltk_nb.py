import nltk
from nltk import FreqDist
import os,sys,shutil
import re
import csv
import pickle
import random

def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) \
                                    |(\w+:\/\/\S+)", " ", tweet).split())
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return (emoji_pattern.sub(r'', text))


def parseTweet(line):
    parts = line.split(',')
    label = float(parts[0][1])
    rawtext = parts[5].encode('utf-8')
    text=clean_tweet(remove_emoji(rawtext))
    return (text, label)


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def train():
    script_dir = os.path.dirname(__file__)
    filename_training="training.1600000.processed.noemoticon.csv"
    training_file = os.path.join(script_dir, filename_training)
    tweets=[]
    with open(training_file, 'rb') as f:
        alllines=csv.reader(f)
        for line in alllines:
            text=remove_emoji(clean_tweet(line[5]))
            label=line[0]
            words_filtered = [e.lower() for e in text.split() if len(e) >= 3]
            tweets.append((words_filtered, label))
    print("Sampling data")
    sampl_tweets = [tweets[i] for i in sorted(random.sample(xrange(len(tweets)), 80000))]
    word_features = get_word_features(get_words_in_tweets(sampl_tweets))
    print("extracting features")

    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    training_set = nltk.classify.apply_features(extract_features, sampl_tweets)
    print("Training..........")
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("done training")
    print("Saving model")
    f = open('my_classifier_80000.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
    print("Done saving model")

    f = open('my_classifier_80000.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    print("Done loading model")
    testing=[]
    with open("testdata.manual.2009.06.14.csv") as f:
        alllines=csv.reader(f)
        for line in alllines:
            text=remove_emoji(clean_tweet(line[5]))
            label=line[0]
            words_filtered = [e.lower() for e in text.split() if len(e) >= 3]
            testing.append((words_filtered, label))
    count=0
    for test in testing:
        if(test[1]==classifier.classify(extract_features(test[0]))):
            count+=1.0
    print("accuracy:", count/len(testing))

def main():
    train()

if __name__ == "__main__":
    # calling main function
    main()
