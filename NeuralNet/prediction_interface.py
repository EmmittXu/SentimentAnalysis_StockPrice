#Auther Guowei Xu
#August 25th, 2017
import cPickle
import numpy as np
import tensorflow as tf
import re

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

#Vectorize string text to word2vec matrix
def vectorize(sentence, word2vec, word_idx_map):
    sentence = clean_str(sentence)
    embedded=np.zeros(shape=[30,300], dtype=np.float64)
    i=0
    for word in sentence.split():
        if i>=30:
            break
        if(word in word_idx_map):
            embedded[i]=word2vec[word_idx_map[word]]
            i+=1
    return embedded.reshape(-1,9000)

def build_hidden_layers(input, n_neurons, n_layers):
    is_training=tf.placeholder_with_default(False, shape=(), name='training')
    for layer in range(n_layers-1):
        input=tf.layers.dropout(input, 0.5, training=is_training)
        input=tf.layers.dense(input, n_neurons, activation=tf.nn.elu,\
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        input=tf.layers.batch_normalization(input, momentum=0.99, training=is_training)
    return input

def initialization():
    print("initializing model...")
    x = cPickle.load(open("Data/imdb-train-val-test.pickle", "rb"))
    word2vec, word_idx_map = x[1], x[2]

    he_init = tf.contrib.layers.variance_scaling_initializer()
    X=tf.placeholder(tf.float32, shape=(None, 30*300), name='tweets')
    y=tf.placeholder(tf.int64, shape=(None), name='sentiment')
    output=build_hidden_layers(X, 150, 5)
    logits=tf.layers.dense(output, 2, activation=tf.nn.softmax, kernel_initializer=he_init)
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss=tf.reduce_mean(xentropy, name='loss')
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
    training_op=optimizer.minimize(loss)
    correct=tf.nn.in_top_k(logits, y, 1)
    accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))
    Y_prob = tf.nn.softmax(logits, name="Y_proba")
    init=tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./new_dnn_sentiment.ckpt")

        while(1):
            sentence = raw_input("Please enter a sentence('exit' to stop):")
            if(sentence=="exit"):
                break
            else:
                print("You entered {}".format(sentence))
                x=vectorize(sentence, word2vec, word_idx_map)
                probability = Y_prob.eval(feed_dict={X:x})
                if(probability[0][0]>probability[0][1]):
                    print("Negative")
                else:
                    print("Positive")
initialization()
