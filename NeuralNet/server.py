#!/usr/bin/env python2.7

"""
Columbia's COMS W4111.001 Introduction to Databases
Example Webserver

To run locally:

  python server.py

Go to http://localhost:8111 in your browser.

A debugger such as "pdb" may be helpful for debugging.
Read about it online.
"""
import socket, sys

import os
import cPickle
import tensorflow as tf
from flask import Flask, request, render_template
from prediction_interface import vectorize, clean_str, build_hidden_layers


print("initializing model...")
x = cPickle.load(open("imdb-train-val-test.pickle", "rb"))
word2vec, word_idx_map = x[1], x[2]

#default_sess=None
sess=tf.Session()
#default_sess=sess
he_init = tf.contrib.layers.variance_scaling_initializer()
X = tf.placeholder(tf.float32, shape=(None, 30 * 300), name='tweets')
y = tf.placeholder(tf.int64, shape=(None), name='sentiment')
output = build_hidden_layers(X, 150, 5)
logits = tf.layers.dense(output, 2, activation=tf.nn.softmax, kernel_initializer=he_init)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
Y_prob = tf.nn.softmax(logits, name="Y_proba")
saver = tf.train.Saver()
saver.restore(sess, "./new_dnn_sentiment.ckpt")
print("Model loaded!")

def predict(sentence):
    x = vectorize(sentence, word2vec, word_idx_map)
    return sess.run(Y_prob, feed_dict={X: x})

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

i=0
@app.route('/', methods=['GET','POST'])
def get_sentence():
    tmp="Result"
    global i
    sentence = request.args.get('sentence')
    if(sentence!=None):
        probability=predict(sentence)
        if (probability[0][0] > probability[0][1]):
            label="Negative"
        else:
            label="Positive"
    if(sentence!=None and label!=None):
        tmp=sentence+": "+label
        i+=1
    return render_template('home.html', text_label=tmp, n_sent=i)

def run(debug=True, threaded=True, host='0.0.0.0', port=8111):
    HOST, PORT = host, port
    print "running on %s:%d" % (HOST, PORT)
    app.run(host=HOST, port=PORT, debug=True, threaded=threaded)

if __name__ == "__main__":
    run()