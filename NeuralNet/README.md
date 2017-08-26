# A Tensorflow implementation of neural network to do binary sentiment classification on Tweets.  

Watch a quick demo [here](https://www.youtube.com/watch?v=JS51p-aXDas&feature=youtu.be)

Training data is http://help.sentiment140.com/for-students/, each data is a labled tweets, positive/negative.  
Used pretrained Google [word2vec](https://en.wikipedia.org/wiki/Word2vec) to do word embedding https://code.google.com/archive/p/word2vec/

Build_vocabulary.ipynb loads training data and word2vec model, build vocabulary and word_index_mapping.  
sentiment_analysis_DNN is a fully connected network with various optimization and normalization algorithms.  
I am still working on the CNN implementation. Training is expensive without GPU support. Code is fine, finding a powerful machine is a problem.

# Dependency:
python2.7, tensorflow, numpy, pandas, scikit learn, matplotlib, jupyter notebook
