Training data is http://help.sentiment140.com/for-students/  
Used pretrained Google word2vec to do word embedding https://code.google.com/archive/p/word2vec/

Build_vocabulary.ipynb loads training data and word2vec model, build vocabulary and word_index_mapping.  
sentiment_analysis_DNN is a feed forward model with various optimization and normalization algorithms.  
I am still working on the CNN implementation. Training is expensive without GPU support.
