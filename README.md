# SentimentAnalysis_StockPrice
Final project release for large data stream processing

A first layer NaiveBayes classifier (trained on Sentiment 140 dataset) to do sentiment analysis on Tweets streamed using Twitter API. A second NaiveBayes classifier to predict stock price trend given sentiment vector as input.  
We also implemented a streaming interface using Spark Streaming to do online stock price prediction.

Main files:  
nb_train_test.py trains and tests the first NaiveBayes classifier(sentiment classifier).  
sen_stock.py trains and tests the second NaiveBayes classifier(stock price classifier).  
getTweets.py pulls tweets using Twitter API.  
stream.py and tcp-server.py are the implementation of the online stock price prediction.  

# Recently built and trained a neural network to do sentiment classification.  
See NeuralNet folder for details. Testing accuracy on 4000 test data is 76%, outperformed previous NaiveBayes classifier by 5%~10%.
