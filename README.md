# SentimentAnalysis_StockPrice
Final project release for large data stream processing

A first layer NaiveBayes classifier (trained on Sentiment 140 dataset) to do sentiment analysis on Tweets streamed using Twitter API. A second NaiveBayes classifier to predict stock price trend given sentiment vector as input.  
We also implemented a streaming interface using Spark Streaming to do online stock price prediction.

# Recently built and trained a neural network to do sentiment classification.  
See NeuralNet folder for details. Testing accuracy on 4000 test data is 75%, a boost compared to NaiveBayes classifier.
