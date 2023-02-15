
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import snscrape.modules.twitter as sntwitter
from datetime import date, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class setnimentAnalysis:
    """
    This class is used to perform sentiment analysis on a given query.
    
    Parameters:
        query (str or list of str): The query to search for in Twitter. Can be a single string or a list of strings.
    
    Attributes:
        query (str or list of str): The query to search for in Twitter. Can be a single string or a list of strings.
        data (pd.DataFrame): A DataFrame containing the collected Twitter data, with columns for 'Datetime', 'Tweet Id', 'Text', and 'Username'.
    
    Methods:
        get_twitter_stock_data: Get the Twitter stock data for a given query.
        get_sentiment: Get the sentiment of a given query.
        get_sentiment_by_date: Get the sentiment of a given query by date.
    """
    def __init__(self, query):
        self.query = query
        self.data = self.get_twitter_stock_data(self.query)

    def get_twitter_stock_data(stock_list):
        """
        Get the Twitter stock data for a given query.

        Parameters:
            query (str or list of str): The query to search for in Twitter. Can be a single string or a list of strings.

        Returns:
            pd.DataFrame: A DataFrame containing the collected Twitter data, with columns for 'Datetime', 'Tweet Id', 'Text', and 'Username'.

        Example:
            >>> query = 'AAPL'
            >>> get_twitter_stock_data(query)

            Datetime                  Tweet Id  ... Username
            0       2021-03-01 23:59:59  136628...  ...  1stAmer...
            1       2021-03-01 23:59:59  136628...  ...  1stAmer...
            2       2021-03-01 23:59:59  136628...  ...  1stAmer...
            3       2021-03-01 23:59:59  136628...  ...  1stAmer...

            [4 rows x 4 columns]
        """
        query_data = []
        stock_tweet_data = pd.DataFrame(columns=['Datetime', 'Tweet Id', 'Text', 'Username'], index=None)
        if type(stock_list) == str:
            stock_list = [stock_list]

        for stock in stock_list:
            for index, tweet in enumerate(sntwitter.TwitterSearchScraper(stock + " since:" + str(date.today() - timedelta(5))).get_items()):
                query_data.append([tweet.date, tweet.id, tweet.rawContent, tweet.user.username])
                stock_data = pd.DataFrame(query_data, columns=['Datetime', 'Tweet Id', 'Text', 'Username'], index=None)
                stock_tweet_data = pd.concat([stock_tweet_data , stock_data ])
                query_data.clear()
        return stock_tweet_data

    def get_sentiments(self, tweets):
        """
        Get the sentiment of a given tweets.

        Parameters:
            tweets (list of str): The tweets to get the sentiment of.

        Returns:
            list of float: A list of sentiment scores for each tweet.

        Example:
            >>> tweets = ['I love this!', 'I hate this!']
            >>> get_sentiments(tweets)
            [0.6369, -0.5719]
        """
        analyzer = SentimentIntensityAnalyzer()
        tweet_list = []
        sentiment_list = []
        counter = 0
        for tweet in tweets:
            text = tweet.full_text # get the text of the tweet
            sentiment = analyzer.polarity_scores(text)['compound'] # get the sentiment score
            tweet_list.append(text)
            sentiment_list.append(sentiment)
            counter += 1
            if counter >= 2500:
                break
        return tweet_list, sentiment_list
    
    def plot_sentiments_over_time(self,tweets):
        """
        Plot the sentiment of a given query over time.

        Parameters:
            query (str or list of str): The query to search for in Twitter. Can be a single string or a list of strings.
        """

        tweet_list, sentiment_list = self.get_sentiments(tweets)
        time_stamps = [tweet.created_at for tweet in tweets]
        time_sentiment = {}
        for i in range(len(time_stamps)):
            timestamp = time_stamps[i].replace(minute=0, second=0) # round down to the nearest 30-minute interval
            if timestamp in time_sentiment:
                time_sentiment[timestamp]['sentiment'].append(sentiment_list[i])
            else:
                time_sentiment[timestamp] = {'sentiment': [sentiment_list[i]]}
        for timestamp in time_sentiment:
            time_sentiment[timestamp]['average_sentiment'] = np.mean(time_sentiment[timestamp]['sentiment'])
        plt.figure(figsize=(12,6))
        plt.plot(list(time_sentiment.keys()), [time_sentiment[x]['average_sentiment'] for x in time_sentiment])
        plt.xlabel('Time')
        plt.ylabel('Sentiment Score')
        plt.title('Sentiment Analysis of Tweets about Python')
        plt.show()

    def get_sentiments_textblob(tweets):
        """
        Get the sentiment of a given tweets.

        Parameters:
            tweets (list of str): The tweets to get the sentiment of.

        Returns:
            list of float: A list of sentiment scores for each tweet.

        Example:
            >>> tweets = ['I love this!', 'I hate this!']
            >>> get_sentiments(tweets)
            [0.6369, -0.5719]
        """
        tweet_list = []
        sentiment_list = []
        counter = 0
        for tweet in tweets:
            text = tweet.full_text # get the text of the tweet
            sentiment = TextBlob(text).sentiment.polarity # get the sentiment score
            tweet_list.append(text)
            sentiment_list.append(sentiment)
            counter += 1
            if counter >= 2500:
                break
        return tweet_list, sentiment_list

    def plot_sentiments_over_time_textblob(self,tweets):
        """
        Plot the sentiment of a given query over time.

        Parameters:
            query (str or list of str): The query to search for in Twitter. Can be a single string or a list of strings.
        """

        tweet_list, sentiment_list = self.get_sentiments_textblob(tweets)
        time_stamps = [tweet.created_at for tweet in tweets]
        time_sentiment = {}
        for i in range(len(time_stamps)):
            timestamp = time_stamps[i].replace(minute=0, second=0) # round down to the nearest 30-minute interval
            if timestamp in time_sentiment:
                time_sentiment[timestamp]['sentiment'].append(sentiment_list[i])
            else:
                time_sentiment[timestamp] = {'sentiment': [sentiment_list[i]]}
        for timestamp in time_sentiment:
            time_sentiment[timestamp]['average_sentiment'] = np.mean(time_sentiment[timestamp]['sentiment'])
        plt.figure(figsize=(12,6))
        plt.plot(list(time_sentiment.keys()), [time_sentiment[x]['average_sentiment'] for x in time_sentiment])
        plt.xlabel('Time')
        plt.ylabel('Sentiment Score')
        plt.title('Sentiment Analysis of Tweets about Python')
        plt.show()
        