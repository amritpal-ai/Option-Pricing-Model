
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiment_score(text):
    """
    Calculates the sentiment score for a given text using VADER.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(text)
    return sentiment_dict['compound']

def get_average_sentiment(headlines):
    """
    Calculates the average sentiment score for a list of headlines.
    """
    if not headlines:
        return 0.0
    
    total_score = 0
    for headline in headlines:
        total_score += get_sentiment_score(headline)
        
    return total_score / len(headlines)

import requests

def get_stock_news(stock_symbol):
    """
    Fetches recent news headlines for a stock using Yahoo Finance RSS feed.
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={stock_symbol}&region=IN&lang=en-IN"
    try:
        response = requests.get(url, timeout=5)
        text = response.text
        # Extract <title> tags
        headlines = [line.split("</title>")[0] for line in text.split("<title>")[2:7]]
        return headlines
    except Exception as e:
        print("Error fetching news:", e)
        return []

