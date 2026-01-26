import pandas as pd

def load_news(path="/Users/srigeyan/Documents/Programming/Python/financial-news-sentiment/data/news.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_prices(path="/Users/srigeyan/Documents/Programming/Python/financial-news-sentiment/data/prices.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df
