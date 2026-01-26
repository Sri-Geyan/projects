import streamlit as st
import pandas as pd
import yfinance as yf
from src.sentiment import get_sentiment

st.title("Financial News Sentiment Analyzer")

news_text = st.text_area("Enter Financial News Headline")

ticker = st.text_input("Stock Ticker", "AAPL")

if st.button("Analyze"):
    sentiment = get_sentiment(news_text)
    st.write("Sentiment Score:", sentiment)

    stock = yf.download(ticker, period="1mo")
    st.line_chart(stock['Close'])