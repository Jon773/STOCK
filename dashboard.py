import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import requests
from datetime import datetime, timedelta

# Helper Function: Sentiment Analysis
@st.cache_resource
def get_sentiment_scores(headlines):
    try:
        # Load the model and tokenizer locally
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        
        # Process headlines
        scores = []
        for headline in headlines:
            result = sentiment_model(headline)[0]
            scores.append(1 if result['label'] == 'POSITIVE' else -1)
        return np.mean(scores)
    except Exception as e:
        st.error(f"Failed to analyze sentiment: {e}")
        return 0  # Default to neutral sentiment if there's an issue

# Helper Function: Fetch Stock Data
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    data.reset_index(inplace=True)
    return data

# Helper Function: Fetch News Data
def fetch_news_data(stock_name):
    api_key = "your_newsapi_key"  # Replace with your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={stock_name}&from={(datetime.now() - timedelta(days=7)).date()}&sortBy=popularity&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [article['title'] for article in articles if 'title' in article]

# Dashboard Layout
st.title("AI-Powered Stock Trading Dashboard")

# Sidebar Configuration
st.sidebar.header("Stock Selection")
selected_stock = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
days = st.sidebar.slider("Prediction Period (Days):", 1, 10, 5)

# Fetch Stock Data
data_load_state = st.text("Loading stock data...")
stock_data = get_stock_data(selected_stock)
data_load_state.text("Stock data loaded successfully!")

# Fetch News and Sentiment
news_load_state = st.text("Fetching news and sentiment...")
news_headlines = fetch_news_data(selected_stock)
sentiment_score = get_sentiment_scores(news_headlines)
news_load_state.text("News and sentiment fetched successfully!")

# Display Stock Chart
st.subheader(f"Stock Price Data for {selected_stock}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Close Price'))
fig.update_layout(title=f"{selected_stock} Stock Prices", xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig)

# Display Sentiment Score
st.subheader("Sentiment Analysis")
st.write(f"Average Sentiment Score for {selected_stock}: {sentiment_score:.2f}")

if sentiment_score > 0:
    st.success("Overall Positive Sentiment!")
elif sentiment_score < 0:
    st.error("Overall Negative Sentiment!")
else:
    st.warning("Neutral Sentiment")

# Display News Headlines
st.subheader("Recent News Headlines")
for headline in news_headlines[:5]:
    st.write(f"- {headline}")

# Placeholder for LSTM Prediction
st.subheader("Price Prediction (LSTM - Placeholder)")
st.write("Predictions will be displayed here in the next version!")

# Footer
st.write("---")
st.write("Dashboard powered by AI and real-time stock data.")
