import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import requests

# Predefined list of stocks for Top 10 analysis
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "BAC"]

# Pushshift API URL
PUSHSHIFT_URL = "https://api.pushshift.io/reddit/search/submission/"

# Helper Function: Fetch Reddit Posts
def fetch_reddit_posts(stock_name, limit=10):
    query = f"{stock_name} OR {stock_name} stock OR {stock_name} finance"
    params = {
        "q": query,
        "subreddit": "stocks",
        "size": limit,
        "sort": "desc",
        "sort_type": "created_utc",
    }
    response = requests.get(PUSHSHIFT_URL, params=params)
    
    # Check if the request succeeded
    if response.status_code != 200:
        st.error(f"Error fetching Reddit posts: {response.status_code}")
        return []
    
    posts = response.json().get("data", [])
    if not posts:
        st.warning(f"No Reddit posts found for query: {query}")
    return posts

# Helper Function: Summarize Reddit Chatter
def summarize_reddit(posts):
    if not posts:
        return "No Recent Reddit Chatter", 0  # No posts found
    summaries = [post["title"] for post in posts if "title" in post]
    all_text = " ".join(summaries)
    sentiment = TextBlob(all_text).sentiment.polarity  # Calculate sentiment
    summary = " ".join(summaries[:2])  # Use first two post titles for brevity
    return summary, sentiment

# Helper Function: Sentiment Analysis
@st.cache_resource
def get_sentiment_scores(headlines):
    scores = []
    for headline in headlines:
        analysis = TextBlob(headline)
        scores.append(analysis.sentiment.polarity)
    return np.mean(scores)

# Helper Function: Fetch Stock Data
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    data.reset_index(inplace=True)
    return data

# Helper Function: Price Prediction
def predict_prices(data, days):
    data['Day'] = np.arange(len(data))  # Numeric index for each day
    X = data[['Day']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions

# Helper Function: Analyze Top Stocks
def analyze_top_stocks(horizon):
    results = []
    for stock in STOCK_LIST:
        try:
            data = get_stock_data(stock, period="1y")
            predictions = predict_prices(data, horizon)
            posts = fetch_reddit_posts(stock, limit=10)
            reddit_summary, reddit_sentiment = summarize_reddit(posts)

            recent_growth = (data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30] * 100
            future_growth = (predictions[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100

            results.append({
                "Stock": stock,
                "Recent Growth (%)": recent_growth,
                "Predicted Growth (%)": future_growth,
                "Sentiment": reddit_sentiment,
                "Reddit Chatter": reddit_summary,
                "Reason": f"Predicted growth: {future_growth:.2f}% with positive chatter."
            })
        except Exception as e:
            continue

    results = sorted(results, key=lambda x: x["Predicted Growth (%)"], reverse=True)[:10]
    return pd.DataFrame(results)

# Dashboard Layout
st.title("AI-Powered Stock Trading Dashboard")

# Top Section: Top 10 Growth Stocks
st.subheader("Top 10 Growth Stocks")
horizon = st.slider("Select Growth Horizon (Days):", 1, 30, 7)
top_stocks = analyze_top_stocks(horizon)
st.table(top_stocks)

# Stock Selection Section
st.sidebar.header("Stock Selection")
selected_stock = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
days = st.sidebar.slider("Prediction Period (Days):", 1, 10, 5)

# Fetch Stock Data
data_load_state = st.text("Loading stock data...")
stock_data = get_stock_data(selected_stock)
data_load_state.text("Stock data loaded successfully!")

# Fetch Reddit Chatter for Selected Stock
reddit_posts = fetch_reddit_posts(selected_stock, limit=10)
reddit_summary, reddit_sentiment = summarize_reddit(reddit_posts)

# Fetch News and Sentiment
news_load_state = st.text("Fetching news and sentiment...")
news_headlines = [post["title"] for post in reddit_posts]  # Use Reddit posts as news source
sentiment_score = reddit_sentiment
news_load_state.text("News and sentiment fetched successfully!")

# Display Stock Chart
st.subheader(f"Stock Price Data for {selected_stock}")
predictions = predict_prices(stock_data, days)
prediction_dates = [stock_data['Date'].iloc[-1] + timedelta(days=i+1) for i in range(days)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Historical Prices'))
fig.add_trace(go.Scatter(x=prediction_dates, y=predictions, mode='lines', name='Predicted Prices', line=dict(dash='dot')))
fig.update_layout(title=f"{selected_stock} Stock Prices", xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig)

# Display Reddit Chatter Summary
st.subheader("Reddit Chatter Summary")
st.write(f"Summary: {reddit_summary}")
st.write(f"Sentiment Score: {sentiment_score:.2f}")

# Footer
st.write("---")
st.write("Dashboard powered by AI, Pushshift, and real-time stock data.")
