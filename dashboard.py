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

# Helper Function: Fetch News Data
def fetch_news_data(stock_name):
    api_key = "your_newsapi_key"  # Replace with your NewsAPI key
    query = f"{stock_name} stock OR {stock_name} finance"
    url = f"https://newsapi.org/v2/everything?q={query}&from={(datetime.now() - timedelta(days=7)).date()}&sortBy=popularity&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [article['title'] for article in articles if 'title' in article]

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

# Helper Function: Analyze Stocks
def analyze_top_stocks(horizon):
    results = []
    for stock in STOCK_LIST:
        try:
            data = get_stock_data(stock, period="1y")
            predictions = predict_prices(data, horizon)
            sentiment_score = get_sentiment_scores(fetch_news_data(stock))

            # Analyze stock performance
            recent_growth = (data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30] * 100
            future_growth = (predictions[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100

            results.append({
                "Stock": stock,
                "Recent Growth (%)": recent_growth,
                "Predicted Growth (%)": future_growth,
                "Sentiment": sentiment_score,
                "Reason": f"Strong growth potential with {future_growth:.2f}% predicted growth and positive sentiment."
            })
        except Exception as e:
            continue

    # Sort by predicted growth and return top 10
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

# Fetch News and Sentiment
news_load_state = st.text("Fetching news and sentiment...")
news_headlines = fetch_news_data(selected_stock)
sentiment_score = get_sentiment_scores(news_headlines)
news_load_state.text("News and sentiment fetched successfully!")

# Display Stock Chart
st.subheader(f"Stock Price Data for {selected_stock}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Historical Prices'))
fig.add_trace(go.Scatter(x=[data_load_state['Date'].iloc[-1] + timedelta(days=i+1) for i in range(days)],
                         y=predict_prices(stock_data, days),
                         mode='lines',
                         name='Predicted Prices',
                         line=dict(dash='dot')))
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

# Footer
st.write("---")
st.write("Dashboard powered by AI and real-time stock data.")
