import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import requests
import openai

# Predefined list of stocks for Top 10 analysis
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "BAC"]

# OpenAI API Key
openai.api_key = "your_openai_api_key"  # Replace with your OpenAI API key

# Helper Function: Fetch News Articles
def fetch_news_articles(stock_name, limit=5):
    api_key = "your_newsapi_key"  # Replace with your NewsAPI key
    query = f"{stock_name} stock"
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={limit}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return []
    
    articles = response.json().get("articles", [])
    return [{"title": article["title"], "description": article["description"]} for article in articles]

# Helper Function: Summarize with OpenAI
def summarize_with_gpt(content):
    if not content:
        return "No recent chatter available", 0  # No content to summarize
    
    prompt = f"Summarize the following news articles and provide sentiment analysis:\n{content}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
    )
    summary = response["choices"][0]["message"]["content"]
    return summary

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
            articles = fetch_news_articles(stock, limit=5)
            content = " ".join([f"{article['title']} {article['description']}" for article in articles])
            news_summary = summarize_with_gpt(content)

            recent_growth = (data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30] * 100
            future_growth = (predictions[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100

            results.append({
                "Stock": stock,
                "Recent Growth (%)": recent_growth,
                "Predicted Growth (%)": future_growth,
                "Summary": news_summary,
                "Reason": f"Predicted growth: {future_growth:.2f}% based on trends and sentiment."
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

# Fix for ArrowTypeError: Truncate long strings and ensure string types
if "Summary" in top_stocks.columns:
    top_stocks["Summary"] = top_stocks["Summary"].apply(lambda x: x[:200] + "..." if len(x) > 200 else x)
top_stocks = top_stocks.astype(str)  # Convert all columns to strings
st.table(top_stocks)

# Stock Selection Section
st.sidebar.header("Stock Selection")
selected_stock = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
days = st.sidebar.slider("Prediction Period (Days):", 1, 10, 5)

# Fetch Stock Data
data_load_state = st.text("Loading stock data...")
stock_data = get_stock_data(selected_stock)
data_load_state.text("Stock data loaded successfully!")

# Fetch News and Summarize for Selected Stock
articles = fetch_news_articles(selected_stock, limit=5)
content = " ".join([f"{article['title']} {article['description']}" for article in articles])
news_summary = summarize_with_gpt(content)

# Display Stock Chart
st.subheader(f"Stock Price Data for {selected_stock}")
predictions = predict_prices(stock_data, days)
prediction_dates = [stock_data['Date'].iloc[-1] + timedelta(days=i+1) for i in range(days)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Historical Prices'))
fig.add_trace(go.Scatter(x=prediction_dates, y=predictions, mode='lines', name='Predicted Prices', line=dict(dash='dot')))
fig.update_layout(title=f"{selected_stock} Stock Prices", xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig)

# Display News Summary
st.subheader("Open-Source Chatter Summary")
st.write(f"Summary: {news_summary}")

# Footer
st.write("---")
st.write("Dashboard powered by AI, NewsAPI, and OpenAI GPT-4.")
