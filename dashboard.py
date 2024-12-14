import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import openai
import requests

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Predefined list of top investors and their strategies
TOP_INVESTORS = [
    {"name": "Warren Buffett", "strategy": "Focus on intrinsic value and long-term fundamentals."},
    {"name": "Peter Lynch", "strategy": "Invest in what you know and look for undervalued growth stocks."},
    {"name": "Charlie Munger", "strategy": "Focus on quality companies with a strong competitive advantage."},
]

# Predefined list of stock tickers for analysis
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "BAC"]

# Fetch stock data using yfinance
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    data.reset_index(inplace=True)
    return data

# Predict target price using historical data
def predict_target_price(data, days):
    data['Day'] = np.arange(len(data))  # Add a numeric day index
    X = data[['Day']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions[-1]  # Return the last prediction as the target price

# Analyze sentiment for a stock
def analyze_sentiment(stock_name):
    prompt = f"Analyze the sentiment and current chatter about the stock {stock_name}. Provide a summary and indicate whether sentiment is positive, neutral, or negative."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial market analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

# Generate insights from top investors
def generate_investor_insights(stock_name):
    insights = []
    for investor in TOP_INVESTORS:
        prompt = f"Based on {investor['strategy']}, would {investor['name']} recommend buying, holding, or selling {stock_name}? Provide reasoning."
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are {investor['name']}, a renowned investor."},
                    {"role": "user", "content": prompt},
                ],
            )
            insights.append(f"{investor['name']}: {response['choices'][0]['message']['content']}")
        except Exception as e:
            insights.append(f"{investor['name']}: Error generating insights: {str(e)}")
    return "\n".join(insights)

# Analyze the top 10 stocks
def analyze_top_stocks(horizon):
    results = []
    for stock in STOCK_LIST:
        try:
            data = get_stock_data(stock, period="1y")
            projected_price = predict_target_price(data, horizon)
            current_price = data['Close'].iloc[-1]
            analyst_target_price = yf.Ticker(stock).info.get('targetMeanPrice', "N/A")
            sentiment = analyze_sentiment(stock)

            results.append({
                "Stock": stock,
                "Current Price": current_price,
                "Analyst Target Price": analyst_target_price,
                "Projected Price": projected_price,
                "Sentiment": sentiment,
            })
        except Exception as e:
            continue
    return pd.DataFrame(results)

# Real-time API usage
def get_api_usage():
    try:
        response = requests.get(
            "https://api.openai.com/v1/dashboard/billing/usage",
            headers={"Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"},
        )
        if response.status_code == 200:
            usage_data = response.json()
            return f"OpenAI API Usage: {usage_data.get('total_usage', 0)} tokens"
        else:
            return f"Unable to fetch API usage: {response.status_code}"
    except Exception as e:
        return f"Error fetching API usage: {str(e)}"

# Streamlit app layout
st.title("AI-Powered Stock Trading Dashboard")

# Adjustable performance horizon
horizon = st.slider("Select Performance Horizon (Days):", 30, 365, 180)

# Display top 10 stocks
st.subheader("Top 10 Stocks to Buy Now")
top_stocks = analyze_top_stocks(horizon)
st.write(top_stocks)

# Individual stock analysis
st.subheader("Individual Stock Analysis")
stock_picker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
if stock_picker:
    stock_data = get_stock_data(stock_picker)
    investor_insights = generate_investor_insights(stock_picker)
    sentiment = analyze_sentiment(stock_picker)

    st.write(f"Investor Insights for {stock_picker}:")
    st.write(investor_insights)
    st.write(f"Sentiment Analysis for {stock_picker}:")
    st.write(sentiment)

    # Plot historical data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Close Price'))
    st.plotly_chart(fig)

# Footer with API usage
st.write("---")
st.write(get_api_usage())
