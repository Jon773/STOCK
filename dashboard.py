import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import openai
import requests

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Top investors and strategies
TOP_INVESTORS = [
    {"name": "Warren Buffett", "strategy": "Focus on intrinsic value and long-term fundamentals."},
    {"name": "Peter Lynch", "strategy": "Invest in what you know and look for undervalued growth stocks."},
    {"name": "Charlie Munger", "strategy": "Focus on quality companies with a strong competitive advantage."},
]

# Predefined stock tickers
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "BAC"]

# Fetch stock data
def get_stock_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return None

# Predict target price
def predict_target_price(data, days):
    try:
        data['Day'] = np.arange(len(data))
        X = data[['Day']]
        y = data['Close']
        model = LinearRegression()
        model.fit(X, y)
        future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
        predictions = model.predict(future_days)
        return predictions[-1]
    except Exception as e:
        return None

# Fetch OpenAI sentiment analysis
def analyze_sentiment(stock_name):
    prompt = f"Provide sentiment analysis for {stock_name}. Summarize recent investor chatter."
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

# Generate insights for investors
def generate_investor_insights(stock_name):
    insights = []
    for investor in TOP_INVESTORS:
        try:
            prompt = f"Based on {investor['strategy']}, what would {investor['name']} recommend for {stock_name}?"
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

# Analyze top stocks
def analyze_top_stocks(horizon):
    results = []
    for stock in STOCK_LIST:
        data = get_stock_data(stock)
        if data is None:
            continue
        current_price = data['Close'].iloc[-1]
        projected_price = predict_target_price(data, horizon)
        analyst_target_price = yf.Ticker(stock).info.get('targetMeanPrice', "N/A")
        sentiment = analyze_sentiment(stock)
        results.append({
            "Stock": stock,
            "Current Price": current_price,
            "Projected Price": projected_price,
            "Analyst Target Price": analyst_target_price,
            "Sentiment": sentiment,
        })
    return pd.DataFrame(results)

# Get OpenAI API usage
def get_api_usage():
    try:
        response = requests.get(
            "https://api.openai.com/v1/dashboard/billing/usage",
            headers={"Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"},
        )
        if response.status_code == 200:
            usage_data = response.json()
            return f"API Usage: {usage_data.get('total_usage', 0)} tokens used."
        else:
            return "Error fetching API usage."
    except Exception as e:
        return f"Error fetching API usage: {str(e)}"

# App layout
st.title("AI-Powered Stock Dashboard")
horizon = st.slider("Performance Horizon (Days):", 30, 365, 180)

# Top stocks
st.subheader("Top 10 Stocks to Buy Now")
top_stocks = analyze_top_stocks(horizon)
st.table(top_stocks)

# Individual stock analysis
st.subheader("Individual Stock Analysis")
stock = st.text_input("Enter Stock Ticker:", "AAPL")
if stock:
    stock_data = get_stock_data(stock)
    sentiment = analyze_sentiment(stock)
    investor_insights = generate_investor_insights(stock)
    st.write(f"Sentiment for {stock}: {sentiment}")
    st.write(f"Investor Insights for {stock}: {investor_insights}")
    if stock_data is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Price'))
        st.plotly_chart(fig)

# Footer
st.write("---")
st.write(get_api_usage())

