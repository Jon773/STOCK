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
        st.error(f"Error fetching data for {ticker}: {e}")
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
        st.error(f"Error predicting target price: {e}")
        return None

# Fetch analyst target price with error handling
def get_analyst_target_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('targetMeanPrice', "N/A")
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error for {ticker}: {e}")
        return "N/A"
    except Exception as e:
        st.error(f"Error fetching target price for {ticker}: {e}")
        return "N/A"

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
       

