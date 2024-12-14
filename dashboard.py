import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import openai
from datetime import timedelta

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

# Fetch live price and placeholder target price
def get_live_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        return data['Close'].iloc[-1] if not data.empty else "Unavailable"
    except Exception as e:
        st.error(f"Error fetching live price for {ticker}: {e}")
        return "N/A"

# Fetch OpenAI sentiment analysis
def analyze_sentiment(stock_name, concise=True):
    prompt = f"Provide a {'concise' if concise else 'detailed'} sentiment analysis for {stock_name}. Summarize recent investor chatter."
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

# Generate insights for top investors
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

# Back-test logic for confidence level
def backtest_model(data, test_period):
    try:
        train_data = data[:-test_period]
        test_data = data[-test_period:]
        projected_prices = predict_target_price(train_data, test_period)

        actual_prices = test_data['Close'].values
        accuracy = 100 - np.mean(np.abs((actual_prices - projected_prices) / actual_prices)) * 100

        return accuracy
    except Exception as e:
        st.error(f"Error backtesting model: {e}")
        return None

# Analyze top stocks
def analyze_top_stocks(horizon):
    results = []
    for stock in STOCK_LIST:
        data = get_stock_data(stock)
        if data is None:
            continue
        current_price = get_live_price(stock)
        projected_price = predict_target_price(data, horizon)
        sentiment = analyze_sentiment(stock, concise=True)
        investor_sentiment = generate_investor_insights(stock)
        results.append({
            "Stock": stock,
            "Current Price": current_price,
            "Projected Price": projected_price,
            "Investor Sentiment": investor_sentiment,
            "General Sentiment": sentiment,
        })
    return pd.DataFrame(results)

# App layout
st.title("AI-Powered Stock Dashboard")
horizon = st.slider("Performance Horizon (Days):", 30, 365, 180)

# Top stocks
st.subheader("Top 10 Stocks to Buy Now")
top_stocks = analyze_top_stocks(horizon)
if not top_stocks.empty:
    st.table(top_stocks)
else:
    st.write("No stock data available.")

# Individual stock analysis
st.subheader("Individual Stock Analysis")
stock = st.text_input("Enter Stock Ticker:", "AAPL")
if stock:
    stock_data = get_stock_data(stock)
    sentiment = analyze_sentiment(stock, concise=True)
    investor_insights = generate_investor_insights(stock)
    st.write(f"General Sentiment for {stock}: {sentiment}")
    st.write(f"Investor Insights for {stock}: {investor_insights}")
    if stock_data is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Price'))
        st.plotly_chart(fig)
        if st.button("Back Test"):
            confidence = backtest_model(stock_data, test_period=150)  # Adjustable period
            st.write(f"Confidence Level: {confidence:.2f}%")

# Footer
st.write("---")
st.write("AI Dashboard powered by OpenAI and yfinance.")


