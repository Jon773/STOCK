import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import openai
from datetime import timedelta

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Predefined stock tickers (conventional + hidden gems)
CONVENTIONAL_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
HIDDEN_GEMS = ["PLTR", "SHOP", "SQ", "CRWD", "NET"]

# Buy-Hold-Sell Recommendations Mockup (Replace with API as needed)
BUY_HOLD_SELL = {
    "AAPL": {"Buy": 60, "Hold": 30, "Sell": 10},
    "MSFT": {"Buy": 55, "Hold": 35, "Sell": 10},
    "GOOGL": {"Buy": 50, "Hold": 40, "Sell": 10},
    "AMZN": {"Buy": 65, "Hold": 25, "Sell": 10},
    "TSLA": {"Buy": 45, "Hold": 30, "Sell": 25},
    "PLTR": {"Buy": 70, "Hold": 20, "Sell": 10},
    "SHOP": {"Buy": 65, "Hold": 25, "Sell": 10},
    "SQ": {"Buy": 50, "Hold": 30, "Sell": 20},
    "CRWD": {"Buy": 75, "Hold": 15, "Sell": 10},
    "NET": {"Buy": 60, "Hold": 30, "Sell": 10},
}

# Fetch stock data
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    data.reset_index(inplace=True)
    return data

# Predict target price
def predict_target_price(data, days):
    data['Day'] = np.arange(len(data))
    X = data[['Day']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions

# Fetch OpenAI sentiment
def analyze_sentiment(stock_name):
    prompt = f"Summarize sentiment for {stock_name} in 1-2 sentences. Include Bullish, Neutral, and Bearish perspectives."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a concise financial sentiment analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    return response["choices"][0]["message"]["content"]

# Generate Buy-Hold-Sell Gauge
def generate_gauge_chart(stock):
    data = BUY_HOLD_SELL.get(stock, {"Buy": 0, "Hold": 0, "Sell": 0})
    fig = px.pie(
        names=["Buy", "Hold", "Sell"],
        values=[data["Buy"], data["Hold"], data["Sell"]],
        title=f"{stock}: Buy-Hold-Sell Consensus",
        hole=0.6
    )
    return fig

# Identify top stocks
def identify_top_stocks(horizon, stocks):
    results = []
    for stock in stocks:
        data = get_stock_data(stock)
        current_price = data['Close'].iloc[-1]
        projected_price = predict_target_price(data, horizon)[-1]
        sentiment = analyze_sentiment(stock)
        reason_to_buy = "Strong upward trend with a temporary dip."  # Placeholder logic
        results.append({
            "Stock": stock,
            "Current Price": f"${current_price:.2f}",
            "Projected Price": f"${projected_price:.2f}",
            "Sentiment": sentiment,
            "Reason to Buy": reason_to_buy
        })
    return pd.DataFrame(results)

# Backtest model
def backtest_model(data, test_period):
    train_data = data[:-test_period]
    test_data = data[-test_period:]
    projected_prices = predict_target_price(train_data, test_period)
    actual_prices = test_data['Close'].values
    accuracy = 100 - np.mean(np.abs((actual_prices - projected_prices) / actual_prices)) * 100
    return actual_prices, projected_prices, accuracy

# Layout and Dashboard
st.title("AI-Powered Stock Dashboard")
horizon = st.slider("Performance Horizon (Days):", 30, 365, 180)

# Top Stocks
st.subheader("Top 5 Conventional Picks and 5 Hidden Gems")
st.write("### Conventional Stocks")
conventional_df = identify_top_stocks(horizon, CONVENTIONAL_STOCKS)
st.table(conventional_df)

st.write("### Hidden Gems")
hidden_gems_df = identify_top_stocks(horizon, HIDDEN_GEMS)
st.table(hidden_gems_df)

# User-selected Stock Analysis
st.subheader("Individual Stock Analysis")
stock = st.text_input("Enter Stock Ticker:", "AAPL")
if stock:
    data = get_stock_data(stock)
    sentiment = analyze_sentiment(stock)
    st.write(f"**Sentiment for {stock}:** {sentiment}")
    
    # Historical + Projected Graph
    st.write("### Historical and Projected Performance")
    projections = predict_target_price(data, horizon)
    future_dates = [data['Date'].iloc[-1] + timedelta(days=i) for i in range(horizon)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Historical Prices"))
    fig.add_trace(go.Scatter(x=future_dates, y=projections, name="Projected Prices", line=dict(dash="dash")))
    st.plotly_chart(fig)
    
    # Backtest
    if st.button("Back Test"):
        actual, predicted, confidence = backtest_model(data, test_period=150)
        st.write(f"### Backtest Confidence Level: {confidence:.2f}%")
        backtest_fig = go.Figure()
        backtest_fig.add_trace(go.Scatter(y=actual, name="Actual Prices"))
        backtest_fig.add_trace(go.Scatter(y=predicted, name="Predicted Prices", line=dict(dash="dash")))
        st.plotly_chart(backtest_fig)

    # Buy-Hold-Sell Chart
    st.write("### Buy-Hold-Sell Recommendations")
    st.plotly_chart(generate_gauge_chart(stock))

st.write("---")
st.write("AI Dashboard powered by OpenAI and yfinance.")
