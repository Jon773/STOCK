import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import requests

# Helper Function: Sentiment Analysis
@st.cache_resource
def get_sentiment_scores(headlines):
    scores = []
    for headline in headlines:
        analysis = TextBlob(headline)
        scores.append(analysis.sentiment.polarity)  # Polarity ranges from -1 (negative) to +1 (positive)
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
    url = f"https://newsapi.org/v2/everything?q={stock_name}&from={(datetime.now() - timedelta(days=7)).date()}&sortBy=popularity&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [article['title'] for article in articles if 'title' in article]

# Helper Function: Price Prediction
def predict_prices(data, days):
    # Prepare data for prediction
    data['Day'] = np.arange(len(data))  # Numeric index for each day
    X = data[['Day']]  # Features
    y = data['Close']  # Target variable (closing prices)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions

# Dashboard Layout
st.title("Jonathan's AI-Powered Stock Dashboard v3")

# Top Section for Stock Selection
col1, col2 = st.columns([2, 2])
with col1:
    selected_stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")
with col2:
    days = st.slider("Prediction Period (Days):", 1, 10, 5)

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

# Price Prediction
st.subheader(f"Price Prediction for the Next {days} Days")
predictions = predict_prices(stock_data, days)
prediction_dates = [stock_data['Date'].iloc[-1] + timedelta(days=i+1) for i in range(days)]
prediction_df = pd.DataFrame({'Date': prediction_dates, 'Predicted Price': predictions})

# Display Predictions
st.write(prediction_df)

# Plot Predictions
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Historical Prices'))
fig_pred.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Predicted Price'], mode='lines', name='Predicted Prices'))
fig_pred.update_layout(title="Historical and Predicted Prices", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig_pred)

# Footer
st.write("---")
st.write("Dashboard powered by AI and real-time stock data.")

