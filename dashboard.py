import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import openai  # For ChatGPT API

# Set up OpenAI API key (replace 'YOUR_API_KEY' with your OpenAI key)
openai.api_key = "YOUR_API_KEY"

# Function to get sentiment or chatter summary using OpenAI ChatGPT
def get_summary(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use 'text-davinci-003' for older API versions
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return "Sentiment unavailable due to API error."


# Streamlit App Title
st.title("📈 Stock Picker, Sentiment, & Projections Dashboard")

# Simulated Data Generation
np.random.seed(42)
num_stocks = 5
num_days = 300
dates = pd.date_range(start="2023-01-01", periods=num_days, freq="D")
stock_data = {f"Stock_{i+1}": np.cumsum(np.random.randn(num_days) * 2 + 1) + 100 + i * 10 for i in range(num_stocks)}
data = pd.DataFrame(stock_data, index=dates)

# Display Data
st.header("📊 Simulated Stock Data")
st.write(data.head())

# Sidebar: User Input for Stock and Projection Period
st.sidebar.header("🔍 Stock Lookup & Projections")
specific_stock = st.sidebar.selectbox("Select a Stock:", options=data.columns)
projection_period = st.sidebar.slider("Select Projection Period (Months):", 1, 12, 6)

# Top Performing Stock
st.header("🏆 Top Performing Stock")
recent_performance = data.pct_change().tail(30).mean()
top_stock = recent_performance.idxmax()
current_price = data[top_stock].iloc[-1]
target_price = current_price * (1 + np.random.uniform(0.05, 0.15))

# Get Sentiment and Chatter for Top Stock
sentiment_prompt = f"Summarize investor sentiment for {top_stock} based on recent market performance."
chatter_prompt = f"Summarize public chatter and discussions about {top_stock} on social media and forums."

top_stock_sentiment = get_summary(sentiment_prompt)
top_stock_chatter = get_summary(chatter_prompt)

st.write(f"**Top Stock Based on 30-Day Performance:** {top_stock}")
st.write(f"**Current Price:** ${current_price:.2f}")
st.write(f"**Target Price (Next {projection_period} Months):** ${target_price:.2f}")
st.write(f"**Investor Sentiment:** {top_stock_sentiment}")
st.write(f"**Public Chatter:** {top_stock_chatter}")

# Hidden Gems Finder
st.header("💎 Hidden Gems Finder")
train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)
hidden_gems = {}

for stock in data.columns:
    y_train = train_data[stock].values
    X_train = np.arange(len(y_train)).reshape(-1, 1)
    X_test = np.arange(len(y_train), len(y_train) + len(test_data)).reshape(-1, 1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    stacking = StackingRegressor([
        ('lin_reg', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=50)),
        ('svr', SVR(kernel='rbf'))
    ], final_estimator=RandomForestRegressor(n_estimators=30))

    stacking.fit(X_train_scaled, y_train)
    predictions = stacking.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(test_data[stock].values, predictions))
    hidden_gems[stock] = rmse

hidden_gem_stock = min(hidden_gems, key=hidden_gems.get)
st.write(f"**Hidden Gem Stock (Lowest Prediction RMSE):** {hidden_gem_stock}")

# ARIMA Forecast for Selected Stock
st.header(f"📈 {specific_stock} Analysis & Projections")
if specific_stock in data.columns:
    stock_data_selected = data[specific_stock]
    train, test = train_test_split(stock_data_selected, test_size=0.3, shuffle=False)

    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=projection_period * 30)

    # Sentiment and Chatter for Selected Stock
    stock_sentiment = get_summary(f"Summarize investor sentiment for {specific_stock}.")
    stock_chatter = get_summary(f"Summarize public chatter about {specific_stock}.")

    st.write(f"**Investor Sentiment:** {stock_sentiment}")
    st.write(f"**Public Chatter:** {stock_chatter}")
    st.write("**Model-Based Future Projection:** Based on ARIMA model predictions.")

    # Visualization
    fig, ax = plt.subplots()
    ax.plot(train.index, train, label="Training Data", color="blue")
    ax.plot(test.index, test, label="Test Data", color="green")
    ax.plot(pd.date_range(test.index[-1], periods=len(forecast)), forecast, label="Forecast", color="orange")
    ax.set_title(f"{specific_stock} - ARIMA Prediction")
    ax.legend()
    st.pyplot(fig)
else:
    st.write("Stock not found. Please select a valid stock.")
