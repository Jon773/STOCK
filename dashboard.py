import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Simulated Data Generation (Replace with Real Data)
np.random.seed(42)
num_stocks = 5
num_days = 300
dates = pd.date_range(start="2023-01-01", periods=num_days, freq="D")
stock_data = {}

for i in range(num_stocks):
    stock_name = f"Stock_{i+1}"
    prices = np.cumsum(np.random.randn(num_days) * 2 + 1) + 100 + i * 10
    stock_data[stock_name] = prices

data = pd.DataFrame(stock_data, index=dates)
print("--- Simulated Stock Price Data ---")
print(data.head())

# Top Stock Picker based on Recent Performance
recent_performance = data.pct_change().tail(30).mean()
top_stock = recent_performance.idxmax()
print(f"Top Stock Based on Recent 30-Day Performance: {top_stock}")

# Hidden Gems: Stocks with High Future Growth Potential (Prediction Based)
train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)
hidden_gems = {}

for stock in data.columns:
    # Prepare Training Data
    y_train = train_data[stock].values
    X_train = np.arange(len(y_train)).reshape(-1, 1)
    X_test = np.arange(len(y_train), len(y_train) + len(test_data)).reshape(-1, 1)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    lin_reg = LinearRegression()
    rf_reg = RandomForestRegressor(n_estimators=100)
    svr = SVR(kernel='rbf')
    stacking = StackingRegressor([
        ('lin_reg', lin_reg),
        ('rf', rf_reg),
        ('svr', svr)], final_estimator=RandomForestRegressor(n_estimators=50))
    
    # Fit & Predict
    lin_reg.fit(X_train_scaled, y_train)
    rf_reg.fit(X_train_scaled, y_train)
    svr.fit(X_train_scaled, y_train)
    stacking.fit(X_train_scaled, y_train)
    predictions = stacking.predict(X_test_scaled)

    # RMSE Evaluation
    rmse = np.sqrt(mean_squared_error(test_data[stock].values, predictions))
    hidden_gems[stock] = rmse
    
# Identify Hidden Gems with Lowest RMSE
hidden_gem_stock = min(hidden_gems, key=hidden_gems.get)
print(f"Hidden Gem Stock (Lowest Prediction RMSE): {hidden_gem_stock}")

# ARIMA Model for Top Stock
top_stock_data = train_data[top_stock]
top_arima = ARIMA(top_stock_data, order=(5, 1, 0))
top_arima_fit = top_arima.fit()
top_arima_forecast = top_arima_fit.forecast(steps=len(test_data))

# Visualization of Top Stock and Hidden Gem
plt.figure(figsize=(14, 8))
plt.plot(data.index, data[top_stock], label=f"Actual {top_stock} Prices", color="blue")
plt.plot(test_data.index, top_arima_forecast, label=f"ARIMA Prediction for {top_stock}", linestyle="--", color="orange")
plt.title(f"Top Stock ({top_stock}) - ARIMA Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Hidden Gem Visualization
plt.figure(figsize=(14, 8))
hidden_gem_actual = data[hidden_gem_stock]
plt.plot(data.index, hidden_gem_actual, label=f"Actual {hidden_gem_stock} Prices", color="green")
plt.title(f"Hidden Gem Stock: {hidden_gem_stock}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

print("--- Final Results ---")
print(f"Top Performing Stock: {top_stock}")
print(f"Hidden Gem Stock: {hidden_gem_stock}")
