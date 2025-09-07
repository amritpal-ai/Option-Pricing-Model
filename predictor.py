# predictor.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ---------- Helper: Create dataset ----------
def prepare_data(prices, n_lags=5):
    """
    Convert closing prices into supervised ML dataset.
    Example: if n_lags=5, use last 5 days to predict next day's price.
    """
    prices = np.array(prices).flatten()   # force 1D
    X, y = [], []
    for i in range(len(prices) - n_lags):
        X.append(prices[i:i+n_lags])      # this will be length n_lags
        y.append(prices[i+n_lags])        # next day's price
    return np.array(X), np.array(y)

# ---------- Train Models & Predict ----------
def train_and_predict(prices, n_lags=5):
    """
    Train Linear Regression & Random Forest on price history
    and return both predictions for the next day.
    """
    prices = np.array(prices).flatten()   # force 1D
    X, y = prepare_data(prices, n_lags)

    # Last sequence of n_lags days → input for prediction
    X_pred = prices[-n_lags:].reshape(1, -1)

    # ---------- Linear Regression ----------
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    lr_pred = lr_model.predict(X_pred)[0]

    # ---------- Random Forest ----------
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_pred = rf_model.predict(X_pred)[0]

    return lr_pred, rf_pred


import matplotlib.pyplot as plt

def plot_predictions(prices, lr_pred, rf_pred):
    """
    Plot recent stock prices and predicted next-day prices
    from Linear Regression and Random Forest.
    """
    prices = np.array(prices).flatten()
    days = np.arange(len(prices))

    plt.figure(figsize=(10, 5))
    plt.plot(days, prices, label="Actual Closing Prices", color="blue")

    # Predicted next day is after the last point
    next_day = len(prices)
    plt.scatter(next_day, lr_pred, color="green", marker="o", label="LR Prediction")
    plt.scatter(next_day, rf_pred, color="red", marker="x", label="RF Prediction")

    plt.xlabel("Days (last n days)")
    plt.ylabel("Price")
    plt.title("Stock Price Prediction (Next Day)")
    plt.legend()
    plt.grid(True)
    plt.show()

