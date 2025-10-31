import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


# ---------- 1️⃣ Prepare Data ----------
def prepare_data(prices, n_lags=5):
    """
    Convert price list into ML-supervised format.
    Uses previous `n_lags` days to predict next day's price.
    """
    prices = np.array(prices).flatten()
    X, y = [], []
    for i in range(len(prices) - n_lags):
        X.append(prices[i:i + n_lags])
        y.append(prices[i + n_lags])
    return np.array(X), np.array(y)


# ---------- 2️⃣ Train and Predict ----------
def train_and_predict(prices, n_lags=5):
    """
    Trains Linear Regression and Random Forest on recent data.
    Returns next-day predictions from both models.
    """
    prices = np.array(prices).flatten()
    X, y = prepare_data(prices, n_lags)
    X_pred = prices[-n_lags:].reshape(1, -1)

    # Linear Regression
    lr_model = LinearRegression().fit(X, y)
    lr_pred = lr_model.predict(X_pred)[0]

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    rf_pred = rf_model.predict(X_pred)[0]

    return lr_pred, rf_pred


# ---------- 3️⃣ Plot ML Predictions ----------
def plot_predictions(prices, lr_pred, rf_pred):
    """
    Plot actual vs predicted prices for visual comparison.
    """
    prices = np.array(prices).flatten()
    days = np.arange(len(prices))

    plt.figure(figsize=(10, 5))
    plt.plot(days, prices, label="Actual Closing Prices", color="blue")

    next_day = len(prices)
    plt.scatter(next_day, lr_pred, color="green", marker="o", label="Linear Regression Prediction")
    plt.scatter(next_day, rf_pred, color="red", marker="x", label="Random Forest Prediction")

    plt.xlabel("Days (Recent Data)")
    plt.ylabel("Stock Price (INR)")
    plt.title("Next-Day Stock Price Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------- 4️⃣ Rolling Forecast Evaluation ----------
def rolling_forecast_eval(prices, n_lags=5, test_days=60):
    """
    Rolling window backtest to evaluate prediction accuracy.
    Trains models incrementally to simulate real-world forecasting.
    """
    prices = np.array(prices).flatten()
    train_size = len(prices) - test_days
    train, test = prices[:train_size], prices[train_size:]

    lr_preds, rf_preds, actuals = [], [], []

    for i in range(len(test)):
        end = train_size + i
        X, y = prepare_data(prices[:end], n_lags)
        X_pred = prices[end - n_lags:end].reshape(1, -1)

        lr = LinearRegression().fit(X, y)
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

        lr_preds.append(lr.predict(X_pred)[0])
        rf_preds.append(rf.predict(X_pred)[0])
        actuals.append(test[i])

    stats = {
        "lr": {
            "mae": mean_absolute_error(actuals, lr_preds),
            "mape": mean_absolute_percentage_error(actuals, lr_preds),
        },
        "rf": {
            "mae": mean_absolute_error(actuals, rf_preds),
            "mape": mean_absolute_percentage_error(actuals, rf_preds),
        },
    }

    return stats


# ---------- 5️⃣ Optional Demo ----------
if __name__ == "__main__":
    # Example demo: test with fake data
    np.random.seed(42)
    prices = np.linspace(100, 120, 100) + np.random.normal(0, 1, 100)

    lr_pred, rf_pred = train_and_predict(prices)
    print(f"Linear Regression Prediction: {lr_pred:.2f}")
    print(f"Random Forest Prediction: {rf_pred:.2f}")

    plot_predictions(prices, lr_pred, rf_pred)

    stats = rolling_forecast_eval(prices, n_lags=5, test_days=20)
    print("\nBacktest Results:")
    print(f"Linear Regression → MAE: {stats['lr']['mae']:.2f}, MAPE: {stats['lr']['mape']*100:.2f}%")
    print(f"Random Forest → MAE: {stats['rf']['mae']:.2f}, MAPE: {stats['rf']['mape']*100:.2f}%")
