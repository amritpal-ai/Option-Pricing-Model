# ---------- Imports ----------
import os
import io
import base64
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
from payoff_simulator import plot_payoff_base64
from predictor import train_and_predict
from sentiment_analyzer import get_average_sentiment, get_stock_news



# ---------- Black-Scholes Formula ----------
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
             r * K * np.exp(-r*T) * (norm.cdf(d2) if option_type.lower() == 'call' else norm.cdf(-d2)))
    rho = K * T * np.exp(-r*T) * (norm.cdf(d2) if option_type.lower() == 'call' else -norm.cdf(-d2))

    return price, delta, gamma, vega, theta, rho


from predictor import train_and_predict, rolling_forecast_eval
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from predictor import train_and_predict, rolling_forecast_eval
from sentiment_analyzer import get_stock_news, get_average_sentiment


def run_option_model(stock_symbol, strike, expiry_date, option_type):
    # Ensure correct NSE format
    if not stock_symbol.endswith(".NS"):
        stock_symbol += ".NS"

    # Fetch last 6 months of data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)

    data = yf.download(stock_symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if data.empty:
        raise ValueError(f"No data found for {stock_symbol}. Check symbol and retry.")

    close_prices = data['Close']
    spot_price = float(close_prices.iloc[-1])

    # Volatility (annualized)
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    vol_daily = log_returns.std()
    vol_annual = float(vol_daily * np.sqrt(252))

    # Time to expiry
    expiry_datetime = datetime.strptime(expiry_date, "%Y-%m-%d")
    T = (expiry_datetime - datetime.today()).days / 365.0
    if T <= 0:
        raise ValueError("Expiry must be a future date.")

    # Risk-free rate
    r = 0.06

    # --- Black-Scholes Base Model ---
    price, delta, gamma, vega, theta, rho = black_scholes_price(
        spot_price, strike, T, r, vol_annual, option_type
    )

    # --- Machine Learning Predictions ---
    recent_prices = close_prices[-90:].values
    lr_pred, rf_pred = train_and_predict(recent_prices, n_lags=5)

    bs_price_lr, *_ = black_scholes_price(lr_pred, strike, T, r, vol_annual, option_type)
    bs_price_rf, *_ = black_scholes_price(rf_pred, strike, T, r, vol_annual, option_type)

    df_results = pd.DataFrame({
        "Model": ["Black-Scholes (today)", "Linear Regression (pred)", "Random Forest (pred)"],
        "Spot": [spot_price, lr_pred, rf_pred],
        "Option Price": [price, bs_price_lr, bs_price_rf]
    })

    # --- 🔍 Rolling Forecast Backtest ---
    try:
        backtest_stats = rolling_forecast_eval(close_prices.values, n_lags=5, test_days=60)
        print("\n--- ML Backtest ---")
        print(f"Linear Regression → MAE: {backtest_stats['lr']['mae']:.2f}, MAPE: {backtest_stats['lr']['mape']*100:.2f}%")
        print(f"Random Forest → MAE: {backtest_stats['rf']['mae']:.2f}, MAPE: {backtest_stats['rf']['mape']*100:.2f}%")
    except Exception as e:
        print("Backtest skipped:", e)
        backtest_stats = None

    # --- 📰 Sentiment Analysis Integration ---
    headlines = get_stock_news(stock_symbol)
    avg_sentiment = get_average_sentiment(headlines)

    # --- 🧮 Sentiment Adjustment (optional, slight boost/reduction) ---
    sentiment_factor = 1 + (avg_sentiment * 0.05)  # 5% impact scaling
    lr_pred_adj = lr_pred * sentiment_factor
    rf_pred_adj = rf_pred * sentiment_factor

    # Adjusted prices
    bs_price_lr_adj, *_ = black_scholes_price(lr_pred_adj, strike, T, r, vol_annual, option_type)
    bs_price_rf_adj, *_ = black_scholes_price(rf_pred_adj, strike, T, r, vol_annual, option_type)

    # --- Return Everything to UI ---
    return {
        "stock": stock_symbol,
        "spot": spot_price,
        "vol": vol_annual,
        "T": T,
        "r": r,
        "strike": strike,
        "option_type": option_type,
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "model": "Black-Scholes",
        "ml_preds": {
            "lr_pred": lr_pred_adj,
            "rf_pred": rf_pred_adj,
            "bs_price_lr": bs_price_lr_adj,
            "bs_price_rf": bs_price_rf_adj,
        },
        "backtest_stats": backtest_stats,
        "comparison_table": df_results.to_html(classes="table table-striped", index=False),
        "close_prices": close_prices,
        "sentiment": avg_sentiment,   # ✅ UI will now display sentiment bar
        "headlines": headlines        # ✅ UI will show top headlines
    }




# ---------- Greeks Generator ----------
def get_greeks(result_dict):
    K = result_dict["strike"]
    spot = result_dict["spot"]
    T = result_dict["T"]
    r = result_dict["r"]
    sigma = result_dict["vol"]
    option_type = result_dict["option_type"]

    K_range = np.linspace(K * 0.8, K * 1.2, 50)
    deltas, gammas, vegas, thetas, rhos = [], [], [], [], []

    for k in K_range:
        _, d, g, v, t, r_ = black_scholes_price(spot, k, T, r, sigma, option_type)
        deltas.append(d)
        gammas.append(g)
        vegas.append(v)
        thetas.append(t)
        rhos.append(r_)

    return {
        "K_range": K_range.tolist(),
        "delta": deltas,
        "gamma": gammas,
        "vega": vegas,
        "theta": thetas,
        "rho": rhos
    }


# ---------- Plot Greeks (Base64) ----------
def plot_greeks(greeks, stock_symbol):
    K_range = greeks["K_range"]

    fig, axs = plt.subplots(5, 1, figsize=(8, 18), sharex=True)

    axs[0].plot(K_range, greeks["delta"], color='blue')
    axs[0].set_ylabel('Delta')
    axs[0].set_title(f'Greeks vs Strike Price for {stock_symbol}')

    axs[1].plot(K_range, greeks["gamma"], color='green')
    axs[1].set_ylabel('Gamma')

    axs[2].plot(K_range, greeks["vega"], color='red')
    axs[2].set_ylabel('Vega')

    axs[3].plot(K_range, greeks["theta"], color='purple')
    axs[3].set_ylabel('Theta')

    axs[4].plot(K_range, greeks["rho"], color='orange')
    axs[4].set_ylabel('Rho')
    axs[4].set_xlabel('Strike Price')

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf-8")
    plt.close(fig)
    return plot_url


def plot_stock_history(stock_symbol, close_prices, lr_pred=None, rf_pred=None):
    import os
    import matplotlib.pyplot as plt

    # ✅ Get absolute path for static directory
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    os.makedirs(static_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(close_prices.index, close_prices.values, label="Closing Price", color="blue")
    close_prices.rolling(20).mean().plot(label="20-day MA", color="orange")
    close_prices.rolling(50).mean().plot(label="50-day MA", color="green")

    # ✅ Add ML predictions
    if lr_pred is not None:
        plt.scatter(close_prices.index[-1], lr_pred, color="red", label="LR Predicted", marker="o")
    if rf_pred is not None:
        plt.scatter(close_prices.index[-1], rf_pred, color="purple", label="RF Predicted", marker="x")

    plt.title(f"{stock_symbol} - Last 6 Months Price History")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # ✅ Use full path to save image
    filename = os.path.join(static_dir, f"{stock_symbol}_history.png")
    plt.savefig(filename)
    plt.close()

    # ✅ Return relative path (for Flask templates)
    return f"static/{stock_symbol}_history.png"

