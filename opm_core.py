# ---------- Imports ----------
import os
import io
import base64
import requests
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

from payoff_simulator import plot_payoff_base64
from predictor import train_and_predict, rolling_forecast_eval
from sentiment_analyzer import get_average_sentiment, get_stock_news


# ---------- AlphaVantage Fallback ----------
ALPHA_API_KEY = "KASZF271MGMR246C"

def fetch_alpha_vantage_history(symbol):
    """
    Fetch closing price history using AlphaVantage (fallback for Render).
    """
    url = (
        f"https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}"
        f"&outputsize=compact&apikey={ALPHA_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "Time Series (Daily)" not in data:
            print("AlphaVantage error:", data)
            return None

        ts = data["Time Series (Daily)"]
        dates, closes = [], []

        for date_str, row in ts.items():
            dates.append(date_str)
            closes.append(float(row["4. close"]))

        df = pd.DataFrame({"Close": closes}, index=pd.to_datetime(dates))
        df = df.sort_index()

        return df["Close"]

    except Exception as e:
        print("AlphaVantage fetch failed:", e)
        return None


# ---------- Smart NSE Detection ----------
INDIAN_LIST = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "HDFC",
    "SBIN", "KOTAKBANK", "ICICIBANK", "ITC",
    "AXISBANK", "LT", "MARUTI", "SUNPHARMA",
    "BAJAJFINSERV", "ZOMATO", "TATAMOTORS",
    "ADANIPORTS", "ADANIENT", "ULTRACEMCO",
    "BHARTIARTL", "WIPRO", "HCLTECH", "POWERGRID",
    "JSWSTEEL", "COALINDIA", "BAJAJ-AUTO"
}

def normalize_symbol(symbol):
    """
    Auto-add .NS only for known Indian stocks.
    Avoid modifying US/global symbols.
    """
    s = symbol.upper().strip()

    # If already has a suffix (AAPL, TSLA, RELIANCE.NS) → don't modify
    if "." in s:
        return s

    # If it matches our Indian stock list → add .NS
    if s in INDIAN_LIST:
        return s + ".NS"

    # Otherwise leave unchanged (AAPL, TSLA, AMZN, BTC-USD)
    return s


# ---------- Black-Scholes Formula ----------
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r*T) *
        (norm.cdf(d2) if option_type.lower() == 'call' else norm.cdf(-d2))
    )
    rho = K * T * np.exp(-r*T) * (
        norm.cdf(d2) if option_type.lower() == 'call' else -norm.cdf(-d2)
    )

    return price, delta, gamma, vega, theta, rho


# ---------- Main Option Model ----------
def run_option_model(stock_symbol, strike, expiry_date, option_type):

    IS_RENDER = os.environ.get("RENDER") == "true"

    # Smart normalization
    stock_symbol = normalize_symbol(stock_symbol)

    # Fetch 6 months of data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)

    # 1️⃣ Try yfinance first
    data = yf.download(
        stock_symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d")
    )

    # 2️⃣ Fallback to AlphaVantage if empty
    if not data.empty:
        close_prices = data["Close"]
    else:
        print("⚠️ yfinance failed → switching to AlphaVantage...")
        close_prices = fetch_alpha_vantage_history(stock_symbol)

    if close_prices is None or len(close_prices) < 30:
        raise ValueError(f"No reliable data found for {stock_symbol}.")

    # Spot price
    spot_price = float(close_prices.iloc[-1])

    # Volatility
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    vol_annual = float(log_returns.std() * np.sqrt(252))

    # Expiry
    expiry_datetime = datetime.strptime(expiry_date, "%Y-%m-%d")
    T = (expiry_datetime - datetime.today()).days / 365.0
    if T <= 0:
        raise ValueError("Expiry must be a future date.")

    r = 0.06  # Risk-free rate

    # Base Black-Scholes
    price, delta, gamma, vega, theta, rho = black_scholes_price(
        spot_price, strike, T, r, vol_annual, option_type
    )

    # ---------- ML Predictions ----------
    recent_prices = close_prices[-90:].values
    lr_pred, rf_pred = train_and_predict(recent_prices, n_lags=5)

    bs_price_lr, *_ = black_scholes_price(lr_pred, strike, T, r, vol_annual, option_type)
    bs_price_rf, *_ = black_scholes_price(rf_pred, strike, T, r, vol_annual, option_type)

    df_results = pd.DataFrame({
        "Model": ["Black-Scholes (today)", "Linear Regression (pred)", "Random Forest (pred)"],
        "Spot": [spot_price, lr_pred, rf_pred],
        "Option Price": [price, bs_price_lr, bs_price_rf]
    })

    # Disable backtest on Render
    if not IS_RENDER:
        try:
            backtest_stats = rolling_forecast_eval(close_prices.values, n_lags=5, test_days=60)
        except:
            backtest_stats = None
    else:
        backtest_stats = None

    # ---------- Sentiment Analysis ----------
    headlines = get_stock_news(stock_symbol)
    avg_sentiment = get_average_sentiment(headlines)

    sentiment_factor = 1 + avg_sentiment * 0.05
    lr_pred_adj = lr_pred * sentiment_factor
    rf_pred_adj = rf_pred * sentiment_factor

    bs_price_lr_adj, *_ = black_scholes_price(lr_pred_adj, strike, T, r, vol_annual, option_type)
    bs_price_rf_adj, *_ = black_scholes_price(rf_pred_adj, strike, T, r, vol_annual, option_type)

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
        "sentiment": avg_sentiment,
        "headlines": headlines
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


# ---------- Greeks Plot ----------
def plot_greeks(greeks, stock_symbol):
    K_range = greeks["K_range"]

    fig, axs = plt.subplots(5, 1, figsize=(8, 18), sharex=True)

    labels = ["Delta", "Gamma", "Vega", "Theta", "Rho"]
    colors = ["blue", "green", "red", "purple", "orange"]

    for i, key in enumerate(["delta", "gamma", "vega", "theta", "rho"]):
        axs[i].plot(K_range, greeks[key], color=colors[i])
        axs[i].set_ylabel(labels[i])
    axs[-1].set_xlabel("Strike Price")
    axs[0].set_title(f"Greeks vs Strike Price for {stock_symbol}")

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode("utf-8")
    plt.close(fig)
    return encoded


# ---------- Stock History Plot ----------
def plot_stock_history(stock_symbol, close_prices, lr_pred=None, rf_pred=None,
                       bs_price_lr=None, bs_price_rf=None):

    import matplotlib.dates as mdates

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    ax.plot(close_prices.index, close_prices.values, label="Close", color="blue")
    close_prices.rolling(20).mean().plot(ax=ax, label="20-day MA", color="orange")
    close_prices.rolling(50).mean().plot(ax=ax, label="50-day MA", color="green")

    last = close_prices.index[-1]
    next_day = last + pd.Timedelta(days=1)

    if lr_pred is not None:
        ax.scatter(next_day, lr_pred, color="purple", s=60)
    if rf_pred is not None:
        ax.scatter(next_day, rf_pred, color="red", s=60)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=35)
    plt.title(f"{stock_symbol} - Price History")
    plt.grid(True)
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return base64.b64encode(img.read()).decode("utf-8")
