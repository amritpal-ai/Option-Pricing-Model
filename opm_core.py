# ---------- Imports ----------
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from datetime import datetime

from payoff_simulator import plot_payoff_base64
from predictor import train_and_predict
from sentiment_analyzer import get_average_sentiment, get_stock_news



# ---------- Detect NSE Symbols ----------
INDIAN_LIST = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "HDFC", "SBIN", "KOTAKBANK",
    "ICICIBANK", "ITC", "AXISBANK", "LT", "MARUTI", "SUNPHARMA",
    "BAJAJFINSERV", "BHARTIARTL", "TITAN", "ULTRACEMCO", "ASIANPAINT"
]

def normalize_symbol(stock):
    """
    Add .NS ONLY for Indian stocks.
    """
    s = stock.upper().strip()
    if s in INDIAN_LIST:
        return s + ".NS"
    return s



# ---------- Yahoo Finance Fetcher ----------
def fetch_yf_history(symbol):
    """
    Fetch 6 months of data from yfinance.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo")

        if hist.empty:
            print("yfinance returned empty data for:", symbol)
            return None

        return hist["Close"]

    except Exception as e:
        print("yfinance fetch failed:", e)
        return None



# ---------- Black-Scholes ----------
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) *
        (norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2))
    )
    rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2))

    return price, delta, gamma, vega, theta, rho



# ---------- Main Option Model ----------
def run_option_model(stock_symbol, strike, expiry_date, option_type):

    symbol = normalize_symbol(stock_symbol)

    # Fetch data using yfinance only
    close_prices = fetch_yf_history(symbol)

    if close_prices is None or len(close_prices) < 50:
        raise ValueError(f"No reliable price data found for {symbol}.")

    spot_price = float(close_prices.iloc[-1])

    # Volatility
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    vol_annual = float(log_returns.std() * np.sqrt(252))

    # Expiry
    expiry = datetime.strptime(expiry_date, "%Y-%m-%d")
    T = (expiry - datetime.today()).days / 365.0

    if T <= 0:
        raise ValueError("Expiry must be in the future.")

    r = 0.06

    # BS price
    price, delta, gamma, vega, theta, rho = black_scholes_price(
        spot_price, strike, T, r, vol_annual, option_type
    )

    # ML predictions
    recent = close_prices[-90:].values
    lr_pred, rf_pred = train_and_predict(recent, n_lags=5)

    # Predicted option prices
    bs_lr, *_ = black_scholes_price(lr_pred, strike, T, r, vol_annual, option_type)
    bs_rf, *_ = black_scholes_price(rf_pred, strike, T, r, vol_annual, option_type)

    df_results = pd.DataFrame({
        "Model": ["Black-Scholes", "Linear Regression", "Random Forest"],
        "Spot": [spot_price, lr_pred, rf_pred],
        "Option Price": [price, bs_lr, bs_rf]
    })

    # Disable backtest to avoid slowdowns on deployment
    backtest_stats = None

    # Sentiment
    headlines = get_stock_news(symbol)
    avg_sentiment = get_average_sentiment(headlines)

    sentiment_factor = 1 + avg_sentiment * 0.05
    lr_pred_adj = lr_pred * sentiment_factor
    rf_pred_adj = rf_pred * sentiment_factor

    bs_lr_adj, *_ = black_scholes_price(lr_pred_adj, strike, T, r, vol_annual, option_type)
    bs_rf_adj, *_ = black_scholes_price(rf_pred_adj, strike, T, r, vol_annual, option_type)

    return {
        "stock": symbol,
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
        "comparison_table": df_results.to_html(classes="table table-striped", index=False),
        "ml_preds": {
            "lr_pred": lr_pred_adj,
            "rf_pred": rf_pred_adj,
            "bs_price_lr": bs_lr_adj,
            "bs_price_rf": bs_rf_adj,
        },
        "backtest_stats": backtest_stats,
        "close_prices": close_prices,
        "headlines": headlines,
        "sentiment": avg_sentiment,
    }



# ---------- Greeks ----------
def get_greeks(result):
    K = result["strike"]
    spot = result["spot"]
    T = result["T"]
    r = result["r"]
    sigma = result["vol"]
    opt_type = result["option_type"]

    K_range = np.linspace(K * 0.8, K * 1.2, 50)
    delta, gamma, vega, theta, rho = [], [], [], [], []

    for k in K_range:
        _, d, g, v, t, r_ = black_scholes_price(spot, k, T, r, sigma, opt_type)
        delta.append(d)
        gamma.append(g)
        vega.append(v)
        theta.append(t)
        rho.append(r_)

    return {
        "K_range": K_range.tolist(),
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho
    }



# ---------- Plot Greeks ----------
def plot_greeks(greeks, symbol):
    K = greeks["K_range"]

    fig, axs = plt.subplots(5, 1, figsize=(8, 18), sharex=True)
    keys = ["delta", "gamma", "vega", "theta", "rho"]
    colors = ["blue", "green", "red", "purple", "orange"]

    for i, key in enumerate(keys):
        axs[i].plot(K, greeks[key], color=colors[i])
        axs[i].set_ylabel(key.capitalize())

    axs[-1].set_xlabel("Strike Price")
    axs[0].set_title(f"Greeks vs Strike Price — {symbol}")

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format="png")
    plt.close()

    return base64.b64encode(img.getvalue()).decode("utf-8")



# ---------- Price History Plot ----------
def plot_stock_history(symbol, close_prices, lr_pred=None, rf_pred=None,
                       bs_price_lr=None, bs_price_rf=None):

    import matplotlib.dates as mdates

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    ax.plot(close_prices.index, close_prices.values, label="Close", color="blue")
    close_prices.rolling(20).mean().plot(ax=ax, label="20-day MA", color="orange")
    close_prices.rolling(50).mean().plot(ax=ax, label="50-day MA", color="green")

    last = close_prices.index[-1]
    next_day = last + pd.Timedelta(days=1)

    if lr_pred:
        ax.scatter(next_day, lr_pred, s=60, color="purple")

    if rf_pred:
        ax.scatter(next_day, rf_pred, s=60, color="red")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    plt.legend()
    plt.title(f"{symbol} — Price History")
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close()

    return base64.b64encode(img.read()).decode("utf-8")
