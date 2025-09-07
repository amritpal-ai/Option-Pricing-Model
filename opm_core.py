# ---------- Imports ----------
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import pandas as pd
from payoff_simulator import plot_payoff
from predictor import train_and_predict, plot_predictions


# ---------- Pricing Functions ----------
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


# ---------- Core Runner ----------
def run_option_model(stock_symbol, strike, expiry_date, option_type):
    # Ensure .NS for NSE
    if not stock_symbol.endswith(".NS"):
        stock_symbol += ".NS"

    # Fetch historical prices
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)
    data = yf.download(stock_symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

    if data.empty:
        raise ValueError(f"No data found for {stock_symbol}. Check symbol and retry.")

    close_prices = data['Close']
    spot_price = float(close_prices.iloc[-1])

    # Volatility
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

    # Always Black-Scholes (for now)
    price, delta, gamma, vega, theta, rho = black_scholes_price(
        spot_price, strike, T, r, vol_annual, option_type
    )

    # Machine Learning Predictions
    recent_prices = close_prices[-90:].values
    lr_pred, rf_pred = train_and_predict(recent_prices, n_lags=5)

    bs_price_lr, *_ = black_scholes_price(lr_pred, strike, T, r, vol_annual, option_type)
    bs_price_rf, *_ = black_scholes_price(rf_pred, strike, T, r, vol_annual, option_type)

    df_results = pd.DataFrame({
        "Model": ["Black-Scholes (today)", "Linear Regression (pred)", "Random Forest (pred)"],
        "Spot": [spot_price, lr_pred, rf_pred],
        "Option Price": [price, bs_price_lr, bs_price_rf]
    })

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
            "lr_pred": lr_pred,
            "rf_pred": rf_pred,
            "bs_price_lr": bs_price_lr,
            "bs_price_rf": bs_price_rf,
        },
        "comparison_table": df_results.to_html(classes="table table-striped", index=False)
    }


# ---------- Greeks (for plotting) ----------
def get_greeks(result_dict):
    K = result_dict["strike"]
    spot = result_dict["spot"]
    T = result_dict["T"]
    r = result_dict["r"]
    sigma = result_dict["vol"]
    option_type = result_dict["option_type"]

    K_range = np.linspace(K*0.8, K*1.2, 50)
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
