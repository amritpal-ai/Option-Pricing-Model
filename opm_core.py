# ---------- Imports ----------
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from payoff_simulator import plot_payoff
import pandas as pd
from predictor import train_and_predict, plot_predictions


# ---------- Functions ----------
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

def binomial_price(S, K, T, r, sigma, option_type='call', steps=100):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r*dt) - d) / (u - d)
    
    ST = np.array([S * u**j * d**(steps-j) for j in range(steps+1)])
    if option_type.lower() == 'call':
        option_values = np.maximum(ST - K, 0)
    else:
        option_values = np.maximum(K - ST, 0)
    
    for i in range(steps-1, -1, -1):
        option_values = np.exp(-r*dt) * (p * option_values[1:i+2] + (1-p) * option_values[0:i+1])
    
    return option_values[0], None, None, None, None, None

def monte_carlo_price(S, K, T, r, sigma, option_type='call', simulations=10000):
    Z = np.random.standard_normal(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type.lower() == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    return price, None, None, None, None, None

# ---------- User Inputs ----------
stock_symbol = input("Enter Stock Symbol (e.g., RELIANCE): ").upper()
if not stock_symbol.endswith(".NS"):
    stock_symbol += ".NS"
K = float(input("Enter Strike Price: "))
expiry_date = input("Enter Expiry Date (YYYY-MM-DD): ")
option_type = input("Enter Option Type (call/put): ").lower()

# ---------- Fetch Historical Prices ----------
end_date = datetime.today()
start_date = end_date - timedelta(days=180)  # last 6 months
try:
    data = yf.download(stock_symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if data.empty:
        raise ValueError
except:
    raise ValueError(f"No data found for {stock_symbol}. Check symbol and retry.")
close_prices = data['Close']
spot_price = float(close_prices.iloc[-1])  # ensure float, not Series

# ---------- Compute Volatility ----------
log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
vol_daily = log_returns.std()
vol_annual = float(vol_daily * np.sqrt(252))  # convert to float

# ---------- Time to Expiry ----------
expiry_datetime = datetime.strptime(expiry_date, "%Y-%m-%d")
T = (expiry_datetime - datetime.today()).days / 365.0
if T <= 0:
    raise ValueError("Expiry must be a future date.")

# ---------- Risk-free rate ----------
r = 0.06

# ---------- Model Selection ----------
#if vol_annual > 0.5:
#   model_used = "Monte Carlo"
 #   price, delta, gamma, vega, theta, rho = monte_carlo_price(spot_price, K, T, r, vol_annual, option_type)
#elif T >= 1:
 #   model_used = "Binomial"
  #  price, delta, gamma, vega, theta, rho = binomial_price(spot_price, K, T, r, vol_annual, option_type)
#else:
 #   model_used = "Black-Scholes"
  #  price, delta, gamma, vega, theta, rho = black_scholes_price(spot_price, K, T, r, vol_annual, option_type)
  
# ---------- Model Selection (Forced Black-Scholes for demo) ----------
model_used = "Black-Scholes"
price, delta, gamma, vega, theta, rho = black_scholes_price(
    spot_price, K, T, r, vol_annual, option_type
)


# ---------- Output ----------
print("\n--- Option Pricing Result ---")
print(f"Stock: {stock_symbol}")
print(f"Spot Price: {spot_price:.2f}")
print(f"Annualized Volatility: {vol_annual:.4f}")
print(f"Time to Expiry (years): {T:.4f}")
print(f"Option Type: {option_type.capitalize()}")
print(f"Model Used: {model_used}")
print(f"Strike Price: {K}")
print(f"Option Price: {price:.4f}")
print(f"Delta: {delta if delta is not None else 'N/A'}")
print(f"Gamma: {gamma if gamma is not None else 'N/A'}")
print(f"Vega: {vega if vega is not None else 'N/A'}")
print(f"Theta: {theta if theta is not None else 'N/A'}")
print(f"Rho: {rho if rho is not None else 'N/A'}")

# ---------- Plot Greeks (Separate Subplots) ----------
if delta is not None and gamma is not None and vega is not None:
    K_range = np.linspace(K*0.8, K*1.2, 50)
    deltas, gammas, vegas, thetas, rhos = [], [], [], [], []
    for k in K_range:
        p, d, g, v, t, r_ = black_scholes_price(spot_price, k, T, r, vol_annual, option_type)
        deltas.append(d)
        gammas.append(g)
        vegas.append(v)
        thetas.append(t)
        rhos.append(r_)


    fig,axs = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
    axs[0].plot(K_range, deltas, color='blue')
    axs[0].set_ylabel('Delta')
    axs[0].set_title(f'Option Greeks vs Strike Price for {stock_symbol}')
    axs[0].grid(True)

    axs[1].plot(K_range, gammas, color='green')
    axs[1].set_ylabel('Gamma')
    axs[1].grid(True)

    axs[2].plot(K_range, vegas, color='red')
    axs[2].set_ylabel('Vega')
    axs[2].set_xlabel('Strike Price')
    axs[2].grid(True)
     
    axs[3].plot(K_range, thetas, color='purple')
    axs[3].set_ylabel('Theta')
    axs[3].grid(True)

    axs[4].plot(K_range, rhos, color='orange')
    axs[4].set_ylabel('Rho')
    axs[4].set_xlabel('Strike Price')
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("\nGreeks plotting not available for Binomial or Monte Carlo models.")


# ---------- Payoff Simulation ----------
see_payoff = input("\nDo you want to see the payoff diagram? (y/n): ").lower()
if see_payoff == "y":
    premium = price   # use the option price from the selected model
    print(f"\nUsing model price ({premium:.2f}) as premium for payoff diagram...")
    plot_payoff(K, premium, option_type=option_type)
    
# ---------- ML Prediction Extension ----------
from predictor import train_and_predict

# Use last 90 days of closing prices for ML
recent_prices = close_prices[-90:].values
lr_pred, rf_pred = train_and_predict(recent_prices, n_lags=5)

print("\n--- ML Predicted Next Day ---")
print(f"Linear Regression Predicted Spot: {lr_pred:.2f}")
print(f"Random Forest Predicted Spot:    {rf_pred:.2f}")

# Feed into Black–Scholes to get option price predictions
bs_price_lr, *_ = black_scholes_price(lr_pred, K, T, r, vol_annual, option_type)
bs_price_rf, *_ = black_scholes_price(rf_pred, K, T, r, vol_annual, option_type)
print("\n--- Predicted Option Prices ---")
print(f"Using Linear Regression Spot: {bs_price_lr:.2f}")
print(f"Using Random Forest Spot:    {bs_price_rf:.2f}")

# ---------- Comparison Table ----------
results = {
    "Model": ["Black-Scholes (today)", "Linear Regression (pred)", "Random Forest (pred)"],
    "Spot": [spot_price, lr_pred, rf_pred],
    "Option Price": [price, bs_price_lr, bs_price_rf]
}
df_results = pd.DataFrame(results)
print("\n--- Comparison Table ---")
print(df_results.to_string(index=False))


# ---------- Plot ML predictions ----------
plot_predictions(recent_prices, lr_pred, rf_pred)


# ---------- Backtest Evaluation ----------
#try:
    #eval_stats = rolling_forecast_eval(close_prices.values[-180:], n_lags=5)
   # print("\n--- ML Backtest (last 180 days) ---")
  #  print(f"Linear Regression -> MAE: {eval_stats['lr']['mae']:.2f}, MAPE: {eval_stats['lr']['mape']:.2%}")
 #   print(f"Random Forest     -> MAE: {eval_stats['rf']['mae']:.2f}, MAPE: {eval_stats['rf']['mape']:.2%}")
#except Exception as e:
#    print("Backtest skipped:", e)
