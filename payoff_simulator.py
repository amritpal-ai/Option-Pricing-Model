import numpy as np
import matplotlib.pyplot as plt

def option_payoff(spot_prices, K, premium, option_type="call", position="long"):
    """
    Calculate option payoff at expiry.
    
    spot_prices : list or np.array of possible spot prices at expiry
    K           : strike price
    premium     : option premium (cost paid/received)
    option_type : 'call' or 'put'
    position    : 'long' (buyer) or 'short' (seller)
    """
    if option_type.lower() == "call":
        payoff = np.maximum(spot_prices - K, 0)
    else:  # put option
        payoff = np.maximum(K - spot_prices, 0)
    
    # Profit/Loss = payoff - premium for buyer
    profit = payoff - premium
    
    if position == "short":  # seller’s P/L is opposite
        profit = -profit
    
    return profit

import io
import base64

def plot_payoff_base64(K, premium, option_type="call"):
    # Generate a range of possible spot prices
    spot_prices = np.linspace(K*0.7, K*1.3, 100)
    
    long_profit = option_payoff(spot_prices, K, premium, option_type, position="long")
    short_profit = option_payoff(spot_prices, K, premium, option_type, position="short")
    
    plt.figure(figsize=(10,6))
    plt.axhline(0, color='black', linewidth=1)  # P/L = 0 line
    
    # Long option payoff
    plt.plot(spot_prices, long_profit, label=f"Long {option_type.capitalize()}", color="green")
    
    # Short option payoff
    plt.plot(spot_prices, short_profit, label=f"Short {option_type.capitalize()}", color="red")
    
    # Breakeven
    if option_type.lower() == "call":
        breakeven = K + premium
    else:
        breakeven = K - premium
    plt.axvline(breakeven, color="blue", linestyle="--", label=f"Breakeven: {breakeven:.2f}")
    
    plt.title(f"{option_type.capitalize()} Option Payoff (K={K}, Premium={premium})")
    plt.xlabel("Spot Price at Expiry")
    plt.ylabel("Profit / Loss")
    plt.legend()
    plt.grid(True)

    # 🔑 Save to Base64
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf-8")
    plt.close()

    return plot_url

    



# Example demo
# if __name__ == "__main__":
 #   K = 20200
  #  premium = 100
   # plot_payoff(K, premium, option_type="call") 
