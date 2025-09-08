from flask import Flask, render_template, request
from opm_core import run_option_model, get_greeks, plot_greeks

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

from payoff_simulator import plot_payoff_base64

@app.route("/calculate", methods=["POST"])
def calculate():
    stock = request.form["stock"]
    strike = float(request.form["strike"])
    expiry = request.form["expiry"]
    option_type = request.form["option_type"]

    results = run_option_model(stock, strike, expiry, option_type)
    greeks = get_greeks(results)

# 📈 Add stock history chart here
    history_plot = plot_stock_history (
     stock,
     results["close_prices"],   # <-- include close_prices in results dict earlier
     lr_pred=results["ml_preds"]["lr_pred"],
     rf_pred=results["ml_preds"]["rf_pred"]
     )



    greek_plot = plot_greeks(greeks, stock)

    # ✅ Add payoff diagram
    payoff_plot = plot_payoff_base64(results["strike"], results["price"], option_type)

    ml_preds = results["ml_preds"]

    return render_template(
    "result.html",
    results=results,
    greeks=greeks,
    ml_preds=ml_preds,
    greek_plot=greek_plot,
    payoff_plot=payoff_plot,
    history_plot=history_plot
   )

import matplotlib.pyplot as plt
import io, base64
def plot_stock_history(stock_symbol, close_prices, lr_pred=None, rf_pred=None):
    """
    Plot stock price history with moving averages and optional ML predictions.
    """
    plt.figure(figsize=(10, 5))
    
    # Plot actual prices
    plt.plot(close_prices.index, close_prices.values, label="Closing Price", color="blue")

    # Moving averages
    close_prices.rolling(20).mean().plot(label="20-day MA", color="orange")
    close_prices.rolling(50).mean().plot(label="50-day MA", color="green")

    # Add ML predictions (optional)
    if lr_pred is not None:
        plt.scatter(close_prices.index[-1] + pd.Timedelta(days=1), lr_pred,
                    color="purple", marker="o", label="LR Predicted Next Day")
    if rf_pred is not None:
        plt.scatter(close_prices.index[-1] + pd.Timedelta(days=1), rf_pred,
                    color="red", marker="x", label="RF Predicted Next Day")

    plt.title(f"{stock_symbol} - Last 6 Months Price History + Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure for Flask
    filename = f"static/{stock_symbol}_history.png"
    plt.savefig(filename)
    plt.close()

    return filename



if __name__ == "__main__":
    app.run(debug=True)
