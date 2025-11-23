from flask import Flask, render_template, request
from opm_core import run_option_model, get_greeks, plot_greeks
from payoff_simulator import plot_payoff_base64
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    stock = request.form["stock"]
    strike = float(request.form["strike"])
    expiry = request.form["expiry"]
    option_type = request.form["option_type"]

    # Run main model
    results = run_option_model(stock, strike, expiry, option_type)
    greeks = get_greeks(results)

    # 📈 Stock history chart with ML predictions and annotations
    history_plot = plot_stock_history(
        stock,
        results["close_prices"],
        lr_pred=results["ml_preds"]["lr_pred"],
        rf_pred=results["ml_preds"]["rf_pred"],
        bs_price_lr=results["ml_preds"]["bs_price_lr"],
        bs_price_rf=results["ml_preds"]["bs_price_rf"]
    )

    # Greeks & payoff diagrams
    greek_plot = plot_greeks(greeks, stock)
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


# ---------------------------------------
# 📊 Plot Stock History with annotations
# ---------------------------------------
def plot_stock_history(stock_symbol, close_prices, lr_pred=None, rf_pred=None,
                       bs_price_lr=None, bs_price_rf=None):
    import matplotlib.dates as mdates

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # Plot stock price and moving averages
    ax.plot(close_prices.index, close_prices.values, label=stock_symbol, color="green")
    close_prices.rolling(20).mean().plot(ax=ax, label="20-day MA", color="orange")
    close_prices.rolling(50).mean().plot(ax=ax, label="50-day MA", color="blue")

    last_date = close_prices.index[-1]
    next_date = last_date + pd.Timedelta(days=1)

    # 🔮 ML Predictions + Annotations
    if lr_pred is not None:
        ax.scatter(next_date, lr_pred, color="purple", marker="o", s=60, label="LR Predicted Next Day")
        if bs_price_lr is not None:
            ax.annotate(f"Opt: {bs_price_lr:.2f}", xy=(next_date, lr_pred),
                        xytext=(8, 8), textcoords="offset points",
                        color="purple", fontsize=9, weight='bold')

    if rf_pred is not None:
        ax.scatter(next_date, rf_pred, color="red", marker="x", s=70, label="RF Predicted Next Day")
        if bs_price_rf is not None:
            ax.annotate(f"Opt: {bs_price_rf:.2f}", xy=(next_date, rf_pred),
                        xytext=(8, -14), textcoords="offset points",
                        color="red", fontsize=9, weight='bold')

    # Formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=35)
    plt.title(f"{stock_symbol} - Last 6 Months Price History + Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Convert to Base64 for embedding in Flask
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format="png")
    img_bytes.seek(0)
    encoded = base64.b64encode(img_bytes.read()).decode("utf-8")
    plt.close()

    return encoded


if __name__ == "__main__":
    app.run(debug=True)
