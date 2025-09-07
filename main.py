from flask import Flask, render_template, request
from opm_core import run_option_model, get_greeks

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

    # Run core model (pricing + ML)
    results = run_option_model(stock, strike, expiry, option_type)

    # Get Greeks for plotting
    greeks = get_greeks(results)

    # ML predictions are already inside results["ml_preds"]
    ml_preds = results["ml_preds"]

    return render_template(
        "result.html",
        results=results,
        greeks=greeks,
        ml_preds=ml_preds
    )

if __name__ == "__main__":
    app.run(debug=True)
