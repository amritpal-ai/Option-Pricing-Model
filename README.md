# 📈 Option Pricing Model with ML Forecasting & Sentiment Analysis

This project is a complete **Option Pricing System** built using Python and Flask.  
It combines **financial modeling**, **machine learning**, and **sentiment analysis** to estimate the fair value of an option and analyze factors influencing its price.

---

## 🚀 Features

### 🧮 **1. Black–Scholes Option Pricing**
- Computes theoretical price for **Call** or **Put**
- Calculates **Greeks**: Delta, Gamma, Vega, Theta, Rho  
- Supports **NSE symbols** (e.g., RELIANCE.NS, TCS.NS)

---

### 🤖 **2. Machine Learning Price Prediction**
ML models predict next-day stock prices:
- **Linear Regression**
- **Random Forest**

These predicted prices are fed back into the Black–Scholes model to estimate the **ML-based option price**.

---

### 📊 **3. Rolling Forecast Evaluation**
(MAE & MAPE)
- Validates ML performance on last 60 days
- Helps measure prediction accuracy
- Ensures the model is not overfitting

---

### 📰 **4. News Sentiment Analysis**
- Fetches stock-related news headlines  
- Uses **VADER sentiment analyzer**
- Adjusts ML predictions with sentiment factor

---

### 💹 **5. Payoff Simulator**
Generates payoff diagrams for:
- **Call options**
- **Put options**

Shows profit/loss around the strike price.

---

### 📉 **6. Interactive Stock History Chart**
Includes:
- Last 6-month stock price  
- 20-day & 50-day moving averages  
- ML predicted next-day price  
- Adjusted Black–Scholes price labels  

---

## 🛠️ Tech Stack

**Backend**
- Python  
- Flask  
- yFinance  
- NumPy, Pandas  
- Scikit-Learn  
- VADER Sentiment  
- SciPy  
- Matplotlib  

**Frontend**
- HTML  
- CSS  
- Bootstrap  

---

## 📦 Project Structure

```
├── main.py               # Flask app
├── opm_core.py           # Option model + ML + sentiment
├── predictor.py          # ML model functions
├── sentiment_analyzer.py # News & sentiment logic
├── payoff_simulator.py   # Payoff diagram generator
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── styles.css
└── requirements.txt
```

---

## 📋 How It Works (Short Summary)

1. User enters:
   - Stock symbol  
   - Strike price  
   - Expiry date  
   - Call or Put  

2. System fetches past 6 months stock data.

3. Computes:
   - Black–Scholes option price  
   - All Greeks  

4. ML models predict the next day's stock price.

5. Sentiment score adjusts predictions.

6. Graphs are generated:
   - Greeks vs strike  
   - Stock history  
   - Payoff diagram  

7. Results are displayed in a clean UI.

---

## 📘 Ideal For

- Finance students  
- Quantitative modelling practice  
- ML + Finance mini projects  
- Resume or academic submission  
- Anyone wanting to understand option pricing with real-world data  

---

## ⭐ Future Improvements

- Add Monte Carlo Simulation  
- Add implied volatility calculation  
- Add deep learning model (LSTM)  
- Deploy full version online with stable market data API  

---

<img width="614" height="648" alt="image" src="https://github.com/user-attachments/assets/2c8244cb-0baf-48d7-aa14-21fc873c268c" />

<img width="431" height="911" alt="image" src="https://github.com/user-attachments/assets/fe3d0efa-78ba-4152-8191-1736dea76b4b" />

<img width="1229" height="805" alt="image" src="https://github.com/user-attachments/assets/e6f048f6-9792-451a-a8e0-a7d58ee72758" />

<img width="1639" height="672" alt="image" src="https://github.com/user-attachments/assets/30aded60-13dc-4056-bf8c-a995509ef4b2" />






