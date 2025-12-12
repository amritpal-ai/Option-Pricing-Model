# 📈 Option Pricing Model with ML Forecasting & Sentiment Analysis

A complete **Option Pricing & Analysis System** built using Python and Flask.  
This project combines **financial modeling**, **machine learning**, and **sentiment analysis** to estimate the fair value of stock options using real market data.

---

## 🚀 Features

### 🧮 **1. Black–Scholes Option Pricing**
- Computes theoretical price for **Call** or **Put**
- Full Greeks: **Delta, Gamma, Vega, Theta, Rho**
- Automatic **NSE symbol detection**  
  - Example: entering `RELIANCE` → becomes `RELIANCE.NS`

---

### 🤖 **2. Machine Learning Forecasting**
Predicts next-day stock price using:
- **Linear Regression**
- **Random Forest Regression**

These predicted prices are fed into the Black–Scholes model to generate:
- ML-based Option Price (LR)
- ML-based Option Price (RF)

---

### 📰 **3. News Sentiment Analysis**
- Fetches latest stock-related news headlines  
- Uses **VADER Sentiment Analyzer**
- Adjusts ML predictions based on sentiment score  
  - *(Positive sentiment → slight upward adjustment)*

---

### 💹 **4. Payoff Simulator**
Generates interactive payoff diagrams for:
- Call Options
- Put Options

Shows profit/loss movement around the strike.

---

### 📉 **5. Stock History Chart**
Includes:
- 6-month historical prices  
- 20-day & 50-day moving averages  
- ML predicted next-day prices  
- Adjusted Black–Scholes predictions  

---


## 🛠️ Tech Stack

### **Backend**
- Python  
- Flask  
- yFinance  
- NumPy, Pandas  
- Scikit-Learn  
- SciPy  
- Matplotlib  
- VADER Sentiment

### **Frontend**
- HTML, CSS  
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


---

## 📘 Workflow Summary

1. User enters:
   - Stock symbol  
   - Strike price  
   - Expiry date  
   - Call/Put option

2. System pulls last 6 months of stock data.

3. Computes:
   - Black–Scholes price  
   - Greeks  

4. ML models predict next-day stock price.

5. Sentiment score modifies predictions.

6. Generates:
   - Greeks chart  
   - Payoff chart  
   - Price history chart  

7. Displays results in an interactive UI.

---

## 🎯 Ideal For
- Finance & quant students  
- ML + Finance project portfolios  
- Resume / LinkedIn academic projects  
- Understanding option pricing practically  

---

## ⭐ Future Improvements
- Monte Carlo simulation  
- Implied volatility estimation  
- LSTM deep learning model  
- Full cloud deployment  
- Greeks heatmaps  

---

## 🖼️ Screenshots


<img width="614" height="648" alt="image" src="https://github.com/user-attachments/assets/2c8244cb-0baf-48d7-aa14-21fc873c268c" />

<img width="431" height="911" alt="image" src="https://github.com/user-attachments/assets/fe3d0efa-78ba-4152-8191-1736dea76b4b" />

<img width="1229" height="805" alt="image" src="https://github.com/user-attachments/assets/e6f048f6-9792-451a-a8e0-a7d58ee72758" />

<img width="1639" height="672" alt="image" src="https://github.com/user-attachments/assets/30aded60-13dc-4056-bf8c-a995509ef4b2" />



---

## ✔ No API Keys Needed
This project requires **zero setup** for API keys.  
Just download → install → run.

---







