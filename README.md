# Portfolio Forecasting and Optimization – AI Portfolio Optimizer (Prophet)

This repository contains a Streamlit-based web app that simulates a case study in financial engineering.  
The app combines time series forecasting and portfolio optimization to compare:

- **Historical-based optimal portfolio**
- **Forecast-based optimal portfolio**
- **Realized optimal portfolio**

---

## 🚀 Project Summary

In this project, we:

1. Download historical stock prices from Yahoo Finance
2. Forecast future prices using **Prophet**
3. Convert prices into returns and compute risk metrics
4. Optimize portfolio weights using **mean-variance optimization (Sharpe ratio maximization)**
5. Compare results between historical, forecast, and realized periods

This is a **learning project**, designed for educational purposes and not meant for real trading.

---

## 📌 Features

✔ Interactive UI (Streamlit)  
✔ Rolling forecast for 2024–2025 (monthly retraining)  
✔ One-time forecast for 2026  
✔ Portfolio optimization using Sharpe ratio  
✔ Efficient frontier visualization  
✔ Accuracy metrics (RMSE, MAPE)  
✔ Portfolio performance metrics (Return, Volatility, Sharpe, Sortino, Max Drawdown)

---

## 📈 Data

- **Source**: Yahoo Finance (`yfinance`)
- **Assets**: US equities (e.g., AAPL, MSFT, TSLA, AMZN, GOOG)
- **Price type**: Adjusted Close
- **Frequency**: Daily (default)
- **Date range**: Defined in the code (training and test periods)

---

## 🧠 How It Works


### 1. Forecasting Models for stock price prediction

Project Structure
portfolio-forecasting/
│
└── models/
    ├── arimastockprice.ipynb
    ├── prophetstockprice.ipynb
    └── xgbooststockprice.ipynb


The project uses **three forecasting models**:

1. **ARIMA**
2. **Prophet**
3. **XGBoost**

After testing all three, **XGBoost produced the best forecasting accuracy**.

### 1. Forecasting for portfolio optimization

After testing all two, **Prophet produced the best forecasting accuracy for portfolio optimization**.

Project Structure
portfolio-forecasting/
├── 
├── arimafinal.py
├── prophetfinal.py

- **Rolling Forecast (2024–2025)**  
  Prophet is retrained every month and forecasts the next month’s prices.

- **2026 Forecast**  
  A one-time forecast is made using the last known price from 2025.

### 2. Portfolio Optimization

We compute:

- Expected returns (`mu`)
- Covariance matrix (`cov`)

Then we maximize Sharpe ratio:

- Objective: maximize **(Expected Return / Volatility)**
- Constraints:
  - Each weight between 1% and 40%
  - Total weights sum to 100%

---

## 🔍 Forecast Accuracy

We evaluate forecast performance using:

- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

---

## 📊 Portfolio Metrics

For each portfolio, we compute:

- Annual Return
- Annual Volatility
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown

---

## 🧪 ARIMA Experiment

As part of the exploration, ARIMA was also tested as an alternative forecasting method.  
However, **ARIMA produced worse results than Prophet**, so Prophet was chosen for the final implementation.

---
## 📦 Installation
```bash
pip install streamlit yfinance prophet scikit-learn scipy pandas numpy matplotlib pypfopt xgboost statsmodels

🚀 Run the App
streamlit run prophetfinal.py

