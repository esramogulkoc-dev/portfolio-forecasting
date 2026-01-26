# AI Portfolio Optimizer – Prophet (5-Minute Presentation Notes)

## 1) Libraries and Setup

```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


“In this project, we build a web application using Streamlit. We download stock price data using yfinance, forecast future prices using Prophet, and perform portfolio optimization using PyPortfolioOpt.”

2) Date Ranges
TRAIN_START = "2016-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"


“We split the data into two parts:
training data from 2016–2023,
and testing data from 2024–2025 for model evaluation.”

3) Data Download Function
@st.cache_data
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    return data.dropna()


“This function downloads the closing prices of the selected stocks.
We use Streamlit caching to avoid downloading the same data repeatedly.”

4) Returns and Performance Functions
def make_returns(price):
    return np.log(price / price.shift(1)).dropna()


“We convert prices into returns.
Returns are calculated using the logarithmic change of prices.”

def sharpe_annualized(returns, freq=252):
    mean = returns.mean() * freq
    vol = returns.std() * np.sqrt(freq)
    return mean / vol if vol != 0 else 0


“The Sharpe ratio is a risk-adjusted return metric.
We calculate it by dividing the average return by volatility.”

5) Prophet Model (Forecasting)
def train_prophet_return(ret_series):
    df = ret_series.reset_index()
    df.columns = ["ds", "y"]
    m = Prophet(...)
    m.fit(df)
    return m


“We train the Prophet model on return series.
Prophet captures trends and seasonality in time series data.”

6) Rolling Forecast (2024-2025)
def rolling_prophet_price_forecast(price_series):
    ...


“For 2024–2025, we retrain the model each month and forecast the next month’s prices.
This simulates how models are updated in real life.”

7) 2026 Forecast
def prophet_price_forecast_2026(price_series, start_date, horizon_months, last_known_price):
    ...


“We also make a one-time forecast for 2026.
The forecast starts from the last known price at the end of 2025 and projects forward for the selected number of months.”

8) Portfolio Optimization
def optimize_portfolio(mu, cov, min_w=0.01, max_w=0.4):
    ...


“This function finds the best weights for each stock in the portfolio.
The goal is to maximize the Sharpe ratio.
We also enforce constraints: each stock weight must be between 1% and 40%, and total weights must sum to 100%.”

9) Streamlit UI (User Interface)
st.title("📈 AI Portfolio Optimizer – Prophet")


“The app allows users to select tickers and choose a forecast horizon.
Results update dynamically based on the user’s input.”

10) Download and Display Data
df_close = download_data(tickers, TRAIN_START, TRAIN_END)
realized_close = download_data(tickers, TEST_START, TEST_END)


“We download both historical data and realized test data.
This allows us to compare forecast results with real market performance.”

11) Forecasts (Rolling and 2026)
rolling_forecast_df = ...
forecast_df = ...


“We compute rolling forecasts for 2024–2025 and a one-time forecast for 2026.”

12) Forecast vs Realized Graphs
for t in tickers:
    ...
    ax.plot(realized_close[t], label="Realized")
    ax.plot(rolling_forecast_df[t], label="Prophet Rolling")
    ax.plot(forecast_df[t], label="Prophet 2026")


“We plot realized prices and forecasts side by side.
This helps us visually evaluate forecast accuracy.”

13) Forecast Accuracy (RMSE, MAPE)
acc.append({
    "Ticker": t,
    "RMSE": ...,
    "MAPE (%)": ...
})


“We measure forecast accuracy using RMSE and MAPE.
These metrics quantify how far predictions are from actual values.”

14) Portfolio Weights Calculation
mu_hist = returns_hist.mean().values
cov_hist = returns_hist.cov().values

w_hist = optimize_portfolio(mu_hist, cov_hist)
...


“For each period, we calculate average returns and the risk matrix.
Then we find the portfolio weights that maximize the Sharpe ratio.”

15) Portfolio Cumulative Returns
cum_roll_fc = (1 + returns_roll_fc.dot(w_roll_fc)).cumprod()
cum_real = (1 + returns_real.dot(w_real)).cumprod()
What to say in the presentation:

“We compute cumulative portfolio returns over time.
This shows how the portfolio grows using different weight strategies.”

16) Portfolio Metrics Table
metrics_df = pd.DataFrame({
    "Portfolio": [...],
    "Annual Return (%)": [...],
    "Annual Volatility (%)": [...],
    "Sharpe Ratio": [...],
    ...
})


“Finally, we display key portfolio metrics such as annual return, volatility, Sharpe ratio, Sortino ratio, and max drawdown in a table.”

Closing Summary (5-Minute Wrap-Up)
“This project forecasts future prices using Prophet and then optimizes portfolio weights to maximize the Sharpe ratio.
Results are visualized using graphs and tables, and the model’s accuracy is evaluated using RMSE and MAPE.”

Final Note (Real-World Disclaimer)

“This project is a learning exercise and demonstrates how forecasting and optimization can be combined.
However, in real-world finance, using Prophet forecasts directly to allocate portfolio weights is not practical.
Financial markets are highly noisy and influenced by many unpredictable factors, and model forecasts often underestimate volatility.
Therefore, this should not be used as a real investment strategy—it's mainly for educational purposes.”