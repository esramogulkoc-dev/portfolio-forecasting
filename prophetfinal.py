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
import os
import warnings
warnings.filterwarnings("ignore")

# =============================
# CONFIG & DATES
# =============================
TRAIN_START = "2016-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31" 
CSV_PATH = "data/stockdate.csv"

# =============================
# DATA ENGINE (Hybrid)
# =============================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
        if not data.empty and len(data) > 5:
            return data.dropna()
        else: raise ValueError("Empty")
    except:
        if os.path.exists(CSV_PATH):
            df_all = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
            valid_tickers = [t for t in tickers if t in df_all.columns]
            return df_all[valid_tickers].loc[start:end].dropna()
        else:
            st.error("Data not found!"); st.stop()

# =============================
# MATH HELPER FUNCTIONS
# =============================
def sharpe_annualized(returns, freq=252):
    return (returns.mean() * freq) / (returns.std() * np.sqrt(freq)) if returns.std() != 0 else 0

def sortino_annualized(returns, freq=252, mar=0):
    mean = returns.mean() * freq
    downside = returns[returns < mar]
    downside_std = downside.std() * np.sqrt(freq)
    return mean / downside_std if downside_std != 0 else 0

def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    return ((cum_returns - peak) / peak).min()

def make_returns(price):
    return np.log(price / price.shift(1)).dropna()

# =============================
# PROPHET MODELS
# =============================
def train_prophet_return(ret_series):
    df = ret_series.reset_index(); df.columns = ["ds", "y"]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
    m.fit(df)
    return m

def rolling_prophet_price_forecast(price_series):
    returns = make_returns(price_series)
    preds = []
    retrain_dates = pd.date_range(TEST_START, TEST_END, freq="MS")
    last_price = price_series.loc[:TRAIN_END].iloc[-1]
    for d in retrain_dates:
        train_ret = returns.loc[:d - pd.Timedelta(days=1)]
        if len(train_ret) < 200: continue
        model = train_prophet_return(train_ret)
        future_dates = pd.bdate_range(d, min(d + pd.offsets.MonthEnd(1), pd.to_datetime(TEST_END)))
        future = pd.DataFrame({"ds": future_dates})
        ret_fc = model.predict(future)["yhat"].values
        prices = [last_price]
        for r in ret_fc: prices.append(prices[-1] * np.exp(r))
        price_fc = pd.Series(prices[1:], index=future_dates)
        last_price = price_fc.iloc[-1]
        preds.append(price_fc)
    return pd.concat(preds).groupby(level=0).mean()

def prophet_price_forecast_2026(price_series, start_date, horizon_months, last_known_price):
    returns = make_returns(price_series)
    model = train_prophet_return(returns)
    future_end = pd.to_datetime(start_date) + pd.DateOffset(months=horizon_months-1)
    future_dates = pd.bdate_range(start_date, future_end)
    future = pd.DataFrame({"ds": future_dates})
    ret_fc = model.predict(future)["yhat"].values
    prices = [last_known_price]
    for r in ret_fc: prices.append(prices[-1] * np.exp(r))
    return pd.Series(prices[1:], index=future_dates)

def optimize_portfolio(mu, cov, min_w=0.01, max_w=0.4):
    n = len(mu); bounds = [(min_w, max_w)] * n
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    def neg_sharpe(w): return -(w @ mu) / np.sqrt(w.T @ cov @ w)
    res = minimize(neg_sharpe, np.ones(n)/n, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="Prophet Optimizer", layout="wide")
st.title("📈 AI Portfolio Optimizer – Prophet")

with st.sidebar:
    tickers = st.multiselect("Assets", ["AAPL","MSFT","GOOG","AMZN","NVDA","META","TSLA","JPM","V","WMT"], default=["AAPL","MSFT","GOOG"])
    horizon_months = st.slider("2026 Forecast Horizon", 3, 12, 3)
    if len(tickers) < 2: st.stop()

# 1. FETCH DATA
df_close = get_data(tickers, TRAIN_START, TRAIN_END)
realized_close = get_data(tickers, TEST_START, TEST_END)

# 2. HISTORICAL CHART
st.subheader("1. Historical Performance (Base 100)")
fig1, ax1 = plt.subplots(figsize=(12,4))
for t in tickers: ax1.plot(df_close[t] / df_close[t].iloc[0] * 100, label=t)
ax1.legend(); ax1.grid(alpha=0.3); st.pyplot(fig1)

# 3. RUN MODELS
with st.spinner("AI is calculating forecasts..."):
    rolling_forecast_df = pd.DataFrame()
    for t in tickers: rolling_forecast_df[t] = rolling_prophet_price_forecast(df_close[t])

    forecast_2026_df = pd.DataFrame()
    last_prices_2025 = realized_close.iloc[-1] if not realized_close.empty else df_close.iloc[-1]
    for t in tickers:
        forecast_2026_df[t] = prophet_price_forecast_2026(df_close[t], "2026-01-01", horizon_months, last_prices_2025[t])

# 4. FORECAST VS REALIZED
st.header("2. 🔮 Forecast Analysis")
for t in tickers:
    fig2, ax2 = plt.subplots(figsize=(12,4))
    if not realized_close.empty: ax2.plot(realized_close[t], label="Realized", color="black", linewidth=2)
    ax2.plot(rolling_forecast_df[t], label="24-25 Rolling FC", linestyle="--")
    ax2.plot(forecast_2026_df[t], label="2026 Future FC", linestyle=":")
    ax2.set_title(f"Prediction: {t}"); ax2.legend(); ax2.grid(alpha=0.2); st.pyplot(fig2)

# 5. ACCURACY TABLE
st.subheader("🎯 Accuracy (2024-2025 Rolling Forecast)")
acc_data = []
for t in tickers:
    idx = realized_close.index.intersection(rolling_forecast_df.index)
    y, yhat = realized_close[t].loc[idx], rolling_forecast_df[t].loc[idx]
    acc_data.append({"Ticker": t, "RMSE": np.sqrt(mean_squared_error(y, yhat)), "MAPE (%)": np.mean(np.abs((y - yhat) / y)) * 100})
st.dataframe(pd.DataFrame(acc_data).set_index("Ticker"))

# 6. PORTFOLIO CALCULATIONS
returns_hist = make_returns(df_close)
returns_roll_fc = make_returns(rolling_forecast_df)
returns_real = make_returns(realized_close)
fc_2026_with_prev = pd.concat([realized_close.iloc[[-1]], forecast_2026_df])
returns_fc_2026 = np.log(fc_2026_with_prev / fc_2026_with_prev.shift(1)).dropna()

# Weights
w_hist = optimize_portfolio(returns_hist.mean().values, returns_hist.cov().values)
w_fc_roll = optimize_portfolio(returns_roll_fc.mean().values, returns_roll_fc.cov().values)
w_real = optimize_portfolio(returns_real.mean().values, returns_real.cov().values)
w_fc_2026 = optimize_portfolio(returns_fc_2026.mean().values, returns_fc_2026.cov().values)

# 7. WEIGHTS TABLE (Yüzde Formatı)
st.subheader("⚖️ Portfolio Weights")
weights_df = pd.DataFrame({
    "Ticker": tickers,
    "Historical (%)": w_hist * 100,
    "Forecast 24-25 (%)": w_fc_roll * 100,
    "Realized 24-25 (%)": w_real * 100,
    "Forecast 2026 (%)": w_fc_2026 * 100
}).set_index("Ticker")
st.dataframe(weights_df.style.format("{:.1f}%").background_gradient(cmap="RdYlGn", axis=1))

# 8. CUMULATIVE RETURNS CHART (Orijinal Grafik Eklendi)
st.header("📈 Portfolio Cumulative Returns")

cum_roll_fc = (1 + returns_roll_fc.dot(w_fc_roll)).cumprod()
cum_real = (1 + returns_real.dot(w_real)).cumprod()
cum_real_end_2025 = cum_real.iloc[-1] if not cum_real.empty else 1.0
cum_fc_2026 = (1 + returns_fc_2026.dot(w_fc_2026)).cumprod() * cum_real_end_2025

fig3, ax3 = plt.subplots(figsize=(12,6))
ax3.plot(cum_roll_fc, label="Forecast Portfolio (2024-2025 Rolling)", linewidth=2)
ax3.plot(cum_real, label="Historical Portfolio (2024-2025 Actual)", linewidth=2, linestyle="--")
ax3.plot(cum_fc_2026, label="Forecast Portfolio (2026 Proj.)", linewidth=2, linestyle=":")
ax3.set_title("Portfolio Cumulative Performance")
ax3.legend(); ax3.grid(alpha=0.3)
st.pyplot(fig3)

# 9. METRICS TABLE
st.subheader("📊 Portfolio Metrics")
ann = 252
m_list = []
for name, ret, w in [("Historical", returns_hist, w_hist), ("FC 24-25", returns_roll_fc, w_fc_roll), 
                     ("Realized 24-25", returns_real, w_real), ("FC 2026", returns_fc_2026, w_fc_2026)]:
    port_ret = ret.dot(w)
    m_list.append({
        "Portfolio": name,
        "Ann. Return (%)": port_ret.mean() * ann * 100,
        "Ann. Volatility (%)": port_ret.std() * np.sqrt(ann) * 100,
        "Sharpe": sharpe_annualized(port_ret),
        "Sortino": sortino_annualized(port_ret),
        "Max DD (%)": max_drawdown((1+port_ret).cumprod()) * 100
    })
st.dataframe(pd.DataFrame(m_list).set_index("Portfolio").style.background_gradient(cmap="RdYlGn"))

st.success("Full Analysis Complete!")