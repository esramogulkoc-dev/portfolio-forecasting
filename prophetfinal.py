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
# SMART DATA DOWNLOAD (Hybrid Logic)
# =============================
@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    """
    Önce canlı çekmeyi dener, Yahoo engel koyarsa (Rate Limit) CSV'den okur.
    """
    try:
        # 1. Canlı Veri Denemesi
        data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
        if not data.empty and len(data) > 5:
            return data.dropna()
        else:
            raise ValueError("Boş Veri")
    except Exception:
        # 2. Yedek Plan: CSV'den Oku
        if os.path.exists(CSV_PATH):
            st.warning("⚠️ Yahoo Finance erişimi kısıtlı. Kayıtlı veriler (CSV) kullanılıyor...")
            df_all = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
            # Sadece mevcut ticker'ları ve istenen tarih aralığını filtrele
            valid_tickers = [t for t in tickers if t in df_all.columns]
            return df_all[valid_tickers].loc[start:end].dropna()
        else:
            st.error(f"Kritik Hata: Ne canlı veri çekilebildi ne de {CSV_PATH} bulundu!")
            st.stop()

# =============================
# MATH FUNCTIONS
# =============================
def sharpe_annualized(returns, freq=252):
    mean = returns.mean() * freq
    vol = returns.std() * np.sqrt(freq)
    return mean / vol if vol != 0 else 0

def sortino_annualized(returns, freq=252, mar=0):
    mean = returns.mean() * freq
    downside = returns[returns < mar]
    downside_std = downside.std() * np.sqrt(freq)
    return mean / downside_std if downside_std != 0 else 0

def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

def make_returns(price):
    return np.log(price / price.shift(1)).dropna()

# =============================
# FORECASTING ENGINES
# =============================
def train_prophet_return(ret_series):
    df = ret_series.reset_index()
    df.columns = ["ds", "y"]
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    m.fit(df)
    return m

def rolling_prophet_price_forecast(price_series):
    returns = make_returns(price_series)
    preds = []
    retrain_dates = pd.date_range(TEST_START, TEST_END, freq="MS")
    
    # Check if we have data for the start point
    if price_series.loc[:TRAIN_END].empty:
        return pd.Series()
        
    last_price = price_series.loc[:TRAIN_END].iloc[-1]

    for d in retrain_dates:
        train_ret = returns.loc[:d - pd.Timedelta(days=1)]
        if len(train_ret) < 200: continue

        model = train_prophet_return(train_ret)
        future_dates = pd.bdate_range(d, min(d + pd.offsets.MonthEnd(1), pd.to_datetime(TEST_END)))

        future = pd.DataFrame({"ds": future_dates})
        ret_fc = model.predict(future)["yhat"].values

        prices = [last_price]
        for r in ret_fc:
            prices.append(prices[-1] * np.exp(r))

        price_fc = pd.Series(prices[1:], index=future_dates)
        last_price = price_fc.iloc[-1]
        preds.append(price_fc)

    return pd.concat(preds).groupby(level=0).mean() if preds else pd.Series()

def prophet_price_forecast_2026(price_series, start_date, horizon_months, last_known_price):
    returns = make_returns(price_series)
    model = train_prophet_return(returns)
    future_end = pd.to_datetime(start_date) + pd.DateOffset(months=horizon_months-1)
    future_dates = pd.bdate_range(start_date, future_end)
    future = pd.DataFrame({"ds": future_dates})
    ret_fc = model.predict(future)["yhat"].values
    last_price = last_known_price
    prices = [last_price]
    for r in ret_fc:
        prices.append(prices[-1] * np.exp(r))
    return pd.Series(prices[1:], index=future_dates)

# =============================
# OPTIMIZATION
# =============================
def optimize_portfolio(mu, cov, min_w=0.01, max_w=0.4):
    n = len(mu)
    bounds = [(min_w, max_w)] * n
    cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    def neg_sharpe(w):
        return -(w @ mu) / np.sqrt(w.T @ cov @ w)
    res = minimize(neg_sharpe, np.ones(n)/n, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x

# =============================
# UI SETUP
# =============================
st.set_page_config(page_title="AI Portfolio Optimizer", layout="wide")
st.title("📈 AI Portfolio Optimizer – Prophet")

with st.sidebar:
    st.header("Settings")
    tickers = st.multiselect(
        "Select Assets",
        ["AAPL","MSFT","GOOG","AMZN","NVDA","META","TSLA","JPM","V","WMT"],
        default=["AAPL","MSFT","GOOG"]
    )
    horizon_months = st.slider("2026 Forecast Horizon", 3, 12, 3)
    
    if len(tickers) < 2:
        st.warning("Please select at least 2 tickers.")
        st.stop()

# =============================
# MAIN PIPELINE
# =============================

# 1. Fetch Data
df_close = get_data(tickers, TRAIN_START, TRAIN_END)
realized_close = get_data(tickers, TEST_START, TEST_END)

# Safety check for iloc[0] errors
if df_close.empty:
    st.error("No data available for the selected tickers.")
    st.stop()

# 2. Historical Viz
st.subheader("1. Historical Performance (Baseline)")
fig, ax = plt.subplots(figsize=(12,4))
for t in tickers:
    if t in df_close.columns:
        ax.plot(df_close[t] / df_close[t].iloc[0] * 100, label=t)
ax.set_title("Normalized Prices (Base 100)")
ax.legend(); ax.grid(alpha=0.3)
st.pyplot(fig)

# 3. Generating Forecasts
with st.spinner("AI is thinking... (Running Prophet Models)"):
    rolling_forecast_df = pd.DataFrame()
    for t in tickers:
        rolling_forecast_df[t] = rolling_prophet_price_forecast(df_close[t])

    forecast_2026_df = pd.DataFrame()
    last_known_prices = realized_close.iloc[-1] if not realized_close.empty else df_close.iloc[-1]
    
    for t in tickers:
        forecast_2026_df[t] = prophet_price_forecast_2026(
            df_close[t], "2026-01-01", horizon_months, last_known_prices[t]
        )

# 4. Accuracy & Comparison
st.header("2. 🔮 Forecast vs Realized")
for t in tickers:
    fig, ax = plt.subplots(figsize=(10,4))
    if not realized_close.empty:
        ax.plot(realized_close[t], label="Realized (Actual)", color="black", linewidth=2)
    ax.plot(rolling_forecast_df[t], label="Prophet 24-25 (Rolling)", linestyle="--")
    ax.plot(forecast_2026_df[t], label="Prophet 2026 (Future)", linestyle=":")
    ax.set_title(f"Price Prediction: {t}")
    ax.legend(); ax.grid(alpha=0.2)
    st.pyplot(fig)

# 5. Portfolio Optimization Logic
returns_hist = make_returns(df_close)
returns_roll_fc = make_returns(rolling_forecast_df)
returns_real = make_returns(realized_close)

# Optimization inputs
mu_hist, cov_hist = returns_hist.mean().values, returns_hist.cov().values
mu_fc, cov_fc = returns_roll_fc.mean().values, returns_roll_fc.cov().values
mu_real, cov_real = returns_real.mean().values, returns_real.cov().values

w_hist = optimize_portfolio(mu_hist, cov_hist)
w_fc = optimize_portfolio(mu_fc, cov_fc)
w_real = optimize_portfolio(mu_real, cov_real)

# 6. Results Display
st.header("3. ⚖️ Optimization Results")
cols = st.columns(2)

with cols[0]:
    st.subheader("Optimal Weights")
    w_df = pd.DataFrame({
        "Ticker": tickers,
        "Historical (%)": w_hist * 100,
        "AI Forecast (%)": w_fc * 100,
        "Actual Best (%)": w_real * 100
    }).set_index("Ticker")
    st.dataframe(w_df.round(1).style.background_gradient(cmap="Greens"))

with cols[1]:
    st.subheader("Performance Metrics")
    # Simplify for display
    metrics = {
        "Metric": ["Sharpe Ratio", "Max Drawdown", "Annual Return"],
        "AI Portfolio": [
            round(sharpe_annualized(returns_roll_fc.dot(w_fc)), 2),
            f"{round(max_drawdown((1+returns_roll_fc.dot(w_fc)).cumprod())*100,1)}%",
            f"{round(returns_roll_fc.dot(w_fc).mean()*252*100,1)}%"
        ]
    }
    st.table(pd.DataFrame(metrics))

st.success("App updated successfully with Hybrid Data Logic!")