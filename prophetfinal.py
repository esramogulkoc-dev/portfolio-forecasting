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

# =============================
# DATE CONFIG
# =============================
TRAIN_START = "2016-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"  # Realized test window

# =============================
# DATA DOWNLOAD
# =============================
@st.cache_data
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    return data.dropna()

# =============================
# SHARPE FUNCTION
# =============================
def sharpe_annualized(returns, freq=252):
    mean = returns.mean() * freq
    vol = returns.std() * np.sqrt(freq)
    return mean / vol if vol != 0 else 0

# =============================
# SORTINO FUNCTION
# =============================
def sortino_annualized(returns, freq=252, mar=0):
    """
    mar = minimum acceptable return (0 by default)
    """
    mean = returns.mean() * freq
    downside = returns[returns < mar]
    downside_std = downside.std() * np.sqrt(freq)
    return mean / downside_std if downside_std != 0 else 0

# =============================
# MAX DRAWDOWN FUNCTION
# =============================
def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

# =============================
# RETURNS
# =============================
def make_returns(price):
    return np.log(price / price.shift(1)).dropna()

# =============================
# PROPHET
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


# =============================
# ROLLING FORECAST (FIXED)
# =============================
def rolling_prophet_price_forecast(price_series):
    returns = make_returns(price_series)
    preds = []

    retrain_dates = pd.date_range(TEST_START, TEST_END, freq="MS")

    last_price = price_series.loc[:TRAIN_END].iloc[-1]

    for d in retrain_dates:
        train_ret = returns.loc[:d - pd.Timedelta(days=1)]
        if len(train_ret) < 200:
            continue

        model = train_prophet_return(train_ret)

        future_dates = pd.bdate_range(
            d,
            min(d + pd.offsets.MonthEnd(1), pd.to_datetime(TEST_END))
        )

        future = pd.DataFrame({"ds": future_dates})
        ret_fc = model.predict(future)["yhat"].values

        prices = [last_price]
        for r in ret_fc:
            prices.append(prices[-1] * np.exp(r))

        price_fc = pd.Series(prices[1:], index=future_dates)
        last_price = price_fc.iloc[-1]
        preds.append(price_fc)

    return pd.concat(preds).groupby(level=0).mean()

# =============================
# ONE-TIME FORECAST FOR 2026
# =============================
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

    res = minimize(
        neg_sharpe,
        np.ones(n)/n,
        bounds=bounds,
        constraints=cons,
        method="SLSQP"
    )
    return res.x

# =============================
# EFFICIENT FRONTIER
# =============================
def plot_ef_separate(hist, forecast, realized, min_w=0.01, max_w=0.4):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data, title in zip(
        axes,
        [hist, forecast, realized],
        ["Historical (Train)", "Forecast (Prophet)", "Realized"]
    ):
        mu = expected_returns.mean_historical_return(data, frequency=252)
        S = risk_models.sample_cov(data, frequency=252)
        ef = EfficientFrontier(mu, S, weight_bounds=(min_w, max_w))
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(layout="wide")
st.title("📈 AI Portfolio Optimizer – Prophet")

with st.sidebar:
    tickers = st.multiselect(
        "Select tickers",
        ["AAPL","MSFT","GOOG","AMZN","NVDA","META","TSLA","JPM","V","WMT"],
        default=["AAPL","MSFT","GOOG"]
    )
    horizon_months = st.slider(
        "Forecast Horizon (months) for 2026",
        min_value=3,
        max_value=12,
        value=3
    )
    if len(tickers) < 2:
        st.stop()

# =============================
# HISTORICAL DATA
# =============================
df_close = download_data(tickers, TRAIN_START, TRAIN_END)

# 1) View historical data
with st.expander("👁️ View Historical Data", expanded=False):
    st.dataframe(df_close.head(10))
    st.write(f"Shape: {df_close.shape}")

fig, ax = plt.subplots(figsize=(12,5))
for t in tickers:
    ax.plot(df_close[t] / df_close[t].iloc[0] * 100, label=t)
ax.set_title("Historical Prices (Normalized)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# =============================
# REALIZED DATA
# =============================
realized_close = download_data(tickers, TEST_START, TEST_END)

# 2) View realized data
with st.expander("👁️ View Realized Data (2024-2025)", expanded=False):
    st.dataframe(realized_close.head(10))
    st.dataframe(realized_close.tail(10))
    st.write(f"Shape: {realized_close.shape}")

# =============================
# FORECAST
# =============================
rolling_forecast_df = pd.DataFrame()
for t in tickers:
    rolling_forecast_df[t] = rolling_prophet_price_forecast(df_close[t])

# =============================
# 2026 FORECAST (START PRICE = 2025-12-31)
# =============================
forecast_df = pd.DataFrame()
last_known_prices = realized_close.iloc[-1]

for t in tickers:
    forecast_df[t] = prophet_price_forecast_2026(
        df_close[t],
        start_date="2026-01-01",
        horizon_months=horizon_months,
        last_known_price=last_known_prices[t]
    )

# =============================
# FORECAST VS REALIZED
# =============================
st.header("🔮 Forecast vs Realized")

for t in tickers:
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(realized_close[t], label="Realized", linewidth=3, color="black")
    ax.plot(rolling_forecast_df[t], label="Prophet Rolling (2024-2025)", linestyle="--")
    ax.plot(forecast_df[t], label="Prophet 2026 Forecast", linestyle=":")
    ax.set_title(t)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# =============================
# ACCURACY
# =============================
acc = []
for t in tickers:
    idx = realized_close.index.intersection(rolling_forecast_df.index)
    y = realized_close[t].loc[idx]
    yhat = rolling_forecast_df[t].loc[idx]

    acc.append({
        "Ticker": t,
        "RMSE": np.sqrt(mean_squared_error(y, yhat)),
        "MAPE (%)": np.mean(np.abs((y - yhat) / y)) * 100
    })

st.subheader("🎯 Accuracy (2024-2025 Rolling Forecast)")
st.dataframe(pd.DataFrame(acc).set_index("Ticker"))

# =============================
# PORTFOLIO RETURNS
# =============================
returns_hist = make_returns(df_close)

# Rolling forecast returns
returns_roll_fc = make_returns(rolling_forecast_df)
returns_real = make_returns(realized_close)

common_roll = returns_roll_fc.index.intersection(returns_real.index)
returns_roll_fc = returns_roll_fc.loc[common_roll]
returns_real = returns_real.loc[common_roll]

# =============================
# 2026 forecast returns (DÜZELTİLDİ)
# =============================
forecast_with_prev = pd.concat([realized_close.iloc[[-1]], forecast_df])
returns_fc_2026 = np.log(forecast_with_prev / forecast_with_prev.shift(1)).dropna()
returns_fc_2026 = returns_fc_2026.loc["2026-01-01":]

# =============================
# PORTFOLIO WEIGHTS (Historical / Forecast / Realized)
# =============================
mu_hist = returns_hist.mean().values
cov_hist = returns_hist.cov().values

mu_roll_fc = returns_roll_fc.mean().values
cov_roll_fc = returns_roll_fc.cov().values

mu_real = returns_real.mean().values
cov_real = returns_real.cov().values

mu_fc_2026 = returns_fc_2026.mean().values
cov_fc_2026 = returns_fc_2026.cov().values

w_hist = optimize_portfolio(mu_hist, cov_hist)
w_roll_fc = optimize_portfolio(mu_roll_fc, cov_roll_fc)
w_real = optimize_portfolio(mu_real, cov_real)
w_fc_2026 = optimize_portfolio(mu_fc_2026, cov_fc_2026)

weights_df = pd.DataFrame({
    "Ticker": tickers,
    "Historical (%)": w_hist * 100,
    "Forecast 2024-2025 (%)": w_roll_fc * 100,
    "Realized 2024-2025 (%)": w_real * 100,
    "Forecast 2026 (%)": w_fc_2026 * 100
}).set_index("Ticker")

# 3) Portfolio weights table
st.subheader("⚖️ Portfolio Weights")
st.dataframe(
    weights_df
    .round(1)
    .style
    .format("{:.1f}%")
    .background_gradient(cmap="RdYlGn", axis=1)
)


# =============================
# CUMULATIVE RETURNS
# =============================

cum_roll_fc = (1 + returns_roll_fc.dot(w_roll_fc)).cumprod()
cum_real = (1 + returns_real.dot(w_real)).cumprod()

cum_real_end_2025 = cum_real.iloc[-1]

cum_fc_2026 = (1 + returns_fc_2026.dot(w_fc_2026)).cumprod()
cum_fc_2026 = cum_fc_2026 * cum_real_end_2025

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(cum_roll_fc, label="Forecast Portfolio (2024-2025 Rolling)", linewidth=2)
ax.plot(cum_real, label="Historical Portfolio (2024-2025)", linewidth=2, linestyle="--")
ax.plot(cum_fc_2026, label=f"Forecast Portfolio (2026 + {horizon_months}m)", linewidth=2, linestyle=":")
ax.legend()
ax.grid(alpha=0.3)
ax.set_title("Portfolio Cumulative Returns")
st.pyplot(fig)

# =============================
# PORTFOLIO METRICS (Annual Return, Volatility, Sharpe, Sortino, MDD)
# =============================
ann_factor = 252

metrics_df = pd.DataFrame({
    "Portfolio": ["Historical", "Forecast 2024-2025", "Historical 2024-2025", "Forecast 2026"],
    "Annual Return (%)": [
        np.sum(mu_hist * w_hist) * ann_factor * 100,
        np.sum(mu_roll_fc * w_roll_fc) * ann_factor * 100,
        np.sum(mu_real * w_real) * ann_factor * 100,
        np.sum(mu_fc_2026 * w_fc_2026) * ann_factor * 100,
    ],
    "Annual Volatility (%)": [
        np.sqrt(w_hist.T @ cov_hist @ w_hist) * np.sqrt(ann_factor) * 100,
        np.sqrt(w_roll_fc.T @ cov_roll_fc @ w_roll_fc) * np.sqrt(ann_factor) * 100,
        np.sqrt(w_real.T @ cov_real @ w_real) * np.sqrt(ann_factor) * 100,
        np.sqrt(w_fc_2026.T @ cov_fc_2026 @ w_fc_2026) * np.sqrt(ann_factor) * 100,
    ],
    "Sharpe Ratio": [
        sharpe_annualized(returns_hist.dot(w_hist)),
        sharpe_annualized(returns_roll_fc.dot(w_roll_fc)),
        sharpe_annualized(returns_real.dot(w_real)),
        sharpe_annualized(returns_fc_2026.dot(w_fc_2026)),
    ],
    "Sortino Ratio": [
        sortino_annualized(returns_hist.dot(w_hist)),
        sortino_annualized(returns_roll_fc.dot(w_roll_fc)),
        sortino_annualized(returns_real.dot(w_real)),
        sortino_annualized(returns_fc_2026.dot(w_fc_2026)),
    ],
    "Max Drawdown": [
        max_drawdown((1 + returns_hist.dot(w_hist)).cumprod()),
        max_drawdown(cum_roll_fc),
        max_drawdown(cum_real),
        max_drawdown(cum_fc_2026),
    ]
})

# 5) Portfolio metrics table
st.subheader("📊 Portfolio Metrics")
st.dataframe(metrics_df.style.background_gradient(cmap="RdYlGn", subset=["Sharpe Ratio","Sortino Ratio"]))
