📈 AI Portfolio Optimizer – Prophet

🌟 Quick Overview (STAR Method)
Situation: Financial markets require both precise price prediction and robust risk management strategies.

Task: Develop a dual-track project to evaluate multiple forecasting models and apply them to real-world portfolio optimization.

Action: Benchmarked ARIMA, Prophet, and XGBoost for price forecasting. Implemented a rolling Prophet model within a Streamlit dashboard to optimize Sharpe-ratio-based portfolios.

Result: Identified XGBoost as the leader for raw price accuracy, while Prophet provided the most stable return distributions for portfolio allocation.
...

📂 Project Structure

portfolio-forecasting/
├── price_prediction_models/    # PART A: Raw Price Forecasting Research
│   ├── arimastockprice.ipynb   # ARIMA Experimentation
│   ├── prophetstockprice.ipynb # Prophet Experimentation
│   └── xgbooststockprice.ipynb # XGBoost Experimentation (Best Price Accuracy)
├── portfolio_forecasting/      # PART B: Portfolio Optimization Apps
│   ├── prophetfinal.py         # Main Streamlit App (Prophet Implementation)
│   └── arimafinal.py           # Alternative Streamlit App (ARIMA Implementation)
├── data/                       # Dataset Storage
│   └── stockdate.csv           # Historical Stock Price Data
├── requirements.txt            # Project Dependencies
└── README.md                   # Documentation
...

💡 Key Insights
Dual-Track Evaluation: This project demonstrates that the best model for point-in-time price prediction (XGBoost) may differ from the best model for capturing return distributions for optimization (Prophet).

Rolling Forecast (2024-2025): The model retrains every month, simulating a professional environment where strategies are updated as new data becomes available.

Diversification Guardrails: Individual asset weights are capped between 1% and 40% to prevent extreme concentration risk.

Risk Management: Focuses on Sortino Ratio and Max Drawdown to measure downside protection and "pain-adjusted" returns.
...

🧠 How It Works

1️⃣ Part A: Price Forecasting Models (/price_prediction_models)
We evaluated raw price prediction accuracy across three distinct architectures:

ARIMA: Traditional statistical baseline for time-series.

Prophet: Specialist in handling seasonality and market changepoints.

XGBoost: Winner for Price Accuracy. Delivered the lowest error rates (RMSE/MAPE) for direct price forecasting.

2️⃣ Part B: Forecasting for Portfolio Optimization (/portfolio_forecasting)
This track focuses on generating expected returns (mu) and risk matrices (cov):

Prophet: Winner for Optimization. Provides stable return distributions essential for Mean-Variance Optimization.

Arima (Final): Alternative implementation for comparative benchmarking.

2026 Projection: Long-term projection starting from the end of 2025 using the last known market prices.

3️⃣ Portfolio Optimization Engine
Comparing three distinct strategies:

Historical-based: Optimal allocation using 2016-2023 data.

Forecast-based: AI-predicted allocation for the future.

Realized: Benchmarking against actual 2024-2025 market performance.

...

📌 Features
✔ Interactive Dashboards: Dynamic ticker selection and forecast horizon sliders.

✔ Comparative Frontiers: Visual Efficient Frontier analysis across different time windows.

✔ Accuracy Tracking: Live RMSE and MAPE metrics for forecast validation.

✔ Live Apps: Accessible via Streamlit Cloud for real-time interaction.

...

📦 Installation & Run
Bash
# Clone the repository
git clone https://github.com/esramogulkoc-dev/portfolio-forecasting
cd portfolio-forecasting

# Install dependencies
pip install -r requirements.txt

# Run the Prophet Optimizer (Main App)
streamlit run prophetfinal.py

# Run the ARIMA Optimizer (Alternative)
streamlit arimafinal.py