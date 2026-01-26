
import os
from typing import List
import pandas as pd
import yfinance as yf


def download_data(tickers: List[str], start_date: str, end_date: str, cache_file: str) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False
    )
    data.to_csv(cache_file)
    return data


def load_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    freq: str = "daily",
    cache_file: str = "stock_data.csv"
) -> pd.DataFrame:

    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print("📁 Cache yüklendi.")
    else:
        print("📊 Veri indiriliyor...")
        data = download_data(tickers, start_date, end_date, cache_file)
        print("💾 Cache kaydedildi.")

    # Close fiyatlarını seç
    df_close = data["Close"]

    # Resample
    if freq == "daily":
        return df_close
    elif freq == "monthly":
        return df_close.resample("M").last().dropna()
    elif freq == "yearly":
        return df_close.resample("Y").last().dropna()
    else:
        raise ValueError("freq must be one of ['daily','monthly','yearly']")
