import yfinance as yf
import pandas as pd
import numpy as np


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data for `ticker` and return a DataFrame with:
      - close        : adjusted closing price
      - daily_return : simple daily return
      - log_return   : log daily return
      - volatility   : 20-day rolling annualised volatility of log returns
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' between {start} and {end}.")

    # yfinance returns MultiIndex columns when auto_adjust=True; flatten them
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(index=raw.index)
    df["close"] = raw["Close"]
    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["log_return"].rolling(window=20).std() * np.sqrt(252)

    return df.dropna(subset=["daily_return"])
