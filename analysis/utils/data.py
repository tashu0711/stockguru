

# import yfinance as yf     #yaahoo ki finance library stock ke lie
# import pandas as pd         #pandas tabular manipulation


# yf.ticker too many request, so other is with yf.download
# def get_stock_data(ticker, period='1y'):     #ticker= copany ka unique id in stock market
#     """
#     Fetch historical stock data from Yahoo Finance
#     Args:
#         ticker (str): Stock symbol like 'AAPL', 'TSLA', 'INFY.NS'
#         period (str): Time period like '1mo', '3mo', '6mo', '1y', etc.
#     Returns:
#         DataFrame with Date and OHLCV
#     """
#     try:
#         stock = yf.Ticker(ticker)
#         df = stock.history(period=period)
#         if df.empty:
#             print(f"No data found for {ticker}")
        
#         print("*"*40)
#         print(f"Data for {ticker}")
#         print(df.head())
#         print("*"*40)
#         return df
#     except Exception as e:
#         print(f"Error fetching {ticker}: {e}")
#         return pd.DataFrame()                      #


import time
import yfinance as yf
import pandas as pd

# def get_stock_data(ticker, period='1y', retries=3, delay=5):
#     for attempt in range(retries):
#         try:
#             df = yf.download(ticker, period=period, progress=False)
#             if not df.empty:
#                 df.reset_index(inplace=True)
#                 return df
#         except Exception as e:
#             print("Fetch error:", e)
#         print(f"Retry {attempt+1}/{retries} after {delay}s …")
#         time.sleep(delay)          # ⏲️  wait before next try
#     return pd.DataFrame()          # still empty → give up



import time, pandas as pd, yfinance as yf

def get_stock_data(ticker, period='1y', retries=3, delay=5):
    """
    Robust fetch with retry + empty‑DF guard
    """
    for i in range(retries):
        try:
            df = yf.download(ticker, period=period, progress=False, threads=False)
            if not df.empty:
                df.reset_index(inplace=True)
                # कइ बार Yahoo lowercase colnames दे देता है
                df.rename(columns=str.title, inplace=True)  # 'close' → 'Close'
                return df
        except Exception as e:
            print("Fetch error:", e)
        print(f"Retry {i+1}/{retries} after {delay}s …")
        time.sleep(delay)

    # still failed
    return pd.DataFrame()
