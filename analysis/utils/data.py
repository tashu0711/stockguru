# import yfinance as yf
# import pandas as pd


import yfinance as yf     #yaahoo ki finance library stock ke lie
import pandas as pd         #pandas tabular manipulation

def get_stock_data(ticker='AAPL', period='1mo'):     #ticker= copany ka unique id in stock market
    """
    Fetch historical stock data from Yahoo Finance
    Args:
        ticker (str): Stock symbol like 'AAPL', 'TSLA', 'INFY.NS'
        period (str): Time period like '1mo', '3mo', '6mo', '1y', etc.
    Returns:
        DataFrame with Date and OHLCV
    """
    try:
        stock = yf.Ticker(ticker)                    #ticker ka object bana lia
        df = stock.history(period=period)             #history ticker ke anadar ka ek function jo poora history dega 
        df.reset_index(inplace=True)                # Date ko ek column mein laa dete hain
        return df
    except Exception as e:                          #agar kuch error aata jaise ki internet yaa no data to
        print("Error fetching data:", e)             
        return pd.DataFrame()                      #

