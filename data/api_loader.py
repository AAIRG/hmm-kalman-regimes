import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
from .base_loader import BaseDataLoader


class YFinanceLoader(BaseDataLoader):
    """
    Data loader implementation using Yahoo Finance (yfinance).
    """

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical daily data from Yahoo Finance.

        Args:
            symbol: The ticker symbol to fetch (e.g., 'SPY').
            start_date: The start date in YYYY-MM-DD format.
            end_date: The end date in YYYY-MM-DD format.

        Returns:
            A pandas DataFrame containing the downloaded data.
        """
        data: pd.DataFrame = yf.download(symbol, start=start_date, end=end_date, progress=False)
        return data

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data by calculating log returns from the price.
        Uses 'Adj Close' if available, otherwise falls back to 'Close'.

        Args:
            data: The raw DataFrame downloaded from yfinance.

        Returns:
            A pandas DataFrame with a new 'Log_Return' column, stripped of initial NaN values.
        """
        if data.empty:
            return data

        # Use Adjusted Close for more accurate return calculation if available
        price_col: str = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        
        # Isolate the price series
        prices: pd.Series = data[price_col]

        # Calculate logarithmic returns: ln(P_t / P_{t-1})
        data['Log_Return'] = np.log(prices / prices.shift(1))
        
        # Drop rows where Log_Return is NaN (typically the first row)
        data = data.dropna(subset=['Log_Return'])
        
        return data
