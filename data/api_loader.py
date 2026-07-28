import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
import logging
from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


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
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date} via yfinance...")
        data: pd.DataFrame = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        logger.info(f"Successfully fetched {len(data)} rows for {symbol}.")
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

        price_col: str = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        prices: pd.Series = data[price_col]

        data['Log_Return'] = np.log(prices / prices.shift(1))
        data = data.dropna(subset=['Log_Return'])
        
        logger.info(f"Preprocessed {len(data)} rows (calculated log returns).")
        return data
