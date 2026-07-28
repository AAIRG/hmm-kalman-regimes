import pandas as pd
import numpy as np
from pathlib import Path
import logging
from .base_loader import BaseDataLoader

logger = logging.getLogger(__name__)


class CSVLoader(BaseDataLoader):
    """
    Data loader implementation to read from a local CSV file.
    """

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Loads historical data from a CSV file.
        In this implementation, `symbol` acts as the file path to the CSV.

        Args:
            symbol: The file path to the CSV file (e.g., 'data/sample_data.csv').
            start_date: The start date to filter by in YYYY-MM-DD format.
            end_date: The end date to filter by in YYYY-MM-DD format.

        Returns:
            A pandas DataFrame containing the historical data filtered by date.
        """
        logger.info(f"Loading CSV data from {symbol} between {start_date} and {end_date}")
        file_path = Path(symbol)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found at: {file_path}")

        data: pd.DataFrame = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        data = data.sort_index()

        mask = (data.index >= pd.to_datetime(start_date)) & (data.index <= pd.to_datetime(end_date))
        filtered_data: pd.DataFrame = data.loc[mask]

        logger.info(f"Loaded {len(filtered_data)} rows after filtering.")
        return filtered_data

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data by calculating log returns from the price.
        Uses 'Adj Close' if available, otherwise falls back to 'Close'.

        Args:
            data: The raw DataFrame read from the CSV.

        Returns:
            A pandas DataFrame with a new 'Log_Return' column, stripped of initial NaN values.
        """
        if data.empty:
            return data

        price_col: str = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        if price_col not in data.columns:
            raise KeyError(f"Expected price column '{price_col}' not found in the CSV data.")

        prices: pd.Series = data[price_col]

        data['Log_Return'] = np.log(prices / prices.shift(1))
        data = data.dropna(subset=['Log_Return'])
        
        logger.info(f"Preprocessed {len(data)} rows (calculated log returns).")
        return data
