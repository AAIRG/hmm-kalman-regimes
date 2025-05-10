from abc import ABC, abstractmethod
import pandas as pd


class BaseDataLoader(ABC):
    """
    Abstract Base Class defining the interface for data loaders.
    """

    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a given symbol.

        Args:
            symbol: The ticker symbol to fetch (e.g., 'SPY').
            start_date: The start date in YYYY-MM-DD format.
            end_date: The end date in YYYY-MM-DD format.

        Returns:
            A pandas DataFrame containing the historical data.
        """
        pass

    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the fetched data (e.g., calculating log returns, handling NAs).

        Args:
            data: The raw pandas DataFrame.

        Returns:
            A pandas DataFrame containing the preprocessed data.
        """
        pass
