import yfinance as yf
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def download_sample_data(symbol: str = "SPY", years: int = 5) -> None:
    """
    Downloads historical data for a given symbol and saves it to a CSV file.

    Args:
        symbol: The ticker symbol to download (default: 'SPY').
        years: The number of years of historical data to fetch (default: 5).
    """
    logger.info(f"Downloading {years} years of data for {symbol}...")
    
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    
    data: pd.DataFrame = yf.download(
        symbol, 
        start=start_date.strftime('%Y-%m-%d'), 
        end=end_date.strftime('%Y-%m-%d'), 
        progress=False
    )
    
    if data.empty:
        logger.error(f"Failed to download data for {symbol}.")
        return

    # Flatten the MultiIndex columns returned by newer versions of yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_dir / "sample_data.csv"
    data.to_csv(output_path)
    
    logger.info(f"Successfully saved {len(data)} rows to {output_path}")


if __name__ == "__main__":
    setup_logging()
    download_sample_data()
