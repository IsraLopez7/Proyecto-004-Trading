"""
Script to download historical data for QQQ using yfinance
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def download_stock_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    years: int = 15
) -> pd.DataFrame:
    """
    Download historical stock data using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., 'QQQ')
        start_date: Start date (YYYY-MM-DD) or None for auto
        end_date: End date (YYYY-MM-DD) or None for today
        years: Number of years to download if start_date is None

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Downloading data for {ticker}...")

    # Calculate dates
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')

    logger.info(f"Date range: {start_date} to {end_date}")

    # Download data
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False
    )

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Reset index to make Date a column
    data = data.reset_index()

    logger.info(f"Downloaded {len(data)} rows")
    logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]

    return data


def save_data(df: pd.DataFrame, output_path: str):
    """Save data to CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    logger.info(f"Data saved to {output_path}")


def main():
    """Main execution function."""
    logger.info("Starting data download...")

    # Load config
    config = load_config()

    ticker = config['data']['ticker']

    # Download data
    df = download_stock_data(ticker, years=15)

    # Save data
    output_path = f"data/raw/{ticker}.csv"
    save_data(df, output_path)

    # Print summary
    logger.info("\nData Summary:")
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Years: {(df['Date'].max() - df['Date'].min()).days / 365.25:.1f}")

    logger.info("\nFirst few rows:")
    logger.info(f"\n{df.head()}")

    logger.info("\nData download complete!")


if __name__ == "__main__":
    main()
