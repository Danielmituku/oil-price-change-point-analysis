"""
Script to download Brent oil price data.

Note: You may need to manually download the data from a source like:
- EIA (U.S. Energy Information Administration)
- Fred (Federal Reserve Economic Data)
- Yahoo Finance

This script provides a template for loading and preparing the data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def create_sample_data():
    """
    Create sample Brent oil price data for demonstration.
    
    In production, replace this with actual data from:
    - EIA API: https://www.eia.gov/opendata/
    - FRED: https://fred.stlouisfed.org/series/DCOILBRENTEU
    - Yahoo Finance: Ticker BZ=F
    """
    logger.info("Creating sample data for demonstration...")
    
    # Generate sample dates
    dates = pd.date_range(start='1987-05-20', end='2022-09-30', freq='B')
    
    # Generate realistic-looking price series with trends and volatility
    np.random.seed(42)
    n = len(dates)
    
    # Base trend
    trend = np.linspace(15, 80, n)
    
    # Add some regime changes
    prices = trend.copy()
    
    # Asian crisis drop (1997-1998)
    crisis_1 = (dates >= '1997-07-01') & (dates <= '1999-01-01')
    prices[crisis_1] *= 0.7
    
    # 2008 spike and crash
    spike_2008 = (dates >= '2007-01-01') & (dates <= '2008-07-15')
    prices[spike_2008] *= 1.8
    crash_2008 = (dates >= '2008-07-15') & (dates <= '2009-01-01')
    prices[crash_2008] *= 0.4
    
    # 2014 OPEC price war
    crash_2014 = (dates >= '2014-06-01') & (dates <= '2016-01-01')
    prices[crash_2014] *= 0.6
    
    # COVID crash
    covid = (dates >= '2020-03-01') & (dates <= '2020-05-01')
    prices[covid] *= 0.4
    
    # Add random noise
    noise = np.random.normal(0, 3, n)
    prices += noise
    
    # Ensure positive prices
    prices = np.maximum(prices, 10)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates.strftime('%d-%b-%y'),
        'Price': np.round(prices, 2)
    })
    
    return df


def load_data_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load Brent oil data from a CSV file.
    
    Expected format:
    - Date column: 'dd-mmm-yy' format (e.g., '20-May-87')
    - Price column: numeric USD per barrel
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Validate columns
    if 'Date' not in df.columns or 'Price' not in df.columns:
        raise ValueError("CSV must have 'Date' and 'Price' columns")
    
    return df


def main():
    """Main function to download/create and save data."""
    
    # Ensure directories exist
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Check if real data exists
    real_data_path = DATA_RAW / "brent_oil_prices.csv"
    
    if real_data_path.exists():
        logger.info(f"Data already exists at {real_data_path}")
        df = load_data_from_csv(real_data_path)
    else:
        logger.warning("No real data found. Creating sample data for demonstration.")
        logger.info("To use real data, download from EIA, FRED, or Yahoo Finance")
        logger.info("and save to: data/raw/brent_oil_prices.csv")
        
        df = create_sample_data()
        df.to_csv(real_data_path, index=False)
        logger.info(f"Sample data saved to {real_data_path}")
    
    # Display summary
    print("\n" + "="*60)
    print("Data Summary")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
    print(f"Mean price: ${df['Price'].mean():.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
