"""
Data loading utilities for Brent oil price analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_brent_oil_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load Brent oil price data from CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file containing Brent oil prices
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Date and Price columns
    """
    logger.info(f"Loading data from {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime (handle mixed formats)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Ensure Price is numeric
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    logger.info(f"Loaded {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def load_events_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load major events data from CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file containing event data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with event information
    """
    logger.info(f"Loading events from {filepath}")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded {len(df)} events")
    
    return df


def compute_log_returns(df: pd.DataFrame, price_col: str = 'Price') -> pd.Series:
    """
    Compute log returns from price series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data
    price_col : str
        Name of the price column
        
    Returns
    -------
    pd.Series
        Log returns series
    """
    log_returns = np.log(df[price_col]) - np.log(df[price_col].shift(1))
    return log_returns


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the price data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Date and Price columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional derived features
    """
    df = df.copy()
    
    # Log returns
    df['log_return'] = compute_log_returns(df)
    
    # Simple returns
    df['return'] = df['Price'].pct_change()
    
    # Rolling statistics
    df['rolling_mean_30'] = df['Price'].rolling(window=30).mean()
    df['rolling_std_30'] = df['Price'].rolling(window=30).std()
    df['rolling_volatility_30'] = df['log_return'].rolling(window=30).std() * np.sqrt(252)
    
    # Time features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    
    # Time index (for modeling)
    df['time_idx'] = np.arange(len(df))
    
    return df


def prepare_data_for_modeling(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Prepare data for Bayesian change point modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Date and Price columns
    start_date : str, optional
        Start date for analysis period
    end_date : str, optional
        End date for analysis period
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]
        prices array, time indices, and dates
    """
    df_filtered = df.copy()
    
    if start_date:
        df_filtered = df_filtered[df_filtered['Date'] >= start_date]
    if end_date:
        df_filtered = df_filtered[df_filtered['Date'] <= end_date]
    
    df_filtered = df_filtered.reset_index(drop=True)
    
    prices = df_filtered['Price'].values
    time_idx = np.arange(len(prices))
    dates = df_filtered['Date']
    
    return prices, time_idx, dates


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for the price data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data
        
    Returns
    -------
    dict
        Summary statistics
    """
    return {
        'start_date': df['Date'].min().strftime('%Y-%m-%d'),
        'end_date': df['Date'].max().strftime('%Y-%m-%d'),
        'n_observations': len(df),
        'mean_price': df['Price'].mean(),
        'std_price': df['Price'].std(),
        'min_price': df['Price'].min(),
        'max_price': df['Price'].max(),
        'missing_values': df['Price'].isna().sum()
    }
