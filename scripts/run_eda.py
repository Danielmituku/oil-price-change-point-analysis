"""
Script to run Exploratory Data Analysis on Brent Oil Prices.
Generates statistics and findings for the interim report.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

def load_data():
    """Load Brent oil price data."""
    filepath = DATA_RAW / "BrentOilPrices.csv"
    df = pd.read_csv(filepath)
    # Handle mixed date formats
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    return df

def compute_derived_features(df):
    """Add derived features."""
    df = df.copy()
    df['log_return'] = np.log(df['Price']) - np.log(df['Price'].shift(1))
    df['return'] = df['Price'].pct_change()
    df['rolling_mean_30'] = df['Price'].rolling(window=30).mean()
    df['rolling_std_30'] = df['Price'].rolling(window=30).std()
    df['rolling_volatility_30'] = df['log_return'].rolling(window=30).std() * np.sqrt(252)
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    return df

def test_stationarity(series, name):
    """Perform ADF and KPSS tests."""
    results = {}
    
    # ADF Test
    adf_result = adfuller(series.dropna(), autolag='AIC')
    results['adf_statistic'] = adf_result[0]
    results['adf_pvalue'] = adf_result[1]
    results['adf_stationary'] = adf_result[1] < 0.05
    
    # KPSS Test
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    results['kpss_statistic'] = kpss_result[0]
    results['kpss_pvalue'] = kpss_result[1]
    results['kpss_stationary'] = kpss_result[1] > 0.05
    
    return results

def main():
    print("="*70)
    print("BRENT OIL PRICE - EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Load data
    df = load_data()
    df = compute_derived_features(df)
    
    # Basic statistics
    print("\n1. DATA OVERVIEW")
    print("-"*50)
    print(f"Total observations: {len(df):,}")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Trading days: {(df['Date'].max() - df['Date'].min()).days:,} calendar days")
    print(f"Missing values: {df['Price'].isna().sum()}")
    
    print("\n2. PRICE STATISTICS")
    print("-"*50)
    print(f"Mean price: ${df['Price'].mean():.2f}")
    print(f"Median price: ${df['Price'].median():.2f}")
    print(f"Std deviation: ${df['Price'].std():.2f}")
    print(f"Minimum price: ${df['Price'].min():.2f} ({df.loc[df['Price'].idxmin(), 'Date'].strftime('%Y-%m-%d')})")
    print(f"Maximum price: ${df['Price'].max():.2f} ({df.loc[df['Price'].idxmax(), 'Date'].strftime('%Y-%m-%d')})")
    print(f"Price range: ${df['Price'].max() - df['Price'].min():.2f}")
    
    # Percentiles
    print("\n3. PRICE PERCENTILES")
    print("-"*50)
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: ${df['Price'].quantile(p/100):.2f}")
    
    # Log returns statistics
    log_returns = df['log_return'].dropna()
    print("\n4. LOG RETURNS STATISTICS")
    print("-"*50)
    print(f"Mean daily log return: {log_returns.mean()*100:.4f}%")
    print(f"Std deviation: {log_returns.std()*100:.4f}%")
    print(f"Annualized volatility: {log_returns.std() * np.sqrt(252) * 100:.2f}%")
    print(f"Skewness: {log_returns.skew():.4f}")
    print(f"Kurtosis: {log_returns.kurtosis():.4f}")
    print(f"Min daily return: {log_returns.min()*100:.2f}%")
    print(f"Max daily return: {log_returns.max()*100:.2f}%")
    
    # Stationarity tests
    print("\n5. STATIONARITY TESTS")
    print("-"*50)
    
    print("\nRaw Prices:")
    price_stationarity = test_stationarity(df['Price'], 'Price')
    print(f"  ADF test statistic: {price_stationarity['adf_statistic']:.4f}")
    print(f"  ADF p-value: {price_stationarity['adf_pvalue']:.4f}")
    print(f"  ADF conclusion: {'Stationary' if price_stationarity['adf_stationary'] else 'Non-stationary'}")
    print(f"  KPSS test statistic: {price_stationarity['kpss_statistic']:.4f}")
    print(f"  KPSS p-value: {price_stationarity['kpss_pvalue']:.4f}")
    print(f"  KPSS conclusion: {'Stationary' if price_stationarity['kpss_stationary'] else 'Non-stationary'}")
    
    print("\nLog Returns:")
    return_stationarity = test_stationarity(df['log_return'], 'Log Returns')
    print(f"  ADF test statistic: {return_stationarity['adf_statistic']:.4f}")
    print(f"  ADF p-value: {return_stationarity['adf_pvalue']:.4f}")
    print(f"  ADF conclusion: {'Stationary' if return_stationarity['adf_stationary'] else 'Non-stationary'}")
    print(f"  KPSS test statistic: {return_stationarity['kpss_statistic']:.4f}")
    print(f"  KPSS p-value: {return_stationarity['kpss_pvalue']:.4f}")
    print(f"  KPSS conclusion: {'Stationary' if return_stationarity['kpss_stationary'] else 'Non-stationary'}")
    
    # Yearly statistics
    print("\n6. YEARLY AVERAGE PRICES")
    print("-"*50)
    yearly_stats = df.groupby('year')['Price'].agg(['mean', 'std', 'min', 'max'])
    for year in yearly_stats.index:
        row = yearly_stats.loc[year]
        print(f"  {year}: Mean=${row['mean']:.2f}, Std=${row['std']:.2f}, Range=[${row['min']:.2f}-${row['max']:.2f}]")
    
    # Volatility by decade
    print("\n7. VOLATILITY BY DECADE")
    print("-"*50)
    df['decade'] = (df['year'] // 10) * 10
    decade_vol = df.groupby('decade')['log_return'].std() * np.sqrt(252) * 100
    for decade, vol in decade_vol.items():
        print(f"  {decade}s: {vol:.2f}% annualized volatility")
    
    # Extreme events
    print("\n8. EXTREME PRICE MOVEMENTS (Top 10)")
    print("-"*50)
    df_sorted = df.nlargest(10, 'log_return')[['Date', 'Price', 'log_return']]
    print("Largest daily gains:")
    for _, row in df_sorted.iterrows():
        print(f"  {row['Date'].strftime('%Y-%m-%d')}: +{row['log_return']*100:.2f}% (${row['Price']:.2f})")
    
    print("\nLargest daily losses:")
    df_sorted = df.nsmallest(10, 'log_return')[['Date', 'Price', 'log_return']]
    for _, row in df_sorted.iterrows():
        print(f"  {row['Date'].strftime('%Y-%m-%d')}: {row['log_return']*100:.2f}% (${row['Price']:.2f})")
    
    print("\n" + "="*70)
    print("EDA COMPLETE")
    print("="*70)
    
    # Save processed data
    output_path = PROJECT_ROOT / "data" / "processed" / "brent_oil_processed.csv"
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    main()
