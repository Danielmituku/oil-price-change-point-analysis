"""
Generate EDA visualizations for the interim report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
EVENTS_FILE = PROJECT_ROOT / "events" / "major_events.csv"

# Ensure figures directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load processed data."""
    df = pd.read_csv(DATA_PROCESSED / "brent_oil_processed.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_events():
    """Load events data."""
    events = pd.read_csv(EVENTS_FILE)
    events['date'] = pd.to_datetime(events['date'])
    return events

def plot_price_series(df, events):
    """Plot 1: Full price series with major events."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df['Date'], df['Price'], 'b-', alpha=0.8, linewidth=0.8)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD/barrel)', fontsize=12)
    ax.set_title('Brent Crude Oil Prices (1987-2022) with Major Events', fontsize=14)
    
    # Add event markers
    colors = {
        'Conflict': 'red', 
        'OPEC Policy': 'green', 
        'Economic Crisis': 'orange', 
        'Sanctions': 'purple', 
        'Geopolitical': 'brown'
    }
    
    for _, event in events.iterrows():
        if event['date'] >= df['Date'].min() and event['date'] <= df['Date'].max():
            color = colors.get(event['event_type'], 'gray')
            ax.axvline(event['date'], color=color, linestyle='--', alpha=0.4, linewidth=1)
    
    # Add legend for event types
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, linestyle='--', label=t, alpha=0.7) 
                       for t, c in colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    ax.set_xlim(df['Date'].min(), df['Date'].max())
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '01_price_series_with_events.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_price_series_with_events.png")

def plot_price_distribution(df):
    """Plot 2: Price distribution and box plot by decade."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['Price'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(df['Price'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["Price"].mean():.2f}')
    axes[0].axvline(df['Price'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${df["Price"].median():.2f}')
    axes[0].set_xlabel('Price (USD/barrel)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Brent Oil Prices', fontsize=12)
    axes[0].legend()
    
    # Box plot by decade
    df['decade'] = (df['year'] // 10) * 10
    decade_order = sorted(df['decade'].unique())
    sns.boxplot(data=df, x='decade', y='Price', ax=axes[1], palette='coolwarm')
    axes[1].set_xlabel('Decade', fontsize=11)
    axes[1].set_ylabel('Price (USD/barrel)', fontsize=11)
    axes[1].set_title('Price Distribution by Decade', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '02_price_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_price_distribution.png")

def plot_yearly_average(df):
    """Plot 3: Yearly average prices."""
    yearly_avg = df.groupby('year')['Price'].mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(yearly_avg.index, yearly_avg.values, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Color bars by value
    norm = plt.Normalize(yearly_avg.min(), yearly_avg.max())
    colors = plt.cm.RdYlGn_r(norm(yearly_avg.values))
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    
    ax.axhline(yearly_avg.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Overall Mean: ${yearly_avg.mean():.2f}')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Average Price (USD/barrel)', fontsize=11)
    ax.set_title('Average Annual Brent Oil Prices', fontsize=12)
    ax.legend()
    ax.set_xticks(yearly_avg.index[::2])
    ax.set_xticklabels(yearly_avg.index[::2], rotation=45)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '03_yearly_average_prices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_yearly_average_prices.png")

def plot_rolling_stats(df):
    """Plot 4: Rolling mean and volatility."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Price with rolling mean
    axes[0].plot(df['Date'], df['Price'], 'b-', alpha=0.5, linewidth=0.8, label='Daily Price')
    axes[0].plot(df['Date'], df['rolling_mean_30'], 'r-', linewidth=2, label='30-day Rolling Mean')
    axes[0].set_ylabel('Price (USD/barrel)', fontsize=11)
    axes[0].set_title('Brent Oil Price with 30-day Rolling Mean', fontsize=12)
    axes[0].legend(loc='upper left')
    
    # Rolling volatility
    axes[1].fill_between(df['Date'], 0, df['rolling_volatility_30'], alpha=0.6, color='purple')
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Annualized Volatility', fontsize=11)
    axes[1].set_title('30-day Rolling Volatility (Annualized)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '04_rolling_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_rolling_statistics.png")

def plot_log_returns(df):
    """Plot 5: Log returns analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    log_returns = df['log_return'].dropna()
    
    # Time series of log returns
    axes[0, 0].plot(df['Date'], df['log_return'], 'b-', alpha=0.6, linewidth=0.5)
    axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Log Return')
    axes[0, 0].set_title('Daily Log Returns')
    
    # Distribution of log returns with normal overlay
    axes[0, 1].hist(log_returns, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add normal distribution overlay
    from scipy import stats
    x = np.linspace(log_returns.min(), log_returns.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, log_returns.mean(), log_returns.std()), 
                    'r-', linewidth=2, label='Normal Distribution')
    axes[0, 1].set_xlabel('Log Return')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution of Log Returns vs Normal')
    axes[0, 1].legend()
    
    # Q-Q plot
    stats.probplot(log_returns, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Log Returns vs Normal)')
    
    # Volatility clustering (squared returns)
    axes[1, 1].plot(df['Date'], df['log_return']**2, 'purple', alpha=0.6, linewidth=0.5)
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Squared Log Return')
    axes[1, 1].set_title('Squared Returns (Volatility Clustering)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05_log_returns_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 05_log_returns_analysis.png")

def plot_volatility_by_decade(df):
    """Plot 6: Volatility comparison by decade."""
    df['decade'] = (df['year'] // 10) * 10
    decade_vol = df.groupby('decade')['log_return'].std() * np.sqrt(252) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(decade_vol.index.astype(str) + 's', decade_vol.values, 
                  color=['green', 'blue', 'orange', 'purple', 'red'], 
                  edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('Decade', fontsize=11)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=11)
    ax.set_title('Market Volatility by Decade', fontsize=12)
    
    # Add value labels on bars
    for bar, val in zip(bars, decade_vol.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, max(decade_vol.values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '06_volatility_by_decade.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 06_volatility_by_decade.png")

def plot_extreme_events(df, events):
    """Plot 7: COVID-19 period zoom."""
    # Filter for COVID period
    covid_start = '2020-01-01'
    covid_end = '2020-12-31'
    df_covid = df[(df['Date'] >= covid_start) & (df['Date'] <= covid_end)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df_covid['Date'], df_covid['Price'], 'b-', linewidth=2, marker='o', markersize=2)
    ax.fill_between(df_covid['Date'], df_covid['Price'], alpha=0.3)
    
    # Mark key events
    covid_events = events[(events['date'] >= covid_start) & (events['date'] <= covid_end)]
    for _, event in covid_events.iterrows():
        ax.axvline(event['date'], color='red', linestyle='--', linewidth=2)
        ax.annotate(event['event_name'], xy=(event['date'], df_covid['Price'].max()), 
                   xytext=(10, 0), textcoords='offset points', fontsize=9,
                   rotation=90, va='top')
    
    # Mark the negative price day
    min_idx = df_covid['Price'].idxmin()
    ax.annotate(f"Min: ${df_covid.loc[min_idx, 'Price']:.2f}\n({df_covid.loc[min_idx, 'Date'].strftime('%Y-%m-%d')})", 
                xy=(df_covid.loc[min_idx, 'Date'], df_covid.loc[min_idx, 'Price']),
                xytext=(30, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price (USD/barrel)', fontsize=11)
    ax.set_title('Brent Oil Prices During COVID-19 (2020) - Extreme Volatility Period', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '07_covid_period_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 07_covid_period_analysis.png")

def plot_event_impact_2008(df, events):
    """Plot 8: 2008 Financial Crisis zoom."""
    start = '2007-01-01'
    end = '2009-12-31'
    df_period = df[(df['Date'] >= start) & (df['Date'] <= end)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df_period['Date'], df_period['Price'], 'b-', linewidth=1.5)
    ax.fill_between(df_period['Date'], df_period['Price'], alpha=0.3)
    
    # Mark key events
    period_events = events[(events['date'] >= start) & (events['date'] <= end)]
    for _, event in period_events.iterrows():
        ax.axvline(event['date'], color='red', linestyle='--', linewidth=2)
        ax.annotate(event['event_name'], xy=(event['date'], df_period['Price'].max()), 
                   xytext=(5, -10), textcoords='offset points', fontsize=9,
                   rotation=90, va='top')
    
    # Mark the peak
    max_idx = df_period['Price'].idxmax()
    ax.annotate(f"Peak: ${df_period.loc[max_idx, 'Price']:.2f}\n({df_period.loc[max_idx, 'Date'].strftime('%Y-%m-%d')})", 
                xy=(df_period.loc[max_idx, 'Date'], df_period.loc[max_idx, 'Price']),
                xytext=(-60, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price (USD/barrel)', fontsize=11)
    ax.set_title('Brent Oil Prices During 2008 Financial Crisis', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '08_2008_crisis_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 08_2008_crisis_analysis.png")

def main():
    print("="*60)
    print("GENERATING EDA VISUALIZATIONS")
    print("="*60)
    
    # Load data
    df = load_data()
    events = load_events()
    
    print(f"\nData loaded: {len(df)} observations")
    print(f"Events loaded: {len(events)} events")
    print(f"\nSaving figures to: {FIGURES_DIR}\n")
    
    # Generate all plots
    plot_price_series(df, events)
    plot_price_distribution(df)
    plot_yearly_average(df)
    plot_rolling_stats(df)
    plot_log_returns(df)
    plot_volatility_by_decade(df)
    plot_extreme_events(df, events)
    plot_event_impact_2008(df, events)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED")
    print("="*60)
    print(f"\nFigures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
