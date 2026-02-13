"""
Bayesian Change Point Analysis of Brent Oil Prices using PyMC.

This script:
1. Loads and prepares the Brent oil price data
2. Runs Bayesian change point detection on multiple time periods
3. Generates visualizations of detected change points
4. Quantifies the impact of detected changes
5. Associates change points with known major events

Author: Daniel Mituku
Date: February 2026
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import json

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
az.style.use('arviz-darkgrid')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EVENTS_FILE = PROJECT_ROOT / "events" / "major_events.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def load_data():
    """Load processed Brent oil price data."""
    df = pd.read_csv(DATA_DIR / "brent_oil_processed.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def load_events():
    """Load major events data."""
    events = pd.read_csv(EVENTS_FILE)
    events['date'] = pd.to_datetime(events['date'])
    return events


def prepare_analysis_period(df, start_date, end_date):
    """
    Prepare data for a specific analysis period.
    
    Returns prices, time index, and dates.
    """
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df_period = df[mask].copy().reset_index(drop=True)
    
    prices = df_period['Price'].values
    time_idx = np.arange(len(prices))
    dates = df_period['Date']
    
    return prices, time_idx, dates, df_period


def build_single_changepoint_model(prices, name="changepoint"):
    """
    Build a Bayesian single change point model.
    
    Model:
    - tau ~ DiscreteUniform(0, n-1)  # Change point location
    - mu_1, mu_2 ~ Normal(mean(prices), 2*std(prices))  # Means before/after
    - sigma_1, sigma_2 ~ HalfNormal(std(prices))  # Standard deviations
    - y_t ~ Normal(mu_1, sigma_1) if t < tau else Normal(mu_2, sigma_2)
    """
    n_obs = len(prices)
    
    with pm.Model() as model:
        # Change point prior - uniform over all days
        tau = pm.DiscreteUniform('tau', lower=10, upper=n_obs - 10)
        
        # Parameters before change point
        mu_1 = pm.Normal('mu_1', mu=np.mean(prices), sigma=np.std(prices) * 2)
        sigma_1 = pm.HalfNormal('sigma_1', sigma=np.std(prices))
        
        # Parameters after change point
        mu_2 = pm.Normal('mu_2', mu=np.mean(prices), sigma=np.std(prices) * 2)
        sigma_2 = pm.HalfNormal('sigma_2', sigma=np.std(prices))
        
        # Time index
        idx = np.arange(n_obs)
        
        # Switch function
        mu = pm.math.switch(idx < tau, mu_1, mu_2)
        sigma = pm.math.switch(idx < tau, sigma_1, sigma_2)
        
        # Likelihood
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=prices)
    
    return model


def run_mcmc(model, draws=2000, tune=1000, chains=2, random_seed=42):
    """Run MCMC sampling."""
    print("  Running MCMC sampling...")
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True,
            cores=1  # Use 1 core for stability
        )
    return trace


def get_changepoint_results(trace, dates):
    """Extract change point results from trace."""
    tau_samples = trace.posterior['tau'].values.flatten()
    mu_1_samples = trace.posterior['mu_1'].values.flatten()
    mu_2_samples = trace.posterior['mu_2'].values.flatten()
    
    # Most probable change point (median)
    tau_median = int(np.median(tau_samples))
    tau_lower = int(np.percentile(tau_samples, 2.5))
    tau_upper = int(np.percentile(tau_samples, 97.5))
    
    # Convert to dates
    cp_date = dates.iloc[tau_median]
    cp_date_lower = dates.iloc[tau_lower]
    cp_date_upper = dates.iloc[tau_upper]
    
    # Impact statistics
    before_mean = float(np.mean(mu_1_samples))
    after_mean = float(np.mean(mu_2_samples))
    change = after_mean - before_mean
    pct_change = (change / before_mean) * 100 if before_mean != 0 else 0
    
    # Probability of increase
    prob_increase = float((mu_2_samples > mu_1_samples).mean())
    
    return {
        'tau_index': tau_median,
        'tau_lower': tau_lower,
        'tau_upper': tau_upper,
        'change_point_date': cp_date.strftime('%Y-%m-%d'),
        'ci_lower': cp_date_lower.strftime('%Y-%m-%d'),
        'ci_upper': cp_date_upper.strftime('%Y-%m-%d'),
        'mean_before': round(before_mean, 2),
        'mean_after': round(after_mean, 2),
        'absolute_change': round(change, 2),
        'percent_change': round(pct_change, 2),
        'probability_increase': round(prob_increase, 4)
    }


def find_nearest_event(cp_date, events_df, max_days=60):
    """Find the nearest event to a detected change point."""
    cp_date = pd.to_datetime(cp_date)
    nearest = None
    min_days = float('inf')
    
    for _, event in events_df.iterrows():
        days_diff = abs((event['date'] - cp_date).days)
        if days_diff < min_days and days_diff <= max_days:
            min_days = days_diff
            nearest = {
                'event_name': event['event_name'],
                'event_date': event['date'].strftime('%Y-%m-%d'),
                'event_type': event['event_type'],
                'days_from_cp': int((event['date'] - cp_date).days),
                'description': event['description']
            }
    
    return nearest


def plot_changepoint_analysis(prices, dates, trace, results, events_df, period_name, save_path):
    """Generate comprehensive change point visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Price series with change point
    ax1 = axes[0, 0]
    ax1.plot(dates, prices, 'b-', linewidth=1, alpha=0.7, label='Daily Price')
    
    # Change point line and confidence interval
    cp_date = pd.to_datetime(results['change_point_date'])
    ci_lower = pd.to_datetime(results['ci_lower'])
    ci_upper = pd.to_datetime(results['ci_upper'])
    
    ax1.axvline(cp_date, color='red', linestyle='-', linewidth=2, 
                label=f'Change Point: {results["change_point_date"]}')
    ax1.axvspan(ci_lower, ci_upper, alpha=0.2, color='red', label='95% CI')
    
    # Before/after means
    tau_idx = results['tau_index']
    ax1.hlines(results['mean_before'], dates.iloc[0], cp_date, 
               colors='green', linestyles='--', linewidth=2,
               label=f'Before: ${results["mean_before"]:.2f}')
    ax1.hlines(results['mean_after'], cp_date, dates.iloc[-1], 
               colors='orange', linestyles='--', linewidth=2,
               label=f'After: ${results["mean_after"]:.2f}')
    
    # Mark events
    for _, event in events_df.iterrows():
        if dates.iloc[0] <= event['date'] <= dates.iloc[-1]:
            ax1.axvline(event['date'], color='purple', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD/barrel)')
    ax1.set_title(f'{period_name}: Price with Detected Change Point')
    ax1.legend(loc='upper left', fontsize=9)
    
    # 2. Posterior distribution of tau
    ax2 = axes[0, 1]
    tau_samples = trace.posterior['tau'].values.flatten()
    tau_dates = [dates.iloc[int(t)] for t in tau_samples if int(t) < len(dates)]
    ax2.hist(tau_dates, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(cp_date, color='red', linestyle='-', linewidth=2, label='Median')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Density')
    ax2.set_title('Posterior Distribution of Change Point')
    ax2.legend()
    
    # 3. Before vs After means
    ax3 = axes[1, 0]
    mu_1 = trace.posterior['mu_1'].values.flatten()
    mu_2 = trace.posterior['mu_2'].values.flatten()
    
    ax3.hist(mu_1, bins=50, density=True, alpha=0.6, color='green', label='Before (μ₁)')
    ax3.hist(mu_2, bins=50, density=True, alpha=0.6, color='orange', label='After (μ₂)')
    ax3.axvline(results['mean_before'], color='green', linestyle='--', linewidth=2)
    ax3.axvline(results['mean_after'], color='orange', linestyle='--', linewidth=2)
    ax3.set_xlabel('Price (USD/barrel)')
    ax3.set_ylabel('Density')
    ax3.set_title('Posterior Distributions of Mean Prices')
    ax3.legend()
    
    # 4. Price difference distribution
    ax4 = axes[1, 1]
    diff = mu_2 - mu_1
    ax4.hist(diff, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax4.axvline(np.mean(diff), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean: ${np.mean(diff):.2f}')
    
    # Add probability annotation
    prob_text = f'P(increase): {results["probability_increase"]:.1%}'
    ax4.text(0.95, 0.95, prob_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax4.set_xlabel('Price Change (After - Before)')
    ax4.set_ylabel('Density')
    ax4.set_title('Posterior Distribution of Price Change')
    ax4.legend()
    
    plt.suptitle(f'Bayesian Change Point Analysis: {period_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def analyze_period(df, events_df, period_name, start_date, end_date):
    """Run complete analysis for a specific time period."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {period_name}")
    print(f"Period: {start_date} to {end_date}")
    print('='*60)
    
    # Prepare data
    prices, time_idx, dates, df_period = prepare_analysis_period(df, start_date, end_date)
    print(f"  Observations: {len(prices)}")
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Build and run model
    model = build_single_changepoint_model(prices, name=period_name.replace(' ', '_'))
    trace = run_mcmc(model, draws=2000, tune=1000, chains=2)
    
    # Get results
    results = get_changepoint_results(trace, dates)
    results['period_name'] = period_name
    results['start_date'] = start_date
    results['end_date'] = end_date
    
    # Find nearest event
    nearest_event = find_nearest_event(results['change_point_date'], events_df)
    if nearest_event:
        results['associated_event'] = nearest_event
        print(f"\n  RESULTS:")
        print(f"  Change Point Detected: {results['change_point_date']}")
        print(f"  95% CI: [{results['ci_lower']}, {results['ci_upper']}]")
        print(f"  Mean Before: ${results['mean_before']:.2f}")
        print(f"  Mean After: ${results['mean_after']:.2f}")
        print(f"  Change: ${results['absolute_change']:.2f} ({results['percent_change']:.1f}%)")
        print(f"  P(increase): {results['probability_increase']:.1%}")
        print(f"\n  Associated Event: {nearest_event['event_name']}")
        print(f"  Event Date: {nearest_event['event_date']} ({nearest_event['days_from_cp']} days from CP)")
    else:
        results['associated_event'] = None
    
    # Generate visualization
    fig_path = FIGURES_DIR / f"changepoint_{period_name.replace(' ', '_').lower()}.png"
    plot_changepoint_analysis(prices, dates, trace, results, events_df, period_name, fig_path)
    
    # Check convergence
    summary = az.summary(trace, var_names=['tau', 'mu_1', 'mu_2'])
    results['convergence'] = {
        'r_hat_tau': float(summary.loc['tau', 'r_hat']),
        'r_hat_mu1': float(summary.loc['mu_1', 'r_hat']),
        'r_hat_mu2': float(summary.loc['mu_2', 'r_hat'])
    }
    
    return results


def main():
    """Main function to run all analyses."""
    print("="*70)
    print("BAYESIAN CHANGE POINT ANALYSIS OF BRENT OIL PRICES")
    print("="*70)
    
    # Load data
    df = load_data()
    events_df = load_events()
    print(f"\nData loaded: {len(df)} observations")
    print(f"Events loaded: {len(events_df)} events")
    
    # Define analysis periods
    periods = [
        {
            'name': '2008 Financial Crisis',
            'start': '2007-01-01',
            'end': '2009-12-31'
        },
        {
            'name': 'COVID-19 Pandemic',
            'start': '2019-01-01',
            'end': '2021-06-30'
        },
        {
            'name': '2014 OPEC Price War',
            'start': '2013-06-01',
            'end': '2016-06-30'
        },
        {
            'name': 'Russia Ukraine War',
            'start': '2021-06-01',
            'end': '2022-11-01'
        }
    ]
    
    # Run analyses
    all_results = []
    for period in periods:
        try:
            results = analyze_period(
                df, events_df,
                period['name'],
                period['start'],
                period['end']
            )
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue
    
    # Save all results
    results_file = RESULTS_DIR / "changepoint_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    # Generate summary report
    print("\n" + "="*70)
    print("SUMMARY OF DETECTED CHANGE POINTS")
    print("="*70)
    
    for r in all_results:
        print(f"\n{r['period_name']}:")
        print(f"  Change Point: {r['change_point_date']}")
        print(f"  Impact: ${r['mean_before']:.2f} → ${r['mean_after']:.2f} ({r['percent_change']:+.1f}%)")
        if r.get('associated_event'):
            print(f"  Event: {r['associated_event']['event_name']} ({r['associated_event']['event_date']})")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()
