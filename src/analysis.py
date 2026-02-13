"""
Analysis functions for Bayesian change point detection.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_single_changepoint_model(
    prices: np.ndarray,
    model_name: str = "single_changepoint"
) -> pm.Model:
    """
    Build a Bayesian single change point model using PyMC.
    
    The model assumes:
    - Before change point τ: prices ~ Normal(μ₁, σ₁)
    - After change point τ: prices ~ Normal(μ₂, σ₂)
    
    Parameters
    ----------
    prices : np.ndarray
        Array of price values
    model_name : str
        Name for the model
        
    Returns
    -------
    pm.Model
        PyMC model object
    """
    n_obs = len(prices)
    
    with pm.Model(name=model_name) as model:
        # Change point (discrete uniform over all possible days)
        tau = pm.DiscreteUniform('tau', lower=0, upper=n_obs - 1)
        
        # Parameters before change point
        mu_1 = pm.Normal('mu_1', mu=prices.mean(), sigma=prices.std() * 2)
        sigma_1 = pm.HalfNormal('sigma_1', sigma=prices.std())
        
        # Parameters after change point
        mu_2 = pm.Normal('mu_2', mu=prices.mean(), sigma=prices.std() * 2)
        sigma_2 = pm.HalfNormal('sigma_2', sigma=prices.std())
        
        # Create time index
        idx = np.arange(n_obs)
        
        # Switch function to select parameters based on change point
        mu = pm.math.switch(idx < tau, mu_1, mu_2)
        sigma = pm.math.switch(idx < tau, sigma_1, sigma_2)
        
        # Likelihood
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=prices)
    
    return model


def build_changepoint_model_log_returns(
    log_returns: np.ndarray,
    model_name: str = "changepoint_log_returns"
) -> pm.Model:
    """
    Build a change point model for log returns (more stationary).
    
    Parameters
    ----------
    log_returns : np.ndarray
        Array of log return values (NaN values should be removed)
    model_name : str
        Name for the model
        
    Returns
    -------
    pm.Model
        PyMC model object
    """
    # Remove NaN values
    log_returns = log_returns[~np.isnan(log_returns)]
    n_obs = len(log_returns)
    
    with pm.Model(name=model_name) as model:
        # Change point
        tau = pm.DiscreteUniform('tau', lower=10, upper=n_obs - 10)
        
        # Mean log return before and after
        mu_1 = pm.Normal('mu_1', mu=0, sigma=0.1)
        mu_2 = pm.Normal('mu_2', mu=0, sigma=0.1)
        
        # Volatility before and after
        sigma_1 = pm.HalfNormal('sigma_1', sigma=0.05)
        sigma_2 = pm.HalfNormal('sigma_2', sigma=0.05)
        
        # Time index
        idx = np.arange(n_obs)
        
        # Switch
        mu = pm.math.switch(idx < tau, mu_1, mu_2)
        sigma = pm.math.switch(idx < tau, sigma_1, sigma_2)
        
        # Likelihood
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=log_returns)
    
    return model


def sample_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int = 42
) -> az.InferenceData:
    """
    Run MCMC sampling on the model.
    
    Parameters
    ----------
    model : pm.Model
        PyMC model to sample from
    draws : int
        Number of samples to draw
    tune : int
        Number of tuning samples
    chains : int
        Number of MCMC chains
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    az.InferenceData
        ArviZ InferenceData object with samples
    """
    logger.info(f"Starting MCMC sampling: {draws} draws, {tune} tune, {chains} chains")
    
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=True
        )
    
    logger.info("Sampling complete")
    return trace


def check_convergence(trace: az.InferenceData) -> pd.DataFrame:
    """
    Check convergence diagnostics for the MCMC samples.
    
    Parameters
    ----------
    trace : az.InferenceData
        ArviZ InferenceData object
        
    Returns
    -------
    pd.DataFrame
        Summary statistics including r_hat and ESS
    """
    summary = az.summary(trace, var_names=['tau', 'mu_1', 'mu_2', 'sigma_1', 'sigma_2'])
    return summary


def get_changepoint_date(
    trace: az.InferenceData,
    dates: pd.DatetimeIndex
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Get the most probable change point date and credible interval.
    
    Parameters
    ----------
    trace : az.InferenceData
        ArviZ InferenceData object
    dates : pd.DatetimeIndex
        Dates corresponding to the data
        
    Returns
    -------
    Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]
        (most probable date, lower CI, upper CI)
    """
    tau_samples = trace.posterior['tau'].values.flatten()
    
    # Most probable (mode)
    tau_mode = int(np.median(tau_samples))
    
    # 95% credible interval
    tau_lower = int(np.percentile(tau_samples, 2.5))
    tau_upper = int(np.percentile(tau_samples, 97.5))
    
    return dates.iloc[tau_mode], dates.iloc[tau_lower], dates.iloc[tau_upper]


def quantify_impact(
    trace: az.InferenceData,
    metric: str = 'mean'
) -> Dict[str, float]:
    """
    Quantify the impact of the change point on prices.
    
    Parameters
    ----------
    trace : az.InferenceData
        ArviZ InferenceData object
    metric : str
        Either 'mean' or 'median' for central tendency
        
    Returns
    -------
    Dict[str, float]
        Dictionary with impact statistics
    """
    mu_1 = trace.posterior['mu_1'].values.flatten()
    mu_2 = trace.posterior['mu_2'].values.flatten()
    
    if metric == 'mean':
        before = mu_1.mean()
        after = mu_2.mean()
    else:
        before = np.median(mu_1)
        after = np.median(mu_2)
    
    change = after - before
    pct_change = (change / before) * 100 if before != 0 else np.nan
    
    # Probability that mu_2 > mu_1
    prob_increase = (mu_2 > mu_1).mean()
    
    return {
        'before_mean': before,
        'after_mean': after,
        'absolute_change': change,
        'percent_change': pct_change,
        'prob_increase': prob_increase
    }


def plot_posterior_tau(
    trace: az.InferenceData,
    dates: pd.DatetimeIndex,
    events_df: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot the posterior distribution of the change point.
    
    Parameters
    ----------
    trace : az.InferenceData
        ArviZ InferenceData object
    dates : pd.DatetimeIndex
        Dates corresponding to the data
    events_df : pd.DataFrame, optional
        DataFrame with event information to overlay
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    tau_samples = trace.posterior['tau'].values.flatten()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert tau samples to dates for plotting
    tau_dates = [dates.iloc[int(t)] for t in tau_samples if int(t) < len(dates)]
    
    ax.hist(tau_dates, bins=50, density=True, alpha=0.7, color='steelblue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distribution of Change Point')
    
    # Add event markers if provided
    if events_df is not None:
        for _, event in events_df.iterrows():
            if dates.min() <= event['date'] <= dates.max():
                ax.axvline(event['date'], color='red', linestyle='--', alpha=0.5)
                ax.text(event['date'], ax.get_ylim()[1] * 0.9, 
                       event['event_name'], rotation=90, fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_price_with_changepoint(
    prices: np.ndarray,
    dates: pd.DatetimeIndex,
    trace: az.InferenceData,
    events_df: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot price series with detected change point.
    
    Parameters
    ----------
    prices : np.ndarray
        Price data
    dates : pd.DatetimeIndex
        Dates corresponding to prices
    trace : az.InferenceData
        ArviZ InferenceData object
    events_df : pd.DataFrame, optional
        DataFrame with event information
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot prices
    ax.plot(dates, prices, 'b-', alpha=0.7, label='Brent Oil Price')
    
    # Get change point
    tau_mode, tau_lower, tau_upper = get_changepoint_date(trace, dates)
    
    # Shade change point region
    ax.axvline(tau_mode, color='red', linestyle='-', linewidth=2, 
               label=f'Change Point: {tau_mode.strftime("%Y-%m-%d")}')
    ax.axvspan(tau_lower, tau_upper, alpha=0.2, color='red', 
               label='95% Credible Interval')
    
    # Add means
    impact = quantify_impact(trace)
    idx_mode = list(dates).index(tau_mode) if tau_mode in dates.values else len(dates) // 2
    
    ax.hlines(impact['before_mean'], dates.iloc[0], tau_mode, 
              colors='green', linestyles='--', label=f"Before: ${impact['before_mean']:.2f}")
    ax.hlines(impact['after_mean'], tau_mode, dates.iloc[-1], 
              colors='orange', linestyles='--', label=f"After: ${impact['after_mean']:.2f}")
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD/barrel)')
    ax.set_title('Brent Oil Price with Detected Change Point')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    return fig
