# Data Analysis Workflow

## Change Point Analysis of Brent Oil Prices

### 1. Overview

This document outlines the systematic approach to analyzing how major political and economic events affect Brent oil prices using Bayesian change point detection methods.

---

### 2. Analysis Steps

#### Step 1: Data Acquisition and Preparation
- Load historical Brent oil price data (May 1987 - September 2022)
- Convert date column to datetime format
- Handle missing values and data quality issues
- Create derived features (log returns, rolling statistics)

#### Step 2: Exploratory Data Analysis (EDA)
- Visualize raw price series over time
- Identify trends, seasonality, and volatility patterns
- Test for stationarity (ADF test, KPSS test)
- Analyze log returns distribution
- Identify potential structural breaks visually

#### Step 3: Event Data Integration
- Compile major events dataset (geopolitical, OPEC, economic)
- Align event dates with price data
- Create event markers for visualization

#### Step 4: Bayesian Change Point Modeling
- Define prior distributions for change points (τ)
- Specify likelihood function (Normal distribution)
- Implement PyMC model with switch function
- Run MCMC sampling

#### Step 5: Model Diagnostics
- Check convergence (r_hat, effective sample size)
- Examine trace plots
- Validate posterior distributions

#### Step 6: Results Interpretation
- Identify most probable change point dates
- Quantify parameter changes (before/after)
- Associate change points with documented events
- Calculate impact percentages

#### Step 7: Communication and Reporting
- Create visualizations for stakeholders
- Develop interactive dashboard
- Write comprehensive report with actionable insights

---

### 3. Assumptions and Limitations

#### Assumptions

1. **Data Quality**: The Brent oil price data is accurate and free from significant errors
2. **Single Change Point Model**: Initial model assumes one primary change point (can be extended)
3. **Normal Distribution**: Price changes follow approximately normal distribution
4. **Independence**: Daily price changes are conditionally independent given the regime
5. **Sudden Changes**: Change points represent abrupt rather than gradual transitions

#### Limitations

1. **Correlation vs. Causation**
   - Detecting a statistical change point coinciding with an event does NOT prove causation
   - Multiple factors may contribute to price changes simultaneously
   - Confounding variables may not be captured in the model

2. **Model Simplicity**
   - Simple change point models may miss gradual regime shifts
   - Multiple concurrent change points may be difficult to separate
   - External factors not included in the model may influence results

3. **Data Constraints**
   - Historical data may not fully represent current market dynamics
   - Market structure has evolved significantly over the analysis period
   - High-frequency effects may be masked in daily data

4. **Temporal Precision**
   - Exact timing of change points has inherent uncertainty
   - Market reactions may precede or lag official event dates
   - Information may be priced in before public announcements

---

### 4. Statistical Methodology

#### Change Point Detection Framework

The Bayesian change point model detects structural breaks in time series data by:

1. **Prior Specification**: Define probability distribution over possible change point locations
2. **Likelihood Definition**: Model data generation process before and after change point
3. **Posterior Computation**: Use MCMC to estimate posterior distribution of change points

#### Mathematical Formulation

For a time series y_t with a single change point τ:

```
y_t ~ Normal(μ₁, σ₁)  if t < τ
y_t ~ Normal(μ₂, σ₂)  if t ≥ τ
```

Where:
- μ₁, μ₂: Mean values before and after change point
- σ₁, σ₂: Standard deviations before and after
- τ: Change point location (discrete uniform prior)

---

### 5. Communication Channels

#### Primary Stakeholders
1. **Investors**: Risk assessment and portfolio management
2. **Policymakers**: Energy security and economic planning
3. **Energy Companies**: Operational planning and cost management

#### Communication Formats
- **Technical Report**: Detailed methodology and statistical results
- **Executive Summary**: Key findings and actionable recommendations
- **Interactive Dashboard**: Self-service exploration of data and insights
- **Blog Post**: Public-facing summary for broader audience

#### Key Metrics to Communicate
- Detected change point dates with confidence intervals
- Percentage price changes associated with each event
- Probability of regime change given observed data
- Comparative impact of different event types

---

### 6. Quality Assurance

- Code review and version control (Git)
- Reproducible analysis (Jupyter notebooks with seeds)
- Model validation through sensitivity analysis
- Peer review of findings and interpretations

---

### 7. Timeline

| Phase | Activities | Branch |
|-------|-----------|--------|
| Foundation | Data prep, EDA, event research | task-1-foundation |
| Modeling | PyMC implementation, change point detection | task-2-modeling |
| Dashboard | Flask API, React frontend | task-3-dashboard |
| Reporting | Final report, documentation | main |

---

*Document Version: 1.0*
*Last Updated: February 2026*
