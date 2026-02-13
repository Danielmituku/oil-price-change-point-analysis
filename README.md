# Change Point Analysis and Statistical Modeling of Time Series Data

## Detecting Changes and Associating Causes on Brent Oil Price Time Series Data

### Project Overview

This project analyzes how significant political and economic events affect Brent oil prices using Bayesian change point detection methods. The analysis focuses on identifying structural breaks in oil prices and associating them with major geopolitical events, OPEC decisions, and economic shocks.

**Organization:** Birhan Energies - Data-driven insights for the energy sector

### Business Objectives

1. **Identify Key Events**: Detect significant events that have impacted Brent oil prices over the past decade
2. **Quantify Impact**: Use statistical methods to measure how much these events affect price changes
3. **Provide Insights**: Deliver data-driven insights to guide investment strategies, policy development, and operational planning

### Dataset

- **Source**: Historical Brent oil prices
- **Period**: May 20, 1987 - September 30, 2022
- **Features**:
  - `Date`: Daily date in 'day-month-year' format
  - `Price`: Brent oil price in USD per barrel

### Project Structure

```
week11/
├── data/
│   ├── raw/                    # Raw Brent oil price data
│   └── processed/              # Processed and cleaned data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_change_point_analysis.ipynb  # Bayesian modeling
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading utilities
│   └── analysis.py            # Analysis functions
├── reports/
│   ├── interim_report.md      # Interim submission report
│   └── final_report.md        # Final submission report
├── events/
│   └── major_events.csv       # Key events dataset
├── dashboard/
│   ├── backend/               # Flask API backend
│   └── frontend/              # React frontend
├── docs/
│   └── analysis_workflow.md   # Analysis workflow documentation
└── scripts/
    └── download_data.py       # Data download scripts
```

### Branch Structure

| Branch | Purpose | Tasks |
|--------|---------|-------|
| `main` | Production-ready code | Final merged code |
| `task-1-foundation` | Foundation & EDA | Data workflow, event research, EDA |
| `task-2-modeling` | Change Point Analysis | Bayesian modeling with PyMC |
| `task-3-dashboard` | Dashboard Development | Flask backend + React frontend |

### Key Learning Outcomes

**Skills:**
- Change Point Analysis & Interpretation
- Statistical Reasoning
- PyMC for Bayesian modeling
- Analytical Storytelling with Data

**Knowledge:**
- Probability distributions
- Bayesian inference
- Monte Carlo Markov Chain (MCMC)
- Model comparison
- Policy analysis

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd week11

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. **Data Preparation**
   ```bash
   python scripts/download_data.py
   ```

2. **Run EDA Notebook**
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

3. **Run Change Point Analysis**
   ```bash
   jupyter notebook notebooks/02_change_point_analysis.ipynb
   ```

4. **Start Dashboard**
   ```bash
   # Backend
   cd dashboard/backend
   flask run

   # Frontend (new terminal)
   cd dashboard/frontend
   npm start
   ```

### Deliverables

#### Interim Submission
- GitHub Link (Task 1 complete)
- Interim Report (PDF):
  - Analysis workflow document
  - Major events dataset
  - Initial EDA findings

#### Final Submission
- Complete GitHub repository
- Final Report (blog post format):
  - Complete analysis methodology
  - Change point visualizations
  - Quantified impact statements
  - Dashboard screenshots
  - Limitations and future work

### References

- [Bayesian Changepoint Detection with PyMC3](https://docs.pymc.io/)
- [Change Point Detection in Time Series](https://forecastegy.com/posts/change-point-detection-time-series-python/)
- [MCMC Explained](https://towardsdatascience.com/monte-carlo-markov-chain-mcmc-explained-94e3a6c8de11)

### Author

Daniel Mituku - 10 Academy Week 11 Challenge

### License

This project is for educational purposes as part of 10 Academy's training program.
