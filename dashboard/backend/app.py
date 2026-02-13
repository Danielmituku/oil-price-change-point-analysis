"""
Flask Backend API for Brent Oil Price Analysis Dashboard.

Provides endpoints for:
- Historical price data
- Change point analysis results
- Event correlations
- Statistics and metrics

Author: Daniel Mituku
Date: February 2026
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
EVENTS_FILE = PROJECT_ROOT / "events" / "major_events.csv"

# Cache for data
_data_cache = {}


def load_price_data():
    """Load and cache price data."""
    if 'prices' not in _data_cache:
        df = pd.read_csv(DATA_DIR / "brent_oil_processed.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        _data_cache['prices'] = df
    return _data_cache['prices']


def load_events():
    """Load and cache events data."""
    if 'events' not in _data_cache:
        events = pd.read_csv(EVENTS_FILE)
        events['date'] = pd.to_datetime(events['date'])
        _data_cache['events'] = events
    return _data_cache['events']


def load_changepoint_results():
    """Load change point analysis results."""
    results_file = RESULTS_DIR / "changepoint_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return []


# ============================================
# API Endpoints
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/prices', methods=['GET'])
def get_prices():
    """
    Get historical price data.
    
    Query params:
    - start_date: YYYY-MM-DD (optional)
    - end_date: YYYY-MM-DD (optional)
    - resample: 'D', 'W', 'M', 'Y' (optional, default 'D')
    """
    df = load_price_data()
    
    # Filter by date range
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if start_date:
        df = df[df['Date'] >= start_date]
    if end_date:
        df = df[df['Date'] <= end_date]
    
    # Resample if requested
    resample = request.args.get('resample', 'D')
    if resample != 'D':
        df = df.set_index('Date')
        df = df.resample(resample).agg({
            'Price': 'mean',
            'log_return': 'sum',
            'rolling_volatility_30': 'mean'
        }).reset_index()
    
    # Convert to JSON-friendly format
    data = df[['Date', 'Price']].copy()
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    
    return jsonify({
        'count': len(data),
        'data': data.to_dict('records')
    })


@app.route('/api/prices/summary', methods=['GET'])
def get_price_summary():
    """Get summary statistics for prices."""
    df = load_price_data()
    
    return jsonify({
        'total_observations': len(df),
        'date_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d')
        },
        'price_stats': {
            'mean': round(df['Price'].mean(), 2),
            'median': round(df['Price'].median(), 2),
            'std': round(df['Price'].std(), 2),
            'min': round(df['Price'].min(), 2),
            'max': round(df['Price'].max(), 2)
        },
        'volatility': {
            'annualized': round(df['log_return'].std() * np.sqrt(252) * 100, 2)
        }
    })


@app.route('/api/prices/yearly', methods=['GET'])
def get_yearly_prices():
    """Get yearly average prices."""
    df = load_price_data()
    
    yearly = df.groupby(df['Date'].dt.year).agg({
        'Price': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    yearly.columns = ['mean', 'std', 'min', 'max']
    yearly = yearly.reset_index()
    yearly.columns = ['year', 'mean', 'std', 'min', 'max']
    
    return jsonify({
        'data': yearly.to_dict('records')
    })


@app.route('/api/events', methods=['GET'])
def get_events():
    """Get all major events."""
    events = load_events()
    
    # Filter by type if specified
    event_type = request.args.get('type')
    if event_type:
        events = events[events['event_type'] == event_type]
    
    events_list = events.copy()
    events_list['date'] = events_list['date'].dt.strftime('%Y-%m-%d')
    
    return jsonify({
        'count': len(events_list),
        'event_types': events['event_type'].unique().tolist(),
        'data': events_list.to_dict('records')
    })


@app.route('/api/events/<int:event_id>/impact', methods=['GET'])
def get_event_impact(event_id):
    """Get price impact around a specific event."""
    events = load_events()
    df = load_price_data()
    
    if event_id < 1 or event_id > len(events):
        return jsonify({'error': 'Event not found'}), 404
    
    event = events[events['event_id'] == event_id].iloc[0]
    event_date = event['date']
    
    # Get data window (90 days before and after)
    window_days = int(request.args.get('window', 90))
    start = event_date - pd.Timedelta(days=window_days)
    end = event_date + pd.Timedelta(days=window_days)
    
    df_window = df[(df['Date'] >= start) & (df['Date'] <= end)]
    
    # Calculate before/after statistics
    before = df_window[df_window['Date'] < event_date]['Price']
    after = df_window[df_window['Date'] >= event_date]['Price']
    
    return jsonify({
        'event': {
            'id': int(event['event_id']),
            'name': event['event_name'],
            'date': event_date.strftime('%Y-%m-%d'),
            'type': event['event_type'],
            'description': event['description']
        },
        'window_days': window_days,
        'price_data': df_window[['Date', 'Price']].assign(
            Date=df_window['Date'].dt.strftime('%Y-%m-%d')
        ).to_dict('records'),
        'impact': {
            'before_mean': round(before.mean(), 2) if len(before) > 0 else None,
            'after_mean': round(after.mean(), 2) if len(after) > 0 else None,
            'change': round(after.mean() - before.mean(), 2) if len(before) > 0 and len(after) > 0 else None,
            'change_pct': round((after.mean() - before.mean()) / before.mean() * 100, 2) if len(before) > 0 and len(after) > 0 else None
        }
    })


@app.route('/api/changepoints', methods=['GET'])
def get_changepoints():
    """Get change point analysis results."""
    results = load_changepoint_results()
    
    return jsonify({
        'count': len(results),
        'data': results
    })


@app.route('/api/volatility', methods=['GET'])
def get_volatility():
    """Get volatility analysis by period."""
    df = load_price_data()
    
    # Calculate by decade
    df['decade'] = (df['Date'].dt.year // 10) * 10
    volatility = df.groupby('decade')['log_return'].std() * np.sqrt(252) * 100
    
    return jsonify({
        'by_decade': volatility.round(2).to_dict()
    })


@app.route('/api/correlation', methods=['GET'])
def get_correlations():
    """Get correlations between events and price movements."""
    events = load_events()
    df = load_price_data()
    
    correlations = []
    
    for _, event in events.iterrows():
        event_date = event['date']
        
        # Get 30-day window after event
        mask = (df['Date'] >= event_date) & (df['Date'] <= event_date + pd.Timedelta(days=30))
        after_data = df[mask]
        
        if len(after_data) > 5:
            # Calculate cumulative return in 30 days after event
            start_price = after_data.iloc[0]['Price']
            end_price = after_data.iloc[-1]['Price']
            cum_return = (end_price - start_price) / start_price * 100
            
            # Calculate volatility spike
            vol = after_data['log_return'].std() * np.sqrt(252) * 100
            
            correlations.append({
                'event_id': int(event['event_id']),
                'event_name': event['event_name'],
                'event_date': event_date.strftime('%Y-%m-%d'),
                'event_type': event['event_type'],
                'expected_impact': event['expected_impact'],
                'actual_30d_return': round(cum_return, 2),
                'volatility_after': round(vol, 2)
            })
    
    return jsonify({
        'data': correlations
    })


# ============================================
# Main
# ============================================

if __name__ == '__main__':
    print("Starting Brent Oil Analysis API...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    
    # Pre-load data
    load_price_data()
    load_events()
    
    app.run(debug=True, port=5000)
