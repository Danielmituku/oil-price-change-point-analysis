import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ReferenceLine, Area, AreaChart
} from 'recharts';
import './index.css';

const API_URL = 'http://localhost:5000/api';

function App() {
  const [priceData, setPriceData] = useState([]);
  const [summary, setSummary] = useState(null);
  const [events, setEvents] = useState([]);
  const [changepoints, setChangepoints] = useState([]);
  const [yearlyData, setYearlyData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPeriod, setSelectedPeriod] = useState('all');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Fetch all data in parallel
      const [pricesRes, summaryRes, eventsRes, cpRes, yearlyRes] = await Promise.all([
        fetch(`${API_URL}/prices?resample=W`),
        fetch(`${API_URL}/prices/summary`),
        fetch(`${API_URL}/events`),
        fetch(`${API_URL}/changepoints`),
        fetch(`${API_URL}/prices/yearly`)
      ]);

      const prices = await pricesRes.json();
      const summaryData = await summaryRes.json();
      const eventsData = await eventsRes.json();
      const cpData = await cpRes.json();
      const yearly = await yearlyRes.json();

      setPriceData(prices.data || []);
      setSummary(summaryData);
      setEvents(eventsData.data || []);
      setChangepoints(cpData.data || []);
      setYearlyData(yearly.data || []);
      
    } catch (error) {
      console.error('Error fetching data:', error);
      // Use sample data for demo
      setSummary({
        total_observations: 9011,
        date_range: { start: '1987-05-20', end: '2022-11-14' },
        price_stats: { mean: 48.42, median: 38.57, std: 32.86, min: 9.10, max: 143.95 },
        volatility: { annualized: 40.53 }
      });
    } finally {
      setLoading(false);
    }
  };

  const getEventTypeClass = (type) => {
    switch (type) {
      case 'Conflict': return 'conflict';
      case 'Economic Crisis': return 'economic';
      case 'OPEC Policy': return 'opec';
      case 'Sanctions': return 'sanctions';
      default: return '';
    }
  };

  if (loading) {
    return (
      <div className="dashboard">
        <header className="header">
          <h1>Brent Oil Price Analysis Dashboard</h1>
          <p>Bayesian Change Point Detection & Event Correlation Analysis</p>
        </header>
        <div className="loading">Loading data...</div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="header">
        <h1>Brent Oil Price Analysis Dashboard</h1>
        <p>Bayesian Change Point Detection & Event Correlation Analysis | 1987-2022</p>
      </header>

      <main className="main-content">
        {/* Summary Stats */}
        {summary && (
          <div className="stats-grid">
            <div className="stat-card">
              <div className="label">Total Observations</div>
              <div className="value">{summary.total_observations?.toLocaleString()}</div>
            </div>
            <div className="stat-card">
              <div className="label">Average Price</div>
              <div className="value">${summary.price_stats?.mean}</div>
            </div>
            <div className="stat-card">
              <div className="label">Price Range</div>
              <div className="value">${summary.price_stats?.min} - ${summary.price_stats?.max}</div>
            </div>
            <div className="stat-card">
              <div className="label">Volatility (Annual)</div>
              <div className="value">{summary.volatility?.annualized}%</div>
            </div>
          </div>
        )}

        {/* Price Chart */}
        <div className="charts-section">
          <div className="chart-container">
            <h2>Historical Brent Oil Prices (Weekly)</h2>
            <div className="filters">
              <select 
                className="filter-select"
                value={selectedPeriod}
                onChange={(e) => setSelectedPeriod(e.target.value)}
              >
                <option value="all">All Time</option>
                <option value="2008">2008 Crisis</option>
                <option value="2014">2014 OPEC War</option>
                <option value="2020">COVID-19</option>
                <option value="2022">2022 Ukraine</option>
              </select>
            </div>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={priceData}>
                <defs>
                  <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2874a6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#2874a6" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis 
                  dataKey="Date" 
                  tick={{ fontSize: 11 }}
                  tickFormatter={(date) => date?.slice(0, 4)}
                />
                <YAxis 
                  tick={{ fontSize: 11 }}
                  tickFormatter={(value) => `$${value}`}
                />
                <Tooltip 
                  formatter={(value) => [`$${value?.toFixed(2)}`, 'Price']}
                  labelFormatter={(label) => `Date: ${label}`}
                />
                <Area 
                  type="monotone" 
                  dataKey="Price" 
                  stroke="#2874a6" 
                  fillOpacity={1}
                  fill="url(#colorPrice)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Yearly Average Chart */}
          <div className="chart-container">
            <h2>Yearly Average Prices</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={yearlyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis 
                  dataKey="year" 
                  tick={{ fontSize: 11 }}
                />
                <YAxis 
                  tick={{ fontSize: 11 }}
                  tickFormatter={(value) => `$${value}`}
                />
                <Tooltip 
                  formatter={(value) => [`$${value?.toFixed(2)}`, 'Avg Price']}
                />
                <Bar dataKey="mean" fill="#2874a6" radius={[4, 4, 0, 0]} />
                <ReferenceLine y={summary?.price_stats?.mean} stroke="#e74c3c" strokeDasharray="5 5" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Change Points Section */}
        <div className="changepoints-section">
          <h2>Detected Change Points (Bayesian Analysis)</h2>
          {changepoints.length > 0 ? (
            changepoints.map((cp, index) => (
              <div key={index} className="changepoint-card">
                <h3>{cp.period_name}</h3>
                <div className="changepoint-details">
                  <div className="detail-item">
                    <div className="label">Change Point Date</div>
                    <div className="value">{cp.change_point_date}</div>
                  </div>
                  <div className="detail-item">
                    <div className="label">95% CI</div>
                    <div className="value">{cp.ci_lower} to {cp.ci_upper}</div>
                  </div>
                  <div className="detail-item">
                    <div className="label">Price Before</div>
                    <div className="value">${cp.mean_before}</div>
                  </div>
                  <div className="detail-item">
                    <div className="label">Price After</div>
                    <div className="value">${cp.mean_after}</div>
                  </div>
                  <div className="detail-item">
                    <div className="label">Change</div>
                    <div className={`value ${cp.percent_change >= 0 ? 'positive' : 'negative'}`}>
                      {cp.percent_change >= 0 ? '+' : ''}{cp.percent_change}%
                    </div>
                  </div>
                  <div className="detail-item">
                    <div className="label">Associated Event</div>
                    <div className="value">{cp.associated_event?.event_name || 'N/A'}</div>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <p>Run the change point analysis to see results here.</p>
          )}
        </div>

        {/* Events Section */}
        <div className="events-section">
          <h2>Major Events ({events.length})</h2>
          <div className="events-grid">
            {events.slice(0, 12).map((event, index) => (
              <div key={index} className={`event-card ${getEventTypeClass(event.event_type)}`}>
                <h3>{event.event_name}</h3>
                <div className="event-date">{event.date}</div>
                <span className="event-type">{event.event_type}</span>
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{ 
        textAlign: 'center', 
        padding: '20px', 
        color: '#666',
        borderTop: '1px solid #eee',
        marginTop: '20px'
      }}>
        <p>Brent Oil Price Change Point Analysis | 10 Academy Week 11 Challenge</p>
        <p style={{ fontSize: '0.85rem' }}>Author: Daniel Mituku | February 2026</p>
      </footer>
    </div>
  );
}

export default App;
