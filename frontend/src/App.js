import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import './App.css';
import NewsVisualization from './NewsVisualization';
import Sidebar from './Sidebar';
import AboutMe from './AboutMe';
import axios from 'axios';

// ENV VARIABLE MUSH
// Set REACT_APP_API_URL in your Vercel environment variables to your Render backend URL
// e.g. https://your-api.onrender.com
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ─── Main visualization page ─────────────────────────────────────────────────
function MainPage() {
  const navigate = useNavigate();
  const [articles, setArticles] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [visualizationDate, setVisualizationDate] = useState(null);
  const [availableDates, setAvailableDates] = useState([]);
  const [uncachedDates, setUncachedDates] = useState([]);

  useEffect(() => {
    fetchAvailableDates();
  }, []);

  const fetchAvailableDates = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/available-dates`);
      setAvailableDates(response.data.dates || []);
      setUncachedDates(response.data.uncached_dates || []);
    } catch (err) {
      console.error('Error fetching dates:', err);
    }
  };

  const loadVisualization = async (date = null) => {
    setLoading(true);
    setError(null);
    setSelectedCluster(null);

    try {
      const url = date
        ? `${API_BASE_URL}/api/load-visualization?date=${date}`
        : `${API_BASE_URL}/api/load-visualization`;

      const response = await axios.get(url);

      setArticles(response.data.articles);
      setClusters(response.data.clusters);
      setVisualizationDate(response.data.date);
      setLoading(false);
    } catch (err) {
      const detail = err.response?.data?.detail || 'Failed to load visualization';
      setError(detail);
      setLoading(false);
    }
  };

  const handleClusterSelect = (clusterId) => {
    setSelectedCluster(prev => prev === clusterId ? null : clusterId);
  };

  const selectedClusterArticles = selectedCluster !== null
    ? [...articles.filter(a => a.cluster_id === selectedCluster)]
        .sort((a, b) => (b.relevance_score ?? 0) - (a.relevance_score ?? 0))
    : [];

  const selectedClusterInfo = clusters.find(c => c.cluster_id === selectedCluster);

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-left">
          <h1>Global News Visualization</h1>
          {visualizationDate && (
            <p className="date-display">Showing news from {visualizationDate}</p>
          )}
        </div>
        <div className="header-right">
          <button className="about-btn" onClick={() => navigate('/about')}>
            About Me
          </button>
        </div>
      </header>

      <div className="main-container">
        <Sidebar
          clusters={clusters}
          selectedCluster={selectedCluster}
          onClusterSelect={handleClusterSelect}
          onGenerate={loadVisualization}
          loading={loading}
          availableDates={availableDates}
          uncachedDates={uncachedDates}
        />

        <div className="right-panel">
          <div className="visualization-container">
            {loading && (
              <div className="loading-overlay">
                <div className="loading-spinner"></div>
                <p>Loading visualization...</p>
                <p className="loading-subtext">Reading pre-processed cluster data</p>
              </div>
            )}

            {error && (
              <div className="error-overlay">
                <div className="error-box">
                  <h2>Could Not Load Data</h2>
                  <p>{error}</p>
                  <button onClick={() => loadVisualization()}>Try Again</button>
                </div>
              </div>
            )}

            {!loading && !error && articles.length === 0 && (
              <div className="empty-state">
                <h2>Welcome to News Visualization</h2>
                <p>Select a date and click "Load Visualization" to explore the news in 3D</p>
                <button
                  className="cta-button"
                  onClick={() => loadVisualization()}
                >
                  Load Latest
                </button>
              </div>
            )}

            {!loading && articles.length > 0 && (
              <NewsVisualization
                articles={articles}
                clusters={clusters}
                selectedCluster={selectedCluster}
              />
            )}
          </div>

          {selectedCluster !== null && selectedClusterArticles.length > 0 && (
            <div className="article-panel">
              <div className="article-panel-header">
                <span
                  className="article-panel-dot"
                  style={{ backgroundColor: selectedClusterInfo?.color || '#B17457' }}
                />
                <h2 className="article-panel-title">
                  {selectedClusterInfo?.cluster_name}
                </h2>
                <span className="article-panel-count">
                  {selectedClusterArticles.length} articles
                </span>
              </div>

              <div className="article-panel-list">
                {selectedClusterArticles.map((article, idx) => (
                  <a
                    key={article.article_id}
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="article-panel-item"
                  >
                    <span className="article-panel-rank">{idx + 1}</span>
                    <div className="article-panel-body">
                      <div className="article-panel-item-title">{article.title}</div>
                      <div className="article-panel-meta">
                        <span className="article-panel-source">{article.source_name}</span>
                        {article.author && (
                          <span className="article-panel-author"> · {article.author}</span>
                        )}
                      </div>
                    </div>
                    <span className="article-panel-arrow">↗</span>
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/about" element={<AboutMe />} />
      </Routes>
    </Router>
  );
}