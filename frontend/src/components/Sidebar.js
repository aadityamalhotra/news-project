import React, { useState } from 'react';
import './Sidebar.css';

export default function Sidebar({
  clusters,
  selectedCluster,
  onClusterSelect,
  onGenerate,
  loading,
  availableDates,
  uncachedDates,
}) {
  const [showDatePicker, setShowDatePicker] = useState(false);

  const handleGenerateClick = (date = null) => {
    onGenerate(date);
    setShowDatePicker(false);
  };

  const sortedClusters = [...clusters]
    .filter(c => c.cluster_id !== -1)
    .sort((a, b) => b.article_count - a.article_count);

  return (
    <div className="sidebar">
      <div className="sidebar-content">

        {/* ── Load / Date section ──────────────────────────────────────────── */}
        <div className="sidebar-section">
          <button
            className="generate-button"
            onClick={() => handleGenerateClick()}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="button-spinner" />
                Loading...
              </>
            ) : (
              'Load Visualization'
            )}
          </button>

          <button
            className="date-picker-toggle"
            onClick={() => setShowDatePicker(prev => !prev)}
            disabled={loading}
          >
            {showDatePicker ? 'Hide Dates ▲' : 'Select Date ▼'}
          </button>

          {showDatePicker && (
            <div className="date-picker">
              {availableDates.length > 0 && (
                <>
                  <p className="date-picker-label">Ready to load:</p>
                  <div className="date-list">
                    {availableDates.map(date => (
                      <button
                        key={date}
                        className="date-option date-option--ready"
                        onClick={() => handleGenerateClick(date)}
                      >
                        {date}
                      </button>
                    ))}
                  </div>
                </>
              )}

              {uncachedDates.length > 0 && (
                <>
                  <p className="date-picker-label date-picker-label--warn" style={{ marginTop: '10px' }}>
                    In DB (no cache yet):
                  </p>
                  <div className="date-list">
                    {uncachedDates.map(date => (
                      <button
                        key={date}
                        className="date-option date-option--uncached"
                        onClick={() => handleGenerateClick(date)}
                        title="Cache not built — will return an error"
                      >
                        {date} ⚠
                      </button>
                    ))}
                  </div>
                </>
              )}

              {availableDates.length === 0 && uncachedDates.length === 0 && (
                <p className="date-picker-label">No dates available yet.</p>
              )}
            </div>
          )}
        </div>

        {/* ── Clusters section ─────────────────────────────────────────────── */}
        {sortedClusters.length > 0 && (
          <div className="sidebar-section">
            <h2 className="section-title">
              Topics ({sortedClusters.length})
            </h2>

            <div className="clusters-list">
              {sortedClusters.map(cluster => (
                <button
                  key={cluster.cluster_id}
                  className={`cluster-item ${selectedCluster === cluster.cluster_id ? 'selected' : ''}`}
                  onClick={() => onClusterSelect(cluster.cluster_id)}
                >
                  <div
                    className="cluster-color"
                    style={{ backgroundColor: cluster.color }}
                  />
                  <div className="cluster-info">
                    <div className="cluster-name">{cluster.cluster_name}</div>
                    <div className="cluster-count">{cluster.article_count} articles</div>
                  </div>
                  {selectedCluster === cluster.cluster_id && (
                    <span className="cluster-selected-indicator">▶</span>
                  )}
                </button>
              ))}
            </div>

            {selectedCluster !== null && (
              <button
                className="clear-selection"
                onClick={() => onClusterSelect(selectedCluster)}
              >
                Clear Selection
              </button>
            )}
          </div>
        )}

      </div>
    </div>
  );
}