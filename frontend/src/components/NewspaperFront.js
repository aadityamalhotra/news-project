import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './NewspaperFront.css';

console.log('[NewspaperFront] module loaded');

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function formatNewspaperDate(dateStr) {
  if (!dateStr) return '';
  const [year, month, day] = dateStr.split('-').map(Number);
  const date = new Date(year, month - 1, day);
  const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];
  const ordinal = (n) => {
    const s = ['th', 'st', 'nd', 'rd'];
    const v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  };
  return `${dayNames[date.getDay()]}, the ${ordinal(day)} of ${monthNames[month - 1]}, ${year}`;
}

function Divider({ thick = false, ornate = false }) {
  if (ornate) {
    return (
      <div className="divider divider--ornate">
        <span className="divider-line" />
        <span className="divider-diamond">&#9670;</span>
        <span className="divider-line" />
      </div>
    );
  }
  return <div className={`divider ${thick ? 'divider--thick' : ''}`} />;
}

function ArticleCard({ article, variant = 'medium' }) {
  const words = article.summary ? article.summary.split(' ') : [];
  const WORD_LIMIT = variant === 'lead' ? 160 : variant === 'medium' ? 100 : 70;
  const needsTruncation = words.length > WORD_LIMIT;
  const displayText = needsTruncation
    ? words.slice(0, WORD_LIMIT).join(' ')
    : article.summary;

  return (
    <article className={`article-card article-card--${variant}`}>
      <div className="article-meta-top">
        <span className="article-source">{article.source_name}</span>
        {article.author && (
          <span className="article-author"> &middot; {article.author}</span>
        )}
      </div>

      <h2 className={`article-headline article-headline--${variant}`}>
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          className="article-headline-link"
        >
          {article.title}
        </a>
      </h2>

      <div className="article-topic-tag">{article.cluster_name}</div>

      <Divider />

      <p className="article-body">
        {displayText}
        {needsTruncation && (
          <>
            {'... '}
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="article-read-more"
            >
              [Read full article &#8599;]
            </a>
          </>
        )}
      </p>
    </article>
  );
}

export default function NewspaperFront() {
  console.log('[NewspaperFront] component rendering');

  const [highlights, setHighlights] = useState([]);
  const [date, setDate] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHighlights = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/highlights`);
        setHighlights(response.data.highlights || []);
        setDate(response.data.date || '');
        setLoading(false);
      } catch (err) {
        const detail = err.response?.data?.detail || 'Failed to load highlights.';
        setError(detail);
        setLoading(false);
      }
    };
    fetchHighlights();
  }, []);

  const formattedDate = formatNewspaperDate(date);

  const lead = highlights[0];
  const mediums = highlights.slice(1, 3);
  const smalls = highlights.slice(3, 5);

  return (
    <div className="newspaper">

      <header className="masthead">
        <div className="masthead-top-rule" />
        <div className="masthead-eyebrow">
          <span className="masthead-eyebrow-left">Est. by Algorithm &amp; Ink</span>
          <span className="masthead-eyebrow-right">Intelligence, Distilled Daily</span>
        </div>
        <div className="masthead-title-row">
          <h1 className="masthead-title">Tomorrow&#39;s Archive</h1>
        </div>
        <div className="masthead-subtitle-row">
          <div className="masthead-rule-left" />
          <span className="masthead-date">{formattedDate || '\u00A0'}</span>
          <div className="masthead-rule-right" />
        </div>
        <div className="masthead-nav-row">
          <a href="/viz" className="masthead-nav-btn">
            View 3D Visualization &#8594;
          </a>
          <a href="/about" className="masthead-nav-btn masthead-nav-btn--ghost">
            About
          </a>
        </div>
        <div className="masthead-bottom-rule" />
      </header>

      <main className="newspaper-body">

        {loading && (
          <div className="newspaper-loading">
            <p className="newspaper-loading-text">Typesetting the morning edition&hellip;</p>
          </div>
        )}

        {error && (
          <div className="newspaper-error">
            <p className="newspaper-error-text">{error}</p>
            <p className="newspaper-error-sub">
              The pipeline may not have run yet.{' '}
              <a href="/viz" className="newspaper-error-link">
                Go to visualization &#8594;
              </a>
            </p>
          </div>
        )}

        {!loading && !error && highlights.length > 0 && (
          <>
            {lead && (
              <>
                <section className="layout-lead">
                  <ArticleCard article={lead} variant="lead" />
                </section>
                <Divider ornate />
              </>
            )}

            {mediums.length > 0 && (
              <>
                <section className="layout-medium-pair">
                  {mediums.map((article, i) => (
                    <React.Fragment key={article.article_id}>
                      <ArticleCard article={article} variant="medium" />
                      {i === 0 && mediums.length > 1 && (
                        <div className="column-rule" />
                      )}
                    </React.Fragment>
                  ))}
                </section>
                <Divider ornate />
              </>
            )}

            {smalls.length > 0 && (
              <section className="layout-small-trio">
                {smalls.map((article, i) => (
                  <React.Fragment key={article.article_id}>
                    <ArticleCard article={article} variant="small" />
                    {i < smalls.length - 1 && (
                      <div className="column-rule" />
                    )}
                  </React.Fragment>
                ))}
              </section>
            )}
          </>
        )}
      </main>

      <footer className="newspaper-footer">
        <Divider thick />
        <p className="newspaper-footer-text">
          Tomorrow&#39;s Archive &mdash; News clustered by machine, presented in ink.
          All articles sourced from international wire services and publications.
        </p>
      </footer>
    </div>
  );
}