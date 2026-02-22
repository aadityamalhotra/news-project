import React from 'react';
import { useNavigate } from 'react-router-dom';
import './AboutMe.css';

export default function AboutMe() {
  const navigate = useNavigate();

  return (
    <div className="about-page">
      <header className="about-header">
        <button className="back-btn" onClick={() => navigate('/')}>
          ← Back to Visualization
        </button>
        <h1 className="about-site-title">Global News Visualization</h1>
      </header>

      <main className="about-main">
        <section className="about-hero">
          <h2 className="about-name">Aaditya Malhotra</h2>
          <p className="about-tagline">Data Engineer · Pipeline Architect · System Builder</p>
        </section>

        <div className="about-grid">

          <div className="about-card about-card--wide">
            <h3 className="about-card-title">About Me</h3>
            <div className="about-card-body">
              <p>
                I'm passionate about building intelligent data systems that transform raw information into actionable insights.
                My focus is on designing and automating end-to-end data pipelines that are robust, scalable, and production-ready.
                I thrive at the intersection of data engineering and machine learning, where I can architect systems that not only
                move and process data efficiently, but also extract meaningful patterns and deliver real-time intelligence.
              </p>
              <p style={{ marginTop: '12px' }}>
                This project showcases my approach to building complete data products—from ingestion and processing to
                visualization and deployment. I believe the best way to demonstrate technical skills is through
                end-to-end implementations that solve real problems.
              </p>
            </div>
          </div>

          <div className="about-card">
            <h3 className="about-card-title">What I Built</h3>
            <div className="about-card-body">
              <p>A fully automated news intelligence platform that processes thousands of global articles daily. The system:</p>
              <ul style={{ marginTop: '10px', marginLeft: '18px' }}>
                <li><strong>Ingests</strong> news from 20+ international sources via API with intelligent rate-limiting and anti-blocking techniques</li>
                <li><strong>Scrapes</strong> full article content using custom browser configurations to bypass consent walls</li>
                <li><strong>Stores</strong> structured data in PostgreSQL with deduplication and conflict handling</li>
                <li><strong>Clusters</strong> articles by semantic similarity using sentence transformers, UMAP dimensionality reduction, and DBSCAN</li>
                <li><strong>Labels</strong> each cluster automatically using Ollama (Llama 3.2) with strict prompts for single-topic identification</li>
                <li><strong>Visualizes</strong> in interactive 3D using React and Three.js, with dynamic camera controls and cluster exploration</li>
                <li><strong>Orchestrates</strong> the entire pipeline through GitHub Actions with automatic daily scheduling and chained execution</li>
              </ul>
              <p style={{ marginTop: '10px' }}>
                The frontend loads pre-processed cache files instantly — no real-time clustering or LLM calls during user interaction.
              </p>
            </div>
          </div>

          <div className="about-card">
            <h3 className="about-card-title">What This Taught Me</h3>
            <div className="about-card-body">
              <p>This project pushed me to connect multiple technical domains into a cohesive system:</p>
              <ul style={{ marginTop: '10px', marginLeft: '18px' }}>
                <li><strong>Pipeline orchestration:</strong> Designing reliable CI/CD workflows with proper task dependencies and error recovery</li>
                <li><strong>Data engineering at scale:</strong> Handling thousands of articles with efficient batch processing and database optimization</li>
                <li><strong>ML/NLP integration:</strong> Fine-tuning embedding models and clustering algorithms for real-world noisy data</li>
                <li><strong>LLM prompt engineering:</strong> Crafting strict prompts that produce consistent, structured outputs from generative models</li>
                <li><strong>Full-stack development:</strong> Building responsive frontends that visualize complex data while maintaining performance</li>
                <li><strong>Production thinking:</strong> Pre-processing heavy computations, implementing caching strategies, and optimizing for user experience</li>
                <li><strong>Cloud architecture:</strong> Connecting Supabase, Render, and Vercel into a cohesive serverless stack with GitHub Actions as the orchestration layer</li>
              </ul>
            </div>
          </div>

          <div className="about-card about-card--full">
            <h3 className="about-card-title">Tech Stack</h3>
            <div className="about-card-body stack-grid">
              {[
                { label: 'Orchestration', value: 'GitHub Actions (scheduled workflows)' },
                { label: 'Database', value: 'Supabase PostgreSQL' },
                { label: 'File Storage', value: 'Supabase Storage' },
                { label: 'Data Ingestion', value: 'NewsAPI + Newspaper3k' },
                // Fixed: was still showing old model name all-MiniLM-L6-v2
                { label: 'Embeddings', value: 'Sentence Transformers (all-mpnet-base-v2)' },
                { label: 'Dimensionality Reduction', value: 'UMAP' },
                { label: 'Clustering', value: 'DBSCAN + Hierarchical Splitting (scikit-learn)' },
                { label: 'LLM Labelling', value: 'Ollama (Llama 3.2)' },
                { label: 'Backend API', value: 'FastAPI + Uvicorn (Render.com)' },
                { label: 'Frontend', value: 'React + Three.js + React Three Fiber (Vercel)' },
                { label: 'Languages', value: 'Python, JavaScript, SQL' },
              ].map(item => (
                <div key={item.label} className="stack-item">
                  <span className="stack-label">{item.label}</span>
                  <span className="stack-value">{item.value}</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}