# News Intel — Global News Visualization

**A fully automated pipeline that ingests, clusters, and renders ~1,000 news articles daily as an interactive 3D galaxy.**

Every night at 12:10 AM, articles are pulled from 20 international sources, embedded into semantic space, clustered by topic using ML, labeled by an LLM, and served to a live 3D visualization — with zero manual steps.

🔗 **[Live Demo](https://your-vercel-url.vercel.app)** &nbsp;|&nbsp; 🔗 **[API](https://your-render-url.onrender.com)**

---

## How It Works

The pipeline runs as three chained GitHub Actions workflows:

```
12:10 AM CT   ingest.yml       NewsAPI + Newspaper3k scraper → Supabase PostgreSQL
                                    ↓ on success
              clean.yml        SQL deduplication + malformed row removal
                                    ↓ on success
 ~2:00 AM CT  cluster.yml      ML pipeline → JSON uploaded to Supabase Storage
```

**Stage 1 — Ingestion** pulls articles from 20 sources (BBC, AP, Al Jazeera, Fox, The Verge, Hacker News, and more) via NewsAPI, scrapes full article text with Newspaper3k using a custom browser config to bypass consent walls, and stores everything in PostgreSQL with MD5-based deduplication.

**Stage 2 — Cleaning** removes malformed rows (titles containing raw GMT timestamps) and exact title duplicates via two targeted SQL DELETE statements. Pure SQLAlchemy, no overhead.

**Stage 3 — ML Clustering** runs the full pipeline:
- `all-mpnet-base-v2` generates 768-dimensional sentence embeddings per article
- UMAP reduces 768D → 3D using cosine distance, preserving semantic neighborhood structure
- DBSCAN clusters articles with `eps=0.22`, then a hierarchical splitter recursively re-embeds and sub-clusters any group exceeding 100 articles
- Groq API (Llama 3.1 8B) generates a 3–5 word topic label per cluster
- Cosine similarity scores each article's relevance to its cluster label
- Result uploaded as `YYYY-MM-DD.json` to Supabase Storage

**User experience** — FastAPI on Render downloads the JSON from Supabase Storage and serves it to the React + Three.js frontend on Vercel. No ML runs at request time. Load time is under 2 seconds.

---

## Why These Technical Choices

**DBSCAN over k-means** — k-means requires specifying the number of clusters upfront and forces every article into one. News has genuine outliers — isolated stories with no peers. DBSCAN discovers cluster count naturally and marks outliers as noise, which is semantically honest.

**UMAP over t-SNE** — UMAP preserves global structure, meaning thematically related clusters stay near each other in the 3D projection. t-SNE optimizes only local structure and produces visually nice but globally meaningless layouts. Since spatial position carries semantic information here, global structure matters.

**Cosine distance for embeddings** — Sentence embeddings encode meaning as direction, not magnitude. Cosine distance measures the angle between vectors, not their length — making it the correct metric for semantic similarity regardless of embedding scale.

**Pre-computed cache** — The ML pipeline takes 20–40 minutes. Serving it on-demand per request is impossible. Running nightly and caching as a single JSON file reduces the API to a file download, cutting load times from 40 minutes to under 2 seconds.

**Two `requirements.txt` files** — The backend needs only FastAPI, Supabase, and SQLAlchemy (~50 MB). The cron scripts need PyTorch, UMAP, scikit-learn, and sentence-transformers (~2 GB). Keeping them separate means Render deploys fast and never hits memory limits loading ML libraries it doesn't use.

---

## Stack

| | |
|---|---|
| **Orchestration** | GitHub Actions (chained workflows) |
| **Ingestion** | NewsAPI + Newspaper3k |
| **Database** | Supabase PostgreSQL |
| **Embeddings** | `all-mpnet-base-v2` (768-dim) |
| **Dimensionality Reduction** | UMAP (cosine metric) |
| **Clustering** | DBSCAN + hierarchical splitting |
| **LLM Labeling** | Groq API — Llama 3.1 8B |
| **Cache Storage** | Supabase Storage |
| **Backend** | FastAPI + Uvicorn on Render |
| **Frontend** | React + Three.js + React Three Fiber on Vercel |

---

## Project Structure

```
├── frontend/                   # React app → Vercel
│   └── src/
│       ├── App.js
│       ├── NewsVisualization.js
│       ├── Sidebar.js
│       └── AboutMe.js
├── backend/                    # FastAPI → Render
│   ├── main.py
│   └── requirements.txt
├── cron/                       # Pipeline scripts → GitHub Actions
│   ├── daily_reader.py
│   ├── data_cleaner.py
│   ├── news_clustering.py
│   └── requirements.txt
└── .github/workflows/
    ├── ingest.yml
    ├── clean.yml
    └── cluster.yml
```

---

## Environment Variables

**GitHub Actions Secrets** (Settings → Secrets → Actions)

| Secret | Description |
|---|---|
| `NEWSROOM_API_KEY` | NewsAPI.org key |
| `DATABASE_URL` | Supabase PostgreSQL connection string |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `GROQ_API_KEY` | Groq API key — free at console.groq.com |

**Render** — same DB and Supabase vars, plus `ALLOWED_ORIGINS` set to your Vercel URL

**Vercel** — `REACT_APP_API_URL` set to your Render URL

---

## Supabase Setup

```sql
CREATE TABLE article_data (
    article_id    TEXT PRIMARY KEY,
    source_id     TEXT,
    source_name   TEXT,
    author        TEXT,
    title         TEXT NOT NULL,
    url           TEXT,
    full_content  TEXT,
    publish_date  DATE
);
```

Create a Storage bucket named exactly `cluster-cache`, set to Private.

---

## Running Locally

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend && npm install
REACT_APP_API_URL=http://localhost:8000 npm start

# Pipeline (manual run)
cd cron && pip install -r requirements.txt
python daily_reader.py      # ~1-3 hours
python data_cleaner.py      # ~5 seconds
python news_clustering.py   # ~20-40 minutes
```

---

*Built by Aaditya Malhotra — CS + Data Science, University of Wisconsin–Madison*
