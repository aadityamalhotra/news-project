<div align="center">
<br />
```
███╗   ██╗███████╗██╗    ██╗███████╗    ██╗███╗   ██╗████████╗███████╗██╗     
████╗  ██║██╔════╝██║    ██║██╔════╝    ██║████╗  ██║╚══██╔══╝██╔════╝██║     
██╔██╗ ██║█████╗  ██║ █╗ ██║███████╗    ██║██╔██╗ ██║   ██║   █████╗  ██║     
██║╚██╗██║██╔══╝  ██║███╗██║╚════██║    ██║██║╚██╗██║   ██║   ██╔══╝  ██║     
██║ ╚████║███████╗╚███╔███╔╝███████║    ██║██║ ╚████║   ██║   ███████╗███████╗
╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚══════╝    ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚══════╝
```
Every article. Every source. Every morning. Mapped in three dimensions.
<br />
Show Image
Show Image
Show Image
Show Image
Show Image
Show Image
<br />
</div>

What Is This?
News Intel is a fully automated intelligence platform that ingests, processes, and visualizes the global news cycle every single day — without a single manual step.
Each night, ~1,000 articles are pulled from 20 international news sources, cleaned, embedded into high-dimensional semantic space, clustered by topic using machine learning, labeled by an LLM, and rendered as an interactive 3D galaxy — where proximity equals meaning. Articles that are semantically similar float near each other. Clusters of related stories glow together as bubbles of light. You can orbit the entire day's news with your mouse.
The system runs itself. You just watch.
<br />

The Pipeline
                        ┌─────────────────────────────────────────────────────┐
                        │              EVERY NIGHT AT 12:10 AM                │
                        └─────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — INGESTION                                              ingest.yml  │
│                                                                               │
│  NewsAPI (20 sources) ──► Newspaper3k scraper ──► Supabase PostgreSQL        │
│                                                                               │
│  • ABC News AU  • Al Jazeera  • Associated Press  • BBC News                 │
│  • Breitbart  • Business Insider  • CBC  • CBS  • Financial Post              │
│  • Fortune  • Fox News  • Hacker News  • NBC  • RTE  • TechRadar             │
│  • Times of India  • The Verge  • USA Today  • BuzzFeed  • Fox Sports        │
│                                                                               │
│  Custom browser config bypasses consent walls and rate limits.               │
│  MD5 hashing for deduplication. ~1,000 articles per run.                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — CLEANING                                                clean.yml  │
│                                                                               │
│  SQL Rule 1: DELETE WHERE title LIKE '%GMT%'  (malformed timestamp rows)     │
│  SQL Rule 2: Deduplicate by title — keep MIN(article_id), drop the rest      │
│                                                                               │
│  Runs as pure SQLAlchemy. No pandas. No overhead.                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — ML CLUSTERING                                         cluster.yml  │
│                                                                               │
│  all-mpnet-base-v2          768-dim sentence embeddings                      │
│       │                                                                       │
│       ▼                                                                       │
│  UMAP (cosine metric)       768D ──► 3D  (n_neighbors=15, spread=2.5)        │
│       │                                                                       │
│       ▼                                                                       │
│  DBSCAN (eps=0.22)          Initial clustering pass                          │
│       │                                                                       │
│       ▼                                                                       │
│  Hierarchical Splitter      Recursively splits clusters > 100 articles       │
│       │                     Re-embeds with 1200 chars, tighter eps=0.15      │
│       ▼                                                                       │
│  Visual Exaggeration        Pull members inward, push centroids apart        │
│       │                                                                       │
│       ▼                                                                       │
│  Groq API (Llama 3.1)       3-5 word topic label per cluster                 │
│       │                                                                       │
│       ▼                                                                       │
│  Cosine Similarity          Relevance score: each article vs. cluster label  │
│       │                                                                       │
│       ▼                                                                       │
│  Supabase Storage           Uploads YYYY-MM-DD.json (~2-5 MB)               │
└──────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  USER EXPERIENCE                                                              │
│                                                                               │
│  Vercel (React + Three.js)  ◄──►  Render (FastAPI)  ◄──►  Supabase Storage  │
│                                                                               │
│  JSON downloaded once per session. No ML at request time. Instant load.     │
└──────────────────────────────────────────────────────────────────────────────┘
<br />

Technical Deep Dive
Embedding & Dimensionality Reduction
Each article is embedded as a 768-dimensional vector using all-mpnet-base-v2 — a sentence transformer trained on over 1 billion sentence pairs, chosen for its superior semantic understanding over lighter alternatives. The full article title plus the first 800 characters of body text are embedded together to capture both headline signal and topical context.
UMAP then projects these 768-dimensional vectors into 3D space while preserving local neighborhood structure. The cosine metric is used (not Euclidean) because direction matters more than magnitude for semantic similarity — two articles about the same topic should cluster together even if their embedding vectors differ in length.
pythonreducer = umap.UMAP(
    n_components=3,
    n_neighbors=15,      # balance between local and global structure
    min_dist=0.15,       # breathing room between points within clusters  
    spread=2.5,          # distributes clusters across the full 3D volume
    metric="cosine",
    random_state=42,
)
Hierarchical DBSCAN Clustering
Standard DBSCAN is run first with eps=0.22 on the 3D UMAP projections. This intentionally tight epsilon prevents unrelated articles from being swept into the same cluster. Any cluster exceeding 100 articles is then recursively re-embedded (using 1,200 characters this time, for richer differentiation) and sub-clustered with an even tighter eps=0.15 — until all clusters fall within the 8–150 article range.
This two-pass approach solves a fundamental problem: a single DBSCAN pass either misses fine-grained distinctions (loose epsilon) or over-fragments coherent topics (tight epsilon). The hierarchical approach applies looseness globally and tightness only where needed.
LLM Cluster Labeling
Each cluster's titles are sampled (up to 20 headlines) and sent to the Groq API running llama-3.1-8b-instant. The prompt is engineered to prevent the most common failure modes: pipe-separated alternatives, hedged multi-topic outputs, and sentence-length descriptions. Stop tokens include |, /, and or to cut off any multi-topic generation at the API level.
Relevance Scoring
Every article receives a relevance score between 0 and 1 — the cosine similarity between its title embedding and its cluster's label embedding. When a user clicks a cluster, articles are ranked by this score descending. The result: the most representative, on-topic articles surface first, and outliers that ended up in a cluster due to proximity rather than true topical alignment naturally sink to the bottom.
<br />

Architecture
your-repo/
├── frontend/                          # React app → Vercel
│   ├── src/
│   │   ├── App.js                     # Routing, state, data fetching
│   │   ├── NewsVisualization.js       # Three.js 3D scene, points, bubbles
│   │   ├── Sidebar.js                 # Cluster list, date picker
│   │   └── AboutMe.js                 # Portfolio page
│   └── public/
│
├── backend/                           # FastAPI → Render
│   ├── main.py                        # API endpoints, Supabase Storage reads
│   └── requirements.txt              # Lean — no pandas, no ML libs
│
├── cron/                              # GitHub Actions scripts
│   ├── daily_reader.py               # Ingestion — NewsAPI + scraping
│   ├── data_cleaner.py               # SQL cleaning rules
│   ├── news_clustering.py            # Full ML pipeline + Supabase upload
│   └── requirements.txt             # Heavy — sentence-transformers, umap, etc.
│
└── .github/workflows/
    ├── ingest.yml                     # 12:10 AM CT → triggers clean.yml
    ├── clean.yml                      # Cleaning → triggers cluster.yml
    └── cluster.yml                    # 2:00 AM CT safety net + chained trigger
Why two requirements.txt files? The backend only needs FastAPI, Supabase, and SQLAlchemy — a ~50MB install. The cron scripts need PyTorch (via sentence-transformers), UMAP, scikit-learn, and more — a ~2GB install. Keeping them separate means Render deploys fast and doesn't run out of memory loading ML libraries it never uses.
<br />

Stack
LayerTechnologyPurposeOrchestrationGitHub ActionsScheduled cron + chained workflow triggersIngestionNewsAPI + Newspaper3kArticle metadata + full-text scrapingDatabaseSupabase PostgreSQLPersistent article storage with deduplicationEmbeddingsall-mpnet-base-v2768-dim semantic vector representationsDim. ReductionUMAP768D → 3D with cosine metricClusteringDBSCAN + hierarchical splittingSemantic grouping with size constraintsLLM LabelingGroq API (Llama 3.1 8B)Topic name generation per clusterStorageSupabase StoragePre-processed JSON cache (2-5 MB/day)BackendFastAPI + UvicornREST API serving cache to frontendFrontendReact + Three.js + R3FInteractive 3D visualizationDeploymentVercel + RenderFrontend + backend hosting
<br />

Environment Variables
GitHub Actions Secrets
(Settings → Secrets and variables → Actions)
SecretDescriptionNEWSROOM_API_KEYNewsAPI.org API keyDATABASE_URLSupabase PostgreSQL connection stringSUPABASE_URLSupabase project URLSUPABASE_SERVICE_ROLE_KEYSupabase service role key (not the anon key)GROQ_API_KEYconsole.groq.com — free tier
Render (Backend)
VariableValueDATABASE_URLSupabase connection stringSUPABASE_URLSupabase project URLSUPABASE_SERVICE_ROLE_KEYSupabase service role keyALLOWED_ORIGINSYour Vercel URL e.g. https://your-app.vercel.app
Vercel (Frontend)
VariableValueREACT_APP_API_URLYour Render URL e.g. https://your-api.onrender.com
<br />

Supabase Setup
1. Create the articles table:
sqlCREATE TABLE article_data (
    article_id    TEXT PRIMARY KEY,
    source_id     TEXT,
    source_name   TEXT,
    author        TEXT,
    title         TEXT NOT NULL,
    url           TEXT,
    full_content  TEXT,
    publish_date  DATE
);
2. Create the storage bucket:
Go to Storage → New Bucket → name it exactly cluster-cache → set to Private.
<br />

Running Locally
bash# Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Backend
cd backend
pip install -r requirements.txt
# create a .env file with DATABASE_URL, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, ALLOWED_ORIGINS
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
REACT_APP_API_URL=http://localhost:8000 npm start

# Run the pipeline manually (new terminal)
cd cron
pip install -r requirements.txt
# create a .env file with all secrets
python daily_reader.py    # ~1-3 hours
python data_cleaner.py    # ~5 seconds
python news_clustering.py # ~20-40 minutes
<br />

How The Automation Works
The three GitHub Actions workflows chain together automatically:
12:10 AM CT   ingest.yml starts
              └─ daily_reader.py runs (1-3 hrs)
              └─ on success → dispatches clean.yml

~2:00 AM CT   clean.yml starts  
              └─ data_cleaner.py runs (~5 sec)
              └─ on success → dispatches cluster.yml

              cluster.yml starts
              └─ news_clustering.py runs (~30 min)
              └─ uploads YYYY-MM-DD.json to Supabase Storage
              └─ frontend reflects new data on next user visit

2:00 AM CT    cluster.yml also has its own scheduled trigger
              (safety net in case the dispatch chain breaks)
No servers to manage. No Airflow instance to maintain. No manual steps. The entire data pipeline runs on GitHub's free compute tier.
<br />

Design Decisions Worth Noting
Why pre-compute everything? The ML pipeline takes 20-40 minutes to run. Running it on-demand per user request is obviously impossible. By running it nightly and caching the result as a single JSON file, the user-facing API becomes trivially simple — it just downloads a file and returns it. Load times drop from 40 minutes to under 2 seconds.
Why DBSCAN over k-means? K-means requires specifying the number of clusters upfront and forces every point into a cluster. News doesn't work like that — some articles are genuinely isolated stories with no clear peers. DBSCAN discovers the number of clusters naturally and explicitly marks outliers as noise, which is semantically honest.
Why UMAP over t-SNE? UMAP preserves global structure — clusters that are thematically related in high-dimensional space remain near each other in the 3D projection. t-SNE is optimized purely for local structure and produces visually nice plots that are globally meaningless. Since the spatial positions in this visualization carry semantic information (proximity = similarity), global structure matters.
Why cosine distance for UMAP? Sentence embeddings encode meaning as direction, not magnitude. Two articles about the same topic should cluster together regardless of whether their raw embedding vectors are large or small. Cosine distance measures the angle between vectors, not their length — making it the correct metric for semantic similarity.
<br />

Known Limitations

Ollama/local LLM not available in CI — Groq API is used for cluster labeling in production. Running locally, you can swap back to Ollama by pointing OLLAMA_URL at your local instance and adjusting the labeling function.
GitHub Actions cron drift — GitHub does not guarantee exact cron timing under heavy load. Runs may start up to 15-20 minutes late during peak usage periods.
Render free tier cold starts — The backend spins down after 15 minutes of inactivity. An UptimeRobot monitor pinging / every 5 minutes prevents this in production.
NewsAPI free tier — Limited to 100 articles per source per request. Developer plan supports up to 100 requests/day total across all sources.

<br />

<div align="center">
Built by Aaditya Malhotra
CS + Data Science, University of Wisconsin–Madison
Show Image
Show Image
</div>
