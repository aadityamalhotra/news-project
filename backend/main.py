"""
main.py — FastAPI backend
=========================
Reads pre-processed cluster JSON files from Supabase Storage and serves
them to the React frontend.

Start locally:  uvicorn main:app --reload --port 8000
Deploy:         Render.com (connect GitHub repo, set env vars in dashboard)
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from zoneinfo import ZoneInfo
# CONNECTION MUSH
from supabase import create_client, Client

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ENV VARIABLE MUSH
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,https://news-project-aadityamalhotras-projects.vercel.app,https://news-project-blush.vercel.app").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENV VARIABLE MUSH
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ENV VARIABLE MUSH
SUPABASE_URL = os.getenv("SUPABASE_URL")
# ENV VARIABLE MUSH
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing env vars: SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY")

# CONNECTION MUSH
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SUPABASE_BUCKET = "cluster-cache"


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────
class Article(BaseModel):
    article_id: str
    title: str
    source_name: str
    author: Optional[str]
    url: str
    publish_date: str
    x: float
    y: float
    z: float
    cluster_id: int
    relevance_score: float = 0.0


class ClusterInfo(BaseModel):
    cluster_id: int
    cluster_name: str
    article_count: int
    color: str
    center_x: float
    center_y: float
    center_z: float


class VisualizationResponse(BaseModel):
    articles: List[Article]
    clusters: List[ClusterInfo]
    total_articles: int
    date: str


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _load_cache(date_str: str) -> dict:
    """
    Download the pre-processed cluster JSON for a given date from Supabase Storage.
    Raises 404 HTTPException if the file does not exist yet.
    """
    try:
        # CONNECTION MUSH
        response = supabase.storage.from_(SUPABASE_BUCKET).download(f"{date_str}.json")
        return json.loads(response)
    except Exception as exc:
        print(f"Cache load error for {date_str}: {exc}")
        raise HTTPException(
            status_code=404,
            detail=f"No visualization data available for {date_str}. The clustering pipeline may not have run yet."
        )


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "News Visualization API", "status": "running"}


@app.get("/api/health")
async def health_check():
    """Returns available cached dates from Supabase Storage."""
    try:
        # CONNECTION MUSH
        files = supabase.storage.from_(SUPABASE_BUCKET).list()
        cached_dates = sorted(
            [f["name"].replace(".json", "") for f in files if f["name"].endswith(".json")],
            reverse=True,
        )
    except Exception as exc:
        print(f"Health check storage error: {exc}")
        cached_dates = []

    return {
        "status": "healthy",
        "storage": "supabase",
        "cached_dates": cached_dates[:10],
    }


@app.get("/api/load-visualization")
async def load_visualization(date: Optional[str] = None) -> VisualizationResponse:
    """
    Load pre-processed visualization data from Supabase Storage.

    - If `date` is provided (YYYY-MM-DD), loads that date's cache.
    - If omitted, defaults to yesterday.
    - Returns 404 if no cache exists for the requested date.
    """
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
            target_date_str = date
        except ValueError:
            raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")
    else:
        target_date_str = str((datetime.now(ZoneInfo("America/Chicago")) - timedelta(days=1)).date())

    print(f"Loading cache for {target_date_str}...")
    data = _load_cache(target_date_str)

    articles = [
        Article(
            article_id=a["article_id"],
            title=a["title"],
            source_name=a["source_name"],
            author=a.get("author"),
            url=a["url"],
            publish_date=a["publish_date"],
            x=a["x"],
            y=a["y"],
            z=a["z"],
            cluster_id=a["cluster_id"],
            relevance_score=a.get("relevance_score", 0.0),
        )
        for a in data["articles"]
    ]

    clusters = [
        ClusterInfo(
            cluster_id=c["cluster_id"],
            cluster_name=c["cluster_name"],
            article_count=c["article_count"],
            color=c["color"],
            center_x=c["center_x"],
            center_y=c["center_y"],
            center_z=c["center_z"],
        )
        for c in data["clusters"]
    ]

    print(f"Loaded {len(articles)} articles, {len(clusters)} clusters.")

    return VisualizationResponse(
        articles=articles,
        clusters=clusters,
        total_articles=len(articles),
        date=data["date"],
    )


@app.get("/api/available-dates")
async def get_available_dates():
    """
    Returns dates that have cached cluster files in Supabase Storage,
    plus DB dates that have articles but no cache yet.
    """
    # Dates with cache files in Supabase Storage
    cached = []
    try:
        # CONNECTION MUSH
        files = supabase.storage.from_(SUPABASE_BUCKET).list()
        cached = sorted(
            [f["name"].replace(".json", "") for f in files if f["name"].endswith(".json")],
            reverse=True,
        )[:30]
    except Exception as exc:
        print(f"Warning: could not list Supabase Storage files: {exc}")

    # DB dates that don't have a cache yet — pure SQLAlchemy, no pandas
    uncached = []
    if DATABASE_URL:
        try:
            # CONNECTION MUSH
            engine = create_engine(DATABASE_URL)
            query = text("""
                SELECT DISTINCT DATE(publish_date) AS date
                FROM article_data
                ORDER BY date DESC
                LIMIT 30
            """)
            with engine.connect() as conn:
                result = conn.execute(query)
                db_dates = {str(row[0]) for row in result}

            cached_set = set(cached)
            uncached = sorted(db_dates - cached_set, reverse=True)
        except Exception as exc:
            print(f"Warning: could not query DB for available dates: {exc}")

    return {
        "dates": cached,
        "uncached_dates": uncached,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)