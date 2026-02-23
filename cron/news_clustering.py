"""
news_clustering.py
==================
Reads yesterday's articles from Supabase DB, clusters them semantically,
labels each cluster with Groq (free LLM API), and uploads the result JSON
to Supabase Storage.

Run manually:  python news_clustering.py
Run via CI:    GitHub Actions workflow (.github/workflows/cluster.yml)
               (triggered automatically after ingest.yml succeeds)

PIPELINE:
  1. Read yesterday's articles from the database
  2. Generate sentence embeddings with all-mpnet-base-v2 (768-dim)
  3. Reduce to 3D with UMAP for visualization
  4. Initial clustering with tight DBSCAN (eps=0.28)
  5. Hierarchical splitting: recursively split oversized clusters (>150-200 articles)
  6. Apply visual exaggeration for cleaner 3D separation
  7. Label each cluster with Groq / llama-3.1-8b-instant (single-topic, strict prompt)
  8. Score each article's semantic relevance to its cluster name
  9. Upload result to Supabase Storage as <YYYY-MM-DD>.json

GROQ SETUP:
  - Sign up free at console.groq.com
  - Create an API key
  - Add it as GROQ_API_KEY in GitHub Actions secrets (and Render env vars if needed)
  - Free tier: 14,400 requests/day, more than enough for daily clustering
"""

import colorsys
import json
import os
import random
import re
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import pendulum
import requests
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import DBSCAN
from supabase import create_client

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
LOCAL_TZ = "America/Chicago"

# Groq API — free LLM inference, replaces local Ollama
# Sign up at console.groq.com, create an API key, add as GROQ_API_KEY secret
# ENV VARIABLE MUSH
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"   # free, fast, high quality
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT = 30

# ── EMBEDDING MODEL ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBED_CONTENT_LENGTH = 800

# ── UMAP settings ─────────────────────────────────────────────────────────────
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = 15      # higher = better global structure
UMAP_MIN_DIST = 0.15       # breathing room between points within a cluster
UMAP_SPREAD = 2.5          # wide spread — purely visual, does NOT affect clustering
UMAP_RANDOM_STATE = 42

# ── DBSCAN settings ───────────────────────────────────────────────────────────
# IMPORTANT: DBSCAN now runs on the raw 768-dim embeddings (cosine metric),
# NOT on the 3D UMAP projections. This decouples visual spread from clustering.
# Cosine distance ranges 0–1, so eps is a semantic similarity threshold:
#   eps=0.25 means "articles within 25% cosine distance are neighbours"
DBSCAN_INITIAL_EPS = 0.25          # cosine distance threshold on raw embeddings
DBSCAN_INITIAL_MIN_SAMPLES = 8     # min articles to form a core cluster point

# ── Hierarchical splitting ─────────────────────────────────────────────────────
MAX_CLUSTER_SIZE_SOFT = 100        # split sooner before clusters get mixed
MAX_CLUSTER_SIZE_HARD = 150        # absolute hard cap
MIN_CLUSTER_SIZE_FINAL = 8         # minimum cluster size after all splits
SUBCLUSTER_EPS = 0.18              # tighter eps for subclustering large groups
SUBCLUSTER_MIN_SAMPLES = 5
SUBCLUSTER_CONTENT_LENGTH = 1200

# ── Visual exaggeration ────────────────────────────────────────────────────────
PULL_FACTOR = 0.35       # was 0.06 — pull cluster members tighter inward
PUSH_FACTOR = 0.45       # was 0.10 — push noise and cluster centers apart more

# ── Outlier compression ────────────────────────────────────────────────────────
OUTLIER_COMPRESS_THRESHOLD_MULTIPLIER = 2.0
OUTLIER_LOG_SCALE = 0.5

# ── Supabase Storage bucket name ───────────────────────────────────────────────
# Create a bucket named "cluster-cache" in your Supabase Storage dashboard.
SUPABASE_BUCKET = "cluster-cache"

# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────────
BASE_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
    "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B739", "#52B788",
    "#FF85A2", "#5DADE2", "#F39C12", "#8E44AD", "#16A085",
    "#E74C3C", "#3498DB", "#2ECC71", "#E67E22", "#9B59B6",
    "#FF5733", "#C70039", "#900C3F", "#581845", "#1ABC9C",
    "#27AE60", "#2980B9", "#8E44AD", "#F39C12", "#D35400",
    "#C0392B", "#7F8C8D", "#34495E", "#16A085", "#F1C40F",
]


def _cluster_colors(n: int) -> list:
    colors = BASE_COLORS.copy()
    while len(colors) < n:
        idx = len(colors) - len(BASE_COLORS)
        hue = (idx * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors[:n]


# ──────────────────────────────────────────────────────────────────────────────
# GROQ LABELLING
# ──────────────────────────────────────────────────────────────────────────────
def _label_cluster_groq(titles: list) -> str:
    """
    Ask Groq (free LLM API) for a 2-4 word topic label for a cluster.
    Falls back to keyword extraction if Groq is unavailable or key is missing.

    Groq uses the OpenAI-compatible chat completions API format.
    Model: llama-3.1-8b-instant (free, ~500 tokens/sec)
    """
    if not GROQ_API_KEY:
        print("  ⚠ GROQ_API_KEY not set — using keyword fallback")
        return _label_cluster_fallback(titles)

    sample = random.sample(titles, min(20, len(titles)))
    titles_text = "\n".join(f"- {t}" for t in sample)

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                # CONNECTION MUSH
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a senior news editor. Your only job is to read "
                            "a list of headlines and output ONE single topic label of 3 to 5 words. "
                            "Rules: reply with the topic words only, no punctuation, no explanation, "
                            "no sentences, no pipe characters, no slashes, no alternatives. "
                            "Output exactly one label. "
                            "Good examples: Gaza Ceasefire Talks, Federal Reserve Rate Decision, "
                            "Silicon Valley Tech Layoffs, Ukraine War Frontlines, "
                            "US Immigration Border Policy"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Headlines:\n{titles_text}\n\nTOPIC (2-4 words only):",
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 20,
                "stop": ["\n", ",", "|", "/", " or ", " and "],  # prevent multi-topic output
            },
            timeout=GROQ_TIMEOUT,
        )

        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip any preamble the model snuck in
            for prefix in [
                "Topic:", "Topic name:", "The topic is", "Based on",
                "ONE TOPIC:", "Main topic:", "Answer:", "Label:",
            ]:
                if raw.lower().startswith(prefix.lower()):
                    raw = raw[len(prefix):].strip().lstrip(":- ")

            raw = raw.split("\n")[0].strip().strip("\"'.,;:")
            raw = re.sub(r"^\d+\.\s*", "", raw)
            raw = re.sub(r"^[-•*]\s*", "", raw)
            raw = " ".join(raw.split()[:5])

            if len(raw) >= 3:
                print(f"  ✓ Groq label: '{raw}'")
                return raw
        else:
            print(f"  ✗ Groq HTTP {resp.status_code}: {resp.text[:200]}")

    except Exception as exc:
        print(f"  ✗ Groq error: {exc} — falling back to keyword extraction")

    return _label_cluster_fallback(titles)


def _label_cluster_fallback(titles: list, n: int = 3) -> str:
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "has", "have", "had", "been", "will", "would", "could", "should",
        "says", "said", "after", "new", "over", "more", "how", "what", "when",
        "its", "their", "his", "her", "our", "we", "he", "she", "they", "it",
        "that", "this", "which", "who", "report", "reports",
    }
    words = []
    for t in titles:
        words += [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", t.lower()) if w not in stop]
    if not words:
        return "Miscellaneous"

    counts = Counter(words)
    top = [w for w, _ in counts.most_common(n)]
    return " ".join(top[:3]).title()


def _relevance_score(title: str, label: str, model) -> float:
    try:
        title_emb = model.encode([title], show_progress_bar=False)[0]
        label_emb = model.encode([label], show_progress_bar=False)[0]

        dot_product = np.dot(title_emb, label_emb)
        norm_title = np.linalg.norm(title_emb)
        norm_label = np.linalg.norm(label_emb)

        if norm_title == 0 or norm_label == 0:
            return 0.0

        cosine_sim = dot_product / (norm_title * norm_label)
        return float(max(0.0, min(1.0, cosine_sim)))

    except Exception as e:
        print(f"  Warning: relevance score failed for '{title}' vs '{label}': {e}")
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────
def _split_oversized_clusters(df, embeddings, model, iteration=1):
    print(f"\n{'  ' * iteration}🔍 Iteration {iteration}: Checking for oversized clusters...")

    cluster_sizes = df[df['cluster_id'] != -1].groupby('cluster_id').size()
    oversized_hard = cluster_sizes[cluster_sizes > MAX_CLUSTER_SIZE_HARD].index.tolist()
    oversized_soft = cluster_sizes[
        (cluster_sizes > MAX_CLUSTER_SIZE_SOFT) & (cluster_sizes <= MAX_CLUSTER_SIZE_HARD)
    ].index.tolist()

    if not oversized_hard and not oversized_soft:
        print(f"{'  ' * iteration}✓ All clusters are appropriately sized!")
        return df

    print(f"{'  ' * iteration}Found {len(oversized_hard)} clusters MUST split (>{MAX_CLUSTER_SIZE_HARD})")
    print(f"{'  ' * iteration}Found {len(oversized_soft)} clusters should split (>{MAX_CLUSTER_SIZE_SOFT})")

    max_cluster_id = df['cluster_id'].max()
    next_cluster_id = max_cluster_id + 1

    for cluster_id in oversized_hard:
        cluster_mask = df['cluster_id'] == cluster_id
        cluster_size = cluster_mask.sum()
        print(f"{'  ' * iteration}  Splitting cluster {cluster_id} ({cluster_size} articles) - HARD LIMIT")

        cluster_df = df[cluster_mask].copy()
        rich_text = (
            cluster_df['title'] + " " +
            cluster_df['full_content'].fillna("").str.slice(0, SUBCLUSTER_CONTENT_LENGTH)
        )
        print(f"{'  ' * iteration}    Re-embedding {len(cluster_df)} articles...")
        sub_embeddings = model.encode(rich_text.tolist(), show_progress_bar=False)

        sub_clusterer = DBSCAN(eps=SUBCLUSTER_EPS, min_samples=SUBCLUSTER_MIN_SAMPLES)
        sub_labels = sub_clusterer.fit_predict(sub_embeddings)

        n_subclusters = len([lbl for lbl in np.unique(sub_labels) if lbl != -1])
        n_noise = (sub_labels == -1).sum()
        print(f"{'  ' * iteration}    → Created {n_subclusters} subclusters, {n_noise} noise points")

        for orig_idx, sub_label in zip(cluster_df.index, sub_labels):
            if sub_label == -1:
                df.loc[orig_idx, 'cluster_id'] = -1
            else:
                df.loc[orig_idx, 'cluster_id'] = next_cluster_id + sub_label

        next_cluster_id += n_subclusters

    for cluster_id in oversized_soft:
        cluster_mask = df['cluster_id'] == cluster_id
        cluster_size = cluster_mask.sum()
        print(f"{'  ' * iteration}  Attempting to split cluster {cluster_id} ({cluster_size} articles) - SOFT LIMIT")

        cluster_df = df[cluster_mask].copy()
        rich_text = (
            cluster_df['title'] + " " +
            cluster_df['full_content'].fillna("").str.slice(0, SUBCLUSTER_CONTENT_LENGTH)
        )
        sub_embeddings = model.encode(rich_text.tolist(), show_progress_bar=False)

        sub_clusterer = DBSCAN(eps=SUBCLUSTER_EPS * 1.2, min_samples=SUBCLUSTER_MIN_SAMPLES)
        sub_labels = sub_clusterer.fit_predict(sub_embeddings)

        n_subclusters = len([lbl for lbl in np.unique(sub_labels) if lbl != -1])

        if n_subclusters >= 2:
            subcluster_sizes = pd.Series(sub_labels[sub_labels != -1]).value_counts()
            if all(size >= MIN_CLUSTER_SIZE_FINAL for size in subcluster_sizes):
                print(f"{'  ' * iteration}    → Splitting into {n_subclusters} natural subclusters")

                for orig_idx, sub_label in zip(cluster_df.index, sub_labels):
                    if sub_label == -1:
                        df.loc[orig_idx, 'cluster_id'] = -1
                    else:
                        df.loc[orig_idx, 'cluster_id'] = next_cluster_id + sub_label

                next_cluster_id += n_subclusters
            else:
                print(f"{'  ' * iteration}    → Sub-clusters too small, keeping original")
        else:
            print(f"{'  ' * iteration}    → No natural sub-topics found, keeping original")

    if iteration < 3:
        return _split_oversized_clusters(df, embeddings, model, iteration + 1)
    else:
        print(f"{'  ' * iteration}⚠️  Max recursion depth reached")
        return df


# ──────────────────────────────────────────────────────────────────────────────
# VISUAL EXAGGERATION
# ──────────────────────────────────────────────────────────────────────────────
def _apply_visual_exaggeration(projections, labels):
    adjusted = projections.copy()
    unique_labels = np.unique(labels)

    centroids = {}
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        centroids[lbl] = projections[mask].mean(axis=0)

    for lbl, centroid in centroids.items():
        mask = labels == lbl
        for i in np.where(mask)[0]:
            vec = centroid - projections[i]
            adjusted[i] = projections[i] + vec * PULL_FACTOR

    noise_mask = labels == -1
    if noise_mask.any() and centroids:
        for i in np.where(noise_mask)[0]:
            dists = [np.linalg.norm(projections[i] - c) for c in centroids.values()]
            nearest_centroid = list(centroids.values())[np.argmin(dists)]
            vec = projections[i] - nearest_centroid
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                adjusted[i] = projections[i] + (vec / norm) * PUSH_FACTOR

    return adjusted


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def build_cluster_cache():
    # ENV VARIABLE MUSH
    db_url = os.getenv("DATABASE_URL")
    # ENV VARIABLE MUSH
    supabase_url = os.getenv("SUPABASE_URL")
    # ENV VARIABLE MUSH
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not db_url:
        raise RuntimeError("Missing env var: DATABASE_URL")
    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing env vars: SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY")

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    target_date = pendulum.now(LOCAL_TZ).subtract(days=1).date()
    date_str = str(target_date)
    output_filename = f"{date_str}.json"

    print(f"=== Building cluster cache for {date_str} ===")

    # ── Load articles from DB ──────────────────────────────────────────────────
    # CONNECTION MUSH
    engine = create_engine(db_url)

    query = text("""
        SELECT article_id, source_name, author, title, url, publish_date, full_content
        FROM article_data
        WHERE publish_date = :target_date
    """)
    df = pd.read_sql(query, engine, params={"target_date": date_str})

    if df.empty:
        raise ValueError(f"No articles found for {date_str}")

    print(f"Loaded {len(df)} articles from DB")

    # ── Embed ──────────────────────────────────────────────────────────────────
    df["full_content"] = df["full_content"].fillna("")
    df["embed_text"] = df["title"] + " " + df["full_content"].str.slice(0, EMBED_CONTENT_LENGTH)

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")
    embeddings = model.encode(df["embed_text"].tolist(), show_progress_bar=True)
    print(f"✓ Embeddings shape: {embeddings.shape}")

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        spread=UMAP_SPREAD,
        random_state=UMAP_RANDOM_STATE,
        metric="cosine",
    )
    projections = reducer.fit_transform(embeddings)
    print(f"✓ UMAP complete: {projections.shape}")

    # ── Outlier compression ────────────────────────────────────────────────────
    center = np.median(projections, axis=0)
    dists = np.linalg.norm(projections - center, axis=1)
    median_dist = np.median(dists)
    threshold = median_dist * OUTLIER_COMPRESS_THRESHOLD_MULTIPLIER
    compressed = projections.copy()
    for i, d in enumerate(dists):
        if d > threshold:
            factor = threshold + np.log1p(d - threshold) * OUTLIER_LOG_SCALE
            direction = (projections[i] - center) / d
            compressed[i] = center + direction * factor
    projections = compressed

    # ── DBSCAN ────────────────────────────────────────────────────────────────
    print(f"\nDBSCAN clustering on raw embeddings (eps={DBSCAN_INITIAL_EPS}, min_samples={DBSCAN_INITIAL_MIN_SAMPLES})...")
    print("  Running on 768-dim embeddings with cosine metric — decoupled from visual spread")
    clusterer = DBSCAN(
        eps=DBSCAN_INITIAL_EPS,
        min_samples=DBSCAN_INITIAL_MIN_SAMPLES,
        metric='cosine',       # semantic distance, not euclidean 3D distance
    )
    labels = clusterer.fit_predict(embeddings)  # embeddings, NOT projections
    df["cluster_id"] = labels

    n_clusters = int((labels != -1).any() and labels[labels != -1].max() + 1)
    n_noise = int((labels == -1).sum())
    print(f"✓ Initial: {n_clusters} clusters, {n_noise} noise points")

    # ── Hierarchical splitting ─────────────────────────────────────────────────
    df = _split_oversized_clusters(df, embeddings, model)
    labels = df["cluster_id"].values
    n_clusters = int((labels != -1).any() and labels[labels != -1].max() + 1)
    n_noise = int((labels == -1).sum())
    print(f"✓ After splitting: {n_clusters} clusters, {n_noise} noise points")

    # ── Visual exaggeration ────────────────────────────────────────────────────
    projections = _apply_visual_exaggeration(projections, labels)
    df["x"] = projections[:, 0].astype(float)
    df["y"] = projections[:, 1].astype(float)
    df["z"] = projections[:, 2].astype(float)

    # ── Label clusters ─────────────────────────────────────────────────────────
    print("\nLabelling clusters...")
    palette = _cluster_colors(max(n_clusters, 1))
    clusters_out = []

    for cluster_id in sorted(df["cluster_id"].unique()):
        cluster_df = df[df["cluster_id"] == cluster_id]
        titles = cluster_df["title"].tolist()

        if cluster_id == -1:
            cluster_name = "Uncategorized"
            color = "#808080"
        else:
            print(f"\n  Cluster {cluster_id} ({len(titles)} articles):")
            cluster_name = _label_cluster_groq(titles)
            color = palette[int(cluster_id) % len(palette)]

        relevance_map = {
            row["article_id"]: _relevance_score(row["title"], cluster_name, model)
            for _, row in cluster_df.iterrows()
        }

        clusters_out.append({
            "cluster_id": int(cluster_id),
            "cluster_name": cluster_name,
            "article_count": len(titles),
            "color": color,
            "center_x": float(cluster_df["x"].mean()),
            "center_y": float(cluster_df["y"].mean()),
            "center_z": float(cluster_df["z"].mean()),
            "relevance_map": relevance_map,
        })

    # ── Build articles list ────────────────────────────────────────────────────
    all_relevance = {}
    for c in clusters_out:
        all_relevance.update(c.get("relevance_map", {}))

    articles_out = []
    for _, row in df.iterrows():
        articles_out.append({
            "article_id": str(row["article_id"]),
            "title": str(row["title"]),
            "source_name": str(row["source_name"]),
            "author": str(row["author"]) if pd.notna(row["author"]) else None,
            "url": str(row["url"]),
            "publish_date": str(row["publish_date"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "cluster_id": int(row["cluster_id"]),
            "relevance_score": float(all_relevance.get(str(row["article_id"]), 0.0)),
        })

    clusters_final = [
        {k: v for k, v in c.items() if k != "relevance_map"}
        for c in clusters_out
    ]

    # ── Build payload ──────────────────────────────────────────────────────────
    payload = {
        "date": date_str,
        "total_articles": len(articles_out),
        "articles": articles_out,
        "clusters": clusters_final,
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "umap_params": {
                "n_neighbors": UMAP_N_NEIGHBORS,
                "min_dist": UMAP_MIN_DIST,
                "spread": UMAP_SPREAD,
                "metric": "cosine",
            },
            # Fixed: was referencing undefined DBSCAN_EPS / DBSCAN_MIN_SAMPLES
            "dbscan_params": {
                "initial_eps": DBSCAN_INITIAL_EPS,
                "initial_min_samples": DBSCAN_INITIAL_MIN_SAMPLES,
                "subcluster_eps": SUBCLUSTER_EPS,
                "subcluster_min_samples": SUBCLUSTER_MIN_SAMPLES,
            },
        },
    }

    json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    print(f"\nPayload size: {len(json_bytes) / 1024:.1f} KB")

    # ── Upload to Supabase Storage ─────────────────────────────────────────────
    print(f"Uploading {output_filename} to Supabase Storage bucket '{SUPABASE_BUCKET}'...")

    # CONNECTION MUSH
    supabase_client = create_client(supabase_url, supabase_key)

    # Try to remove existing file first (upsert-style — Supabase Storage doesn't support true upsert)
    try:
        supabase_client.storage.from_(SUPABASE_BUCKET).remove([output_filename])
    except Exception:
        pass  # File didn't exist yet, that's fine

    # CONNECTION MUSH
    supabase_client.storage.from_(SUPABASE_BUCKET).upload(
        path=output_filename,
        file=json_bytes,
        file_options={"content-type": "application/json"},
    )

    print(f"=== Done! Uploaded {output_filename} to Supabase Storage ===")


if __name__ == "__main__":
    build_cluster_cache()