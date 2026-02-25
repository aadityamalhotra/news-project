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
  4. Outlier compression (aesthetic)
  5. DBSCAN clustering on the 3D UMAP coordinates (euclidean)
     → clusters what you SEE, so visual groups = real clusters
  6. Hierarchical splitting: recursively split oversized clusters
  7. Visual exaggeration for cleaner separation
  8. Label each cluster with Groq / llama-3.1-8b-instant
  9. Score each article's semantic relevance to its cluster name
  10. Upload result to Supabase Storage as <YYYY-MM-DD>.json

GROQ SETUP:
  - Sign up free at console.groq.com
  - Create an API key
  - Add it as GROQ_API_KEY in GitHub Actions secrets
  - Free tier: 14,400 requests/day, more than enough for daily clustering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TUNING GUIDE — every parameter you can adjust and what it does
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UMAP_N_NEIGHBORS (default: 8)
  Controls how many nearby points UMAP considers when learning the layout.
  ↑ increase → more global structure, clusters spread further apart, smoother layout
  ↓ decrease → more local structure, tighter local groups, more fragmented layout
  Practical range: 5–30. Below 5 gets noisy. Above 20 loses local detail.

UMAP_MIN_DIST (default: 0.05)
  Controls how tightly points are packed within a cluster in the 3D projection.
  ↑ increase → points spread out within clusters, more open/airy look
  ↓ decrease → points compress into tight dense blobs, DBSCAN finds them more easily
  Practical range: 0.01–0.3. Lower = more clusters detected. Higher = prettier but fewer clusters.

UMAP_SPREAD (default: 2.5)
  Controls the overall scale of the 3D space — how far apart clusters sit from each other.
  ↑ increase → more 3D volume used, clusters more separated visually
  ↓ decrease → everything compresses toward the center
  NOTE: Also affects DBSCAN since clustering is now on 3D coords. If you raise spread,
  you must also raise DBSCAN_INITIAL_EPS proportionally.
  Practical range: 1.0–4.0.

DBSCAN_INITIAL_EPS (default: 0.8)
  The neighbourhood radius in 3D UMAP space. Two points are neighbours if their
  euclidean 3D distance is less than eps. THIS IS THE MOST IMPORTANT PARAMETER.
  ↑ increase → larger neighbourhoods, fewer but larger clusters, more points clustered
  ↓ decrease → smaller neighbourhoods, more clusters, more grey noise points
  Practical range: 0.3–2.0 when UMAP_SPREAD=2.5.
  Rule of thumb: if you see too many grey points → increase eps.
                 If clusters are merging topics that don't belong → decrease eps.

DBSCAN_INITIAL_MIN_SAMPLES (default: 8)
  Minimum number of points required to form a cluster core.
  ↑ increase → only denser groups become clusters, more grey noise
  ↓ decrease → smaller groups qualify as clusters, fewer grey points
  Practical range: 4–15. This is the second most impactful parameter after eps.

MAX_CLUSTER_SIZE_SOFT (default: 100)
  Clusters above this size are attempted to be split if natural sub-topics exist.
  ↑ increase → fewer splits attempted, larger clusters allowed
  ↓ decrease → more aggressive splitting, more final clusters

MAX_CLUSTER_SIZE_HARD (default: 150)
  Clusters above this are ALWAYS force-split regardless of sub-topic quality.
  ↑ increase → very large mixed clusters can survive
  ↓ decrease → hard cap enforced more strictly

MIN_CLUSTER_SIZE_FINAL (default: 8)
  After splitting, sub-clusters smaller than this are dissolved back to noise.
  ↑ increase → only larger sub-clusters survive splitting
  ↓ decrease → smaller sub-clusters survive, more final clusters from splitting

SUBCLUSTER_EPS (default: 0.5)
  The eps used when re-clustering inside an oversized cluster.
  Should be somewhat smaller than DBSCAN_INITIAL_EPS so it finds tighter sub-groups.
  ↑ increase → looser sub-clustering, fewer sub-clusters found
  ↓ decrease → tighter sub-clustering, more sub-clusters found

SUBCLUSTER_MIN_SAMPLES (default: 4)
  Min samples for sub-cluster cores. Lower than DBSCAN_INITIAL_MIN_SAMPLES intentionally.
  ↑ increase → sub-clusters must be denser to survive
  ↓ decrease → smaller sub-clusters survive

PULL_FACTOR (default: 0.35)
  Post-clustering visual tweak. Pulls each point toward its cluster centroid.
  ↑ increase → cluster members bunch tighter together visually (up to 1.0 = all collapse to center)
  ↓ decrease → cluster members stay where UMAP placed them
  Does NOT affect which points belong to which cluster.

PUSH_FACTOR (default: 0.45)
  Post-clustering visual tweak. Pushes grey noise points away from nearest cluster.
  ↑ increase → noise points scatter further from clusters
  ↓ decrease → noise points stay closer to clusters
  Does NOT affect which points are noise.

EMBED_CONTENT_LENGTH (default: 800)
  Characters of article body text appended to the title for embedding.
  ↑ increase → richer embeddings, better semantic distinction between articles
  ↓ decrease → faster embedding, more title-driven clustering
  Practical range: 400–1500.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# Groq API — free LLM inference
# ENV VARIABLE MUSH
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT = 30

# ── EMBEDDING MODEL ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBED_CONTENT_LENGTH = 800        # see tuning guide above

# ── UMAP settings ─────────────────────────────────────────────────────────────
# NOTE: DBSCAN runs on the 3D UMAP output, so UMAP and DBSCAN parameters
# are coupled. If you change UMAP_SPREAD, adjust DBSCAN_INITIAL_EPS too.
UMAP_N_COMPONENTS = 3
UMAP_N_NEIGHBORS = 8              # see tuning guide above
UMAP_MIN_DIST = 0.05              # low = tight blobs = easier for DBSCAN to find
UMAP_SPREAD = 2.5                 # see tuning guide above
UMAP_RANDOM_STATE = 42

# ── DBSCAN settings ───────────────────────────────────────────────────────────
# Runs on 3D UMAP coordinates (euclidean distance).
# This means visual groups = real clusters — what you see is what gets clustered.
# eps is in the same units as the 3D UMAP space (scaled by UMAP_SPREAD).
DBSCAN_INITIAL_EPS = 0.8          # see tuning guide above — START HERE when tuning
DBSCAN_INITIAL_MIN_SAMPLES = 15    # see tuning guide above

# ── Hierarchical splitting ─────────────────────────────────────────────────────
MAX_CLUSTER_SIZE_SOFT = 80       # see tuning guide above
MAX_CLUSTER_SIZE_HARD = 100       # see tuning guide above
MIN_CLUSTER_SIZE_FINAL = 8        # see tuning guide above
SUBCLUSTER_EPS = 0.5              # see tuning guide above — kept proportional to DBSCAN_INITIAL_EPS
SUBCLUSTER_MIN_SAMPLES = 4        # see tuning guide above
SUBCLUSTER_CONTENT_LENGTH = 1200

# ── Visual exaggeration (post-clustering only, does not affect cluster assignments) ──
PULL_FACTOR = 0.35                # see tuning guide above
PUSH_FACTOR = 0.45                # see tuning guide above

# ── Outlier compression ────────────────────────────────────────────────────────
OUTLIER_COMPRESS_THRESHOLD_MULTIPLIER = 2.0
OUTLIER_LOG_SCALE = 0.5

# ── Supabase Storage bucket name ───────────────────────────────────────────────
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
    colors = []
    # Use the Golden Ratio to spread hues evenly
    phi = 0.618033988749895
    hue = 0.5  # Starting point
    
    for i in range(n):
        hue = (hue + phi) % 1.0
        # Vary saturation and value slightly for more "texture"
        # Alternate saturation between 0.5 and 0.8 to distinguish neighbors
        saturation = 0.6 + (i % 2) * 0.2 
        value = 0.85 + (i % 3) * 0.05
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


# ──────────────────────────────────────────────────────────────────────────────
# GROQ LABELLING
# ──────────────────────────────────────────────────────────────────────────────
def _label_cluster_groq(titles: list, embeddings_subset: np.ndarray = None) -> str:
    """
    Ask Groq for a specific 3-5 word topic label.

    Improvements over naive approach:
    - Passes up to 40 titles (was 20) for more context
    - Selects the most central/representative titles using embedding similarity
      to the cluster centroid, so Groq sees the best signal not a random sample
    - Retries up to 3 times with exponential backoff on rate limit (429) errors
    - Prompt explicitly instructs country/person/event naming
    - Falls back to keyword extraction only after all retries exhausted
    """
    import time

    if not GROQ_API_KEY:
        print("  ⚠ GROQ_API_KEY not set — using keyword fallback")
        return _label_cluster_fallback(titles)

    # Select the most representative titles — those closest to the cluster centroid
    # This gives Groq the clearest signal about what the cluster is actually about
    if embeddings_subset is not None and len(embeddings_subset) == len(titles):
        centroid = embeddings_subset.mean(axis=0)
        sims = embeddings_subset @ centroid / (
            np.linalg.norm(embeddings_subset, axis=1) * np.linalg.norm(centroid) + 1e-9
        )
        top_indices = np.argsort(sims)[::-1][:40]
        selected_titles = [titles[i] for i in top_indices]
    else:
        # Fallback: take up to 40 titles as-is (for smaller clusters, all titles)
        selected_titles = titles[:40]

    titles_text = "\n".join(f"- {t}" for t in selected_titles)

    system_prompt = (
        "You are a senior news editor at an international wire service. "
        "Your job is to read a set of article headlines and write ONE precise topic label. "
        "\n\nCritical rules:"
        "\n- Output ONLY the label words. No explanation, no punctuation, no alternatives."
        "\n- Use 3 to 5 words maximum."
        "\n- Be SPECIFIC. Name the country, person, organization, or event if relevant."
        "\n- If headlines mention a specific country, include it (e.g. 'Mexico Drug Cartel Violence' not 'Drug Violence')."
        "\n- If headlines are about a named person, include their last name (e.g. 'Trump Immigration Executive Orders')."
        "\n- If headlines cover a specific event, name it (e.g. 'NFL Combine Draft Prospects')."
        "\n- Never output generic filler words like 'News', 'Update', 'Report', 'Latest'."
        "\n\nGood label examples:"
        "\n  Gaza Ceasefire Hostage Deal"
        "\n  Federal Reserve Interest Rates"
        "\n  Tesla Layoffs Musk Restructuring"
        "\n  Ukraine Russia Frontline Fighting"
        "\n  US Mexico Border Immigration"
        "\n  NBA Trade Deadline Moves"
        "\n  Israel Iran Military Strike"
        "\n  UK Starmer Budget Spending"
        "\n  Canada Trudeau Resignation Liberals"
        "\n  Silicon Valley AI Startup Funding"
    )

    for attempt in range(3):
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
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Here are {len(selected_titles)} headlines from the same news cluster.\n"
                                f"Headlines:\n{titles_text}\n\n"
                                f"Write the topic label (3-5 words, specific, no punctuation):"
                            ),
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens": 25,
                    "stop": ["\n", "|"],   # Groq allows max 4 stop tokens
                },
                timeout=GROQ_TIMEOUT,
            )

            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()

                # Strip any preamble the model snuck in
                for prefix in [
                    "Topic:", "Topic name:", "The topic is", "Based on",
                    "ONE TOPIC:", "Main topic:", "Answer:", "Label:", "Here is",
                    "The label", "Label:", "Category:",
                ]:
                    if raw.lower().startswith(prefix.lower()):
                        raw = raw[len(prefix):].strip().lstrip(":- ")

                raw = raw.split("\n")[0].strip().strip("\"'.,;:")
                raw = re.sub(r"^\d+\.\s*", "", raw)
                raw = re.sub(r"^[-•*]\s*", "", raw)
                # Remove trailing filler words
                raw = re.sub(r"\b(News|Update|Report|Latest|Coverage)$", "", raw, flags=re.IGNORECASE).strip()
                raw = " ".join(raw.split()[:5])

                if len(raw) >= 3:
                    print(f"  ✓ Groq label (attempt {attempt+1}): '{raw}'")
                    return raw

                print(f"  ⚠ Groq returned too-short label '{raw}' on attempt {attempt+1}, retrying...")

            elif resp.status_code == 429:
                # Rate limited — wait with exponential backoff
                wait = (2 ** attempt) * 3
                print(f"  ⚠ Groq rate limited (429), waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            else:
                print(f"  ✗ Groq HTTP {resp.status_code}: {resp.text[:200]}")
                break

        except Exception as exc:
            print(f"  ✗ Groq error on attempt {attempt+1}: {exc}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    print("  ✗ All Groq attempts failed — falling back to keyword extraction")
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

        sub_clusterer = DBSCAN(
            eps=SUBCLUSTER_EPS,
            min_samples=SUBCLUSTER_MIN_SAMPLES,
            metric='cosine',
        )
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

        sub_clusterer = DBSCAN(
            eps=SUBCLUSTER_EPS * 1.2,
            min_samples=SUBCLUSTER_MIN_SAMPLES,
            metric='cosine',
        )
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
    print(f"  n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, spread={UMAP_SPREAD}")
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

    # ── DBSCAN on 3D UMAP coordinates ─────────────────────────────────────────
    # Clustering on the 3D projection means visual groups = real clusters.
    # If you see a group of points close together, they will be clustered together.
    # eps is in euclidean 3D space — tune it alongside UMAP_SPREAD.
    print(f"\nDBSCAN clustering on 3D UMAP coordinates...")
    print(f"  eps={DBSCAN_INITIAL_EPS}, min_samples={DBSCAN_INITIAL_MIN_SAMPLES}, metric=euclidean")
    print(f"  (visual groups = real clusters — what you see is what gets clustered)")
    clusterer = DBSCAN(
        eps=DBSCAN_INITIAL_EPS,
        min_samples=DBSCAN_INITIAL_MIN_SAMPLES,
        metric='euclidean',
    )
    labels = clusterer.fit_predict(projections)  # projections = 3D UMAP output
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
            # Pass the raw embeddings for this cluster so Groq receives the most
            # representative titles (closest to centroid) rather than a random sample
            cluster_indices = cluster_df.index.tolist()
            cluster_embeddings = embeddings[cluster_indices]
            cluster_name = _label_cluster_groq(titles, embeddings_subset=cluster_embeddings)
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
            "dbscan_params": {
                "initial_eps": DBSCAN_INITIAL_EPS,
                "initial_min_samples": DBSCAN_INITIAL_MIN_SAMPLES,
                "metric": "euclidean_on_3d_umap",
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