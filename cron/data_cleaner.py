"""
data_cleaner.py
===============
Cleans yesterday's articles from the Supabase DB before clustering runs.

Runs between daily_reader.py and news_clustering.py in the GitHub Actions chain:
  ingest.yml → clean.yml → cluster.yml

Two cleaning rules applied to yesterday's data:
  1. Remove any row whose title contains 'GMT' (these are malformed/timestamp titles)
  2. Remove duplicate titles — when the same title appears more than once,
     keep exactly one row and delete the rest

Run manually:  python data_cleaner.py
Run via CI:    GitHub Actions workflow (.github/workflows/clean.yml)
"""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
LOCAL_TZ = "America/Chicago"


def clean_articles():
    # ENV VARIABLE MUSH
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise RuntimeError("Missing env var: DATABASE_URL")

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    # Target date = yesterday in Chicago time, matching what daily_reader.py ingested
    target_date = str(
        (datetime.now(ZoneInfo(LOCAL_TZ)) - timedelta(days=1)).date()
    )

    print(f"=== Cleaning articles for {target_date} ===")

    # CONNECTION MUSH
    engine = create_engine(db_url)

    with engine.begin() as conn:

        # ── Rule 1: Remove rows where title contains 'GMT' ────────────────────
        # These are malformed rows where the title is a raw timestamp string
        # e.g. "Mon, 21 Feb 2026 14:32:00 GMT"
        result = conn.execute(text("""
            DELETE FROM article_data
            WHERE publish_date = :target_date
              AND title LIKE '%GMT%'
        """), {"target_date": target_date})

        gmt_deleted = result.rowcount
        print(f"  Rule 1 (GMT titles):     removed {gmt_deleted} rows")

        # ── Rule 2: Remove duplicate titles, keep one row per title ───────────
        # Strategy: for each group of identical titles on this date, keep the
        # row with the smallest article_id (arbitrary but deterministic) and
        # delete all others in that group.
        #
        # This uses a self-join: find every article_id that is NOT the minimum
        # article_id for its title group, then delete those.
        result = conn.execute(text("""
            DELETE FROM article_data
            WHERE publish_date = :target_date
              AND article_id NOT IN (
                  SELECT MIN(article_id)
                  FROM article_data
                  WHERE publish_date = :target_date
                  GROUP BY title
              )
        """), {"target_date": target_date})

        dupes_deleted = result.rowcount
        print(f"  Rule 2 (duplicate titles): removed {dupes_deleted} rows")

    # Read back the final count to confirm
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM article_data
            WHERE publish_date = :target_date
        """), {"target_date": target_date})
        final_count = result.scalar()

    print(f"\n  Rows remaining for {target_date}: {final_count}")
    print(f"=== Cleaning complete: {gmt_deleted + dupes_deleted} total rows removed ===")


if __name__ == "__main__":
    clean_articles()