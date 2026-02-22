"""
daily_reader.py
===============
Fetches yesterday's news articles from NewsAPI, scrapes full content,
and stores them in the Supabase PostgreSQL database.

Run manually:  python daily_reader.py
Run via CI:    GitHub Actions workflow (.github/workflows/ingest.yml)
"""

import hashlib
import os
import random
import time
from datetime import datetime, timedelta
from urllib.parse import quote, urlparse

import newspaper
import pandas as pd
import pendulum
import requests
from sqlalchemy import create_engine, text

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
LOCAL_TZ = "America/Chicago"

SOURCES_LIST = [
    'abc-news-au', 'al-jazeera-english', 'associated-press', 'bbc-news',
    'breitbart-news', 'business-insider', 'buzzfeed', 'cbc-news',
    'cbs-news', 'financial-post', 'fortune', 'fox-news', 'fox-sports',
    'hacker-news', 'nbc-news', 'rte', 'techradar', 'the-times-of-india',
    'the-verge', 'usa-today'
]


def get_browser_config():
    config = newspaper.Config()
    config.browser_user_agent = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
    )
    config.request_timeout = 15
    config.fetch_images = False
    config.memoize_articles = False
    return config


def fetch_and_store_news():
    # ENV VARIABLE MUSH
    api_key = os.getenv("NEWSROOM_API_KEY")
    # ENV VARIABLE MUSH
    db_url = os.getenv("DATABASE_URL")

    if not api_key:
        raise RuntimeError("Missing env var: NEWSROOM_API_KEY")
    if not db_url:
        raise RuntimeError("Missing env var: DATABASE_URL")

    # Normalize postgres:// → postgresql:// for SQLAlchemy
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    # CONNECTION MUSH
    engine = create_engine(db_url)

    target_date = pendulum.now(LOCAL_TZ).subtract(days=1).to_date_string()
    user_config = get_browser_config()

    print(f"=== Fetching news for {target_date} ===")

    for source in SOURCES_LIST:
        print(f"\n--- Processing Source: {source} ---")
        batch_data = []

        for page in range(1, 2):  # max_pages = 1 per source
            print(f"  Fetching page {page}...")
            params = {
                'sources': source,
                'from': target_date,
                'to': target_date,
                'sortBy': 'popularity',
                'pageSize': 100,
                'page': page,
                'apiKey': api_key,
            }

            try:
                response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=30)
                res = response.json()

                if res.get('status') != 'ok':
                    print(f"  API error for {source} page {page}: {res.get('message')}")
                    break

                articles = res.get('articles', [])
                if not articles:
                    print(f"  No articles found for {source} page {page}.")
                    break

                for art in articles:
                    print(f"  Reading: {art['title']}")
                    article_url = art.get('url')
                    if not article_url or "[Removed]" in art['title']:
                        continue

                    try:
                        parsed_url = urlparse(article_url)
                        safe_url = parsed_url._replace(
                            path=quote(parsed_url.path),
                            query=quote(parsed_url.query, safe='=&'),
                            params=quote(parsed_url.params),
                        ).geturl()

                        article_id = hashlib.md5(str(safe_url).encode()).hexdigest()[:16]
                        source_name = art.get('source', {}).get('name', 'unknown')

                        time.sleep(random.uniform(0.8, 1.5))

                        scraper = newspaper.Article(safe_url, config=user_config)
                        scraper.download()
                        scraper.parse()

                        full_text = scraper.text or ""
                        if full_text and "consent" in full_text.lower() and len(full_text) < 500:
                            full_text = "Blocked by Consent Wall"

                        batch_data.append({
                            'article_id': article_id,
                            'source_id': hashlib.md5(str(source_name).strip().lower().encode()).hexdigest()[:16],
                            'source_name': source_name,
                            'author': ", ".join(scraper.authors) if scraper.authors else art.get('author'),
                            'title': art['title'],
                            'url': article_url,
                            'full_content': full_text,
                            'publish_date': art.get('publishedAt'),
                        })

                    except Exception as e:
                        print(f"    !! Skipping {article_url}: {e}")
                        continue

            except Exception as api_err:
                print(f"  API connection error for {source}: {api_err}")
                break

        # Insert batch for this source
        if batch_data:
            df = pd.DataFrame(batch_data)
            df['publish_date'] = pd.to_datetime(df['publish_date']).dt.date

            # CONNECTION MUSH
            with engine.begin() as sql_conn:
                df.to_sql('stg_news_articles', sql_conn, if_exists='replace', index=False)
                sql_conn.execute(text("""
                    INSERT INTO article_data
                        (article_id, source_id, source_name, author, title, url, full_content, publish_date)
                    SELECT article_id, source_id, source_name, author, title, url, full_content, publish_date
                    FROM stg_news_articles
                    ON CONFLICT DO NOTHING;
                """))
                sql_conn.execute(text("DROP TABLE IF EXISTS stg_news_articles;"))

            print(f"  Stored {len(batch_data)} articles for {source}")
        else:
            print(f"  No data to store for {source}")

    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    fetch_and_store_news()