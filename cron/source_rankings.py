"""
source_rankings.py
==================
Ranked list of news sources by international reach, editorial reputation,
and factual reliability. Used by news_clustering.py to select the most
authoritative article per highlight cluster.

Rank 1 = most authoritative. Sources not in this list get rank 999.
"""

SOURCE_RANKINGS = {
    # Tier 1 — Global wire services & flagship broadcasters
    "associated-press":         1,
    "bbc-news":                 2,
    "al-jazeera-english":       3,
    "reuters":                  4,   # may appear via scraping even if not in sources list

    # Tier 2 — Major national broadcasters & newspapers of record
    "abc-news-au":              5,
    "cbc-news":                 6,
    "rte":                      7,
    "nbc-news":                 8,
    "cbs-news":                 9,
    "usa-today":               10,

    # Tier 3 — Business & financial press
    "financial-post":          11,
    "fortune":                 12,
    "business-insider":        13,

    # Tier 4 — Tech & specialist press
    "hacker-news":             14,
    "the-verge":               15,
    "techradar":               16,

    # Tier 5 — International / regional
    "the-times-of-india":      17,

    # Tier 6 — Entertainment & mixed-format
    "buzzfeed":                18,
    "fox-sports":              19,

    # Tier 7 — Opinion-heavy / partisan outlets
    "fox-news":                20,
    "breitbart-news":          21,
}

# Fallback rank for any source not listed above
DEFAULT_RANK = 999


def get_source_rank(source_id: str) -> int:
    """Return the rank for a given source_id. Lower = more authoritative."""
    return SOURCE_RANKINGS.get(source_id.lower().strip(), DEFAULT_RANK)