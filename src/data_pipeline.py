"""End-to-end data preparation pipeline for ESRB enrichment and ML prep.

This script loads the Steam, ESRB, and Steam Achievement Ranking datasets,
cleans them, merges them via fuzzy title matching, enriches the result with
IGDB content descriptors, and saves a unified dataset for downstream ML work.
"""

from __future__ import annotations

import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pandas.errors import ParserWarning
from rapidfuzz import fuzz, process

STOP_TOKENS = {"the", "a", "an", "and"}


@dataclass
class DataPaths:
    """Centralizes all filesystem locations used by the pipeline."""

    steam_games: Path = Path("data/raw/steam_games_dataset/games.csv")
    esrb_ratings: Path = Path("data/raw/video_game_ratings_by_esrb/Video_games_esrb_rating.csv")
    steam_achievements: Path = Path(
        "data/raw/steam_achievment_rankings/amended_first_200k_players.csv"
    )
    processed_dir: Path = Path("data/processed")
    output_file: Path = Path("data/processed/merged_dataset.csv")


# --- Logging helpers -------------------------------------------------------


def log(message: str) -> None:
    """Lightweight pipeline logging."""
    print(f"[DataPipeline] {message}", flush=True)


# --- Utility helpers -------------------------------------------------------


def normalize_title(value: Optional[str]) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for fuzzy matching."""
    if not isinstance(value, str):
        return ""
    value = value.lower()
    value = re.sub(r"(tm|®|©)", "", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def build_bucket_key(value: str) -> str:
    """Hash a normalized title into a short bucket key."""
    tokens = value.split()
    for token in tokens:
        if token and token not in STOP_TOKENS:
            return token[:4]
    return (value[:4] if value else "") or "#"


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise return an empty DataFrame."""
    if not path.exists():
        log(f"File not found: {path} -- returning empty frame.")
        return pd.DataFrame()
    df = pd.read_csv(path, **kwargs)
    log(f"Loaded {len(df):,} rows from {path}.")
    return df


def to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values and replace infinities with NA."""
    numeric = pd.to_numeric(series, errors="coerce")
    if isinstance(numeric, pd.Series):
        numeric = numeric.replace([np.inf, -np.inf], pd.NA)
        numeric = numeric.astype("Float64")
    return numeric


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


# --- Dataset-specific loaders ----------------------------------------------


def load_steam_games(paths: DataPaths) -> pd.DataFrame:
    """Load and clean the Steam Games dataset."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ParserWarning)
        df = safe_read_csv(paths.steam_games, index_col=False, low_memory=False)
    if df.empty:
        return df

    rename_map = {
        "AppID": "app_id",
        "Name": "steam_name",
        "Release date": "release_date",
        "Estimated owners": "estimated_owners",
        "Required age": "required_age",
        "Price": "price_usd",
        "Score rank": "score_rank",
        "User score": "user_score",
        "Positive": "reviews_positive",
        "Negative": "reviews_negative",
        "Average playtime forever": "avg_playtime_forever",
        "Median playtime forever": "median_playtime_forever",
        "Developers": "developers",
        "Publishers": "publishers",
        "Genres": "genres",
        "Tags": "tags",
    }
    df = df.rename(columns=rename_map)
    df["app_id"] = to_numeric_series(df.get("app_id")).astype("Int64")

    for col in ("price_usd", "user_score", "reviews_positive", "reviews_negative"):
        if col in df.columns:
            df[col] = to_numeric_series(df[col])

    df["steam_title_normalized"] = df.get("steam_name", "").apply(normalize_title)
    df["release_year"] = pd.to_datetime(
        df.get("release_date"), errors="coerce"
    ).dt.year.astype("Int64")

    core_columns = [
        "app_id",
        "steam_name",
        "steam_title_normalized",
        "release_date",
        "release_year",
        "estimated_owners",
        "price_usd",
        "score_rank",
        "user_score",
        "reviews_positive",
        "reviews_negative",
        "avg_playtime_forever",
        "median_playtime_forever",
        "developers",
        "publishers",
        "genres",
        "tags",
    ]
    available_columns = [column for column in core_columns if column in df.columns]
    log(f"Steam games cleaned with {len(df):,} entries.")
    return df[available_columns].copy()


def load_esrb_ratings(paths: DataPaths) -> pd.DataFrame:
    """Load and tidy ESRB ratings dataset."""
    df = safe_read_csv(paths.esrb_ratings)
    if df.empty:
        return df

    df = df.rename(columns={"title": "esrb_title", "esrb_rating": "esrb_maturity_rating"})
    df["esrb_title_normalized"] = df["esrb_title"].apply(normalize_title)
    descriptor_cols = [c for c in df.columns if c not in ("esrb_title", "console", "esrb_maturity_rating", "esrb_title_normalized")]
    for col in descriptor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    log(f"ESRB ratings cleaned with {len(df):,} entries.")
    return df


ACHIEVEMENT_PATTERN = re.compile(
    r"\((?P<game>.+?)\s*[–-]\s*(?P<score>\d+(?:\.\d+)?)\s*points?\)",
    flags=re.UNICODE,
)


def parse_best_achievements(cell: str) -> List[Tuple[str, float]]:
    """Extract (game, score) pairs from the textual achievements column."""
    if not isinstance(cell, str):
        return []
    matches = ACHIEVEMENT_PATTERN.findall(cell)
    parsed = []
    for game, score in matches:
        game = game.strip()
        try:
            parsed.append((game, float(score)))
        except ValueError:
            continue
    return parsed


def load_steam_achievement_rankings(
    paths: DataPaths, max_rows: Optional[int] = 50000
) -> pd.DataFrame:
    """Aggregate achievement rankings into per-game engagement signals."""
    read_kwargs = {}
    if max_rows and max_rows > 0:
        read_kwargs["nrows"] = max_rows
    df = safe_read_csv(paths.steam_achievements, **read_kwargs)
    if df.empty:
        return df

    if max_rows and max_rows > 0:
        log(
            "Steam achievement rows limited to "
            f"{len(df):,} (set ACHIEVEMENT_ROW_LIMIT=0 for full dataset)."
        )

    if "Best achievements" not in df.columns:
        log("Steam achievement rankings missing 'Best achievements' column; skipping.")
        return pd.DataFrame()

    records: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        best_achievements = parse_best_achievements(row.get("Best achievements"))
        if not best_achievements:
            continue
        player_rank = pd.to_numeric(row.get("Rank"), errors="coerce")
        completion_rate = pd.to_numeric(row.get("100%"), errors="coerce")
        for game_name, score in best_achievements:
            records.append(
                {
                    "achievement_game": game_name,
                    "achievement_score": score,
                    "player_rank": player_rank,
                    "player_completion_pct": completion_rate,
                }
            )

    if not records:
        log("No per-game achievement records parsed; skipping enrichment.")
        return pd.DataFrame()

    parsed_df = pd.DataFrame(records)
    aggregated = (
        parsed_df.groupby("achievement_game")
        .agg(
            achievement_mentions=("achievement_score", "count"),
            achievement_score_mean=("achievement_score", "mean"),
            achievement_score_max=("achievement_score", "max"),
            player_rank_mean=("player_rank", "mean"),
            player_completion_pct_mean=("player_completion_pct", "mean"),
        )
        .reset_index()
    )
    aggregated["achievement_title_normalized"] = aggregated["achievement_game"].apply(
        normalize_title
    )
    log(
        "Steam achievement rankings aggregated into "
        f"{len(aggregated):,} unique game signals."
    )
    return aggregated


# --- Fuzzy matching --------------------------------------------------------


def fuzzy_enrich(
    base_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    base_col: str,
    lookup_col: str,
    prefix: str,
    score_cutoff: int = 88,
) -> pd.DataFrame:
    """Attach lookup columns to base_df via fuzzy matching on normalized titles."""
    if base_df.empty or lookup_df.empty:
        return base_df

    base_choices_all = {
        idx: value for idx, value in base_df[base_col].fillna("").items() if value
    }
    if not base_choices_all:
        log("No base titles available for fuzzy enrichment.")
        return base_df

    buckets: Dict[str, Dict[int, str]] = {}
    for idx, value in base_choices_all.items():
        key = build_bucket_key(value)
        buckets.setdefault(key, {})[idx] = value

    match_records: List[Tuple[int, int, float]] = []
    for lookup_idx, query in lookup_df[lookup_col].fillna("").items():
        if not query:
            continue
        key = build_bucket_key(query)
        choices = buckets.get(key) or base_choices_all
        match = process.extractOne(
            query,
            choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=score_cutoff,
        )
        if match:
            _, score, base_index = match
            match_records.append((base_index, lookup_idx, score))

    if not match_records:
        log(f"No fuzzy matches found for prefix '{prefix}'.")
        return base_df

    matches_df = pd.DataFrame(
        match_records, columns=["base_index", "lookup_index", "match_score"]
    )
    matches_df = (
        matches_df.sort_values("match_score", ascending=False)
        .drop_duplicates(subset="base_index")
        .set_index("base_index")
    )

    enriched = base_df.copy()
    enriched[prefix + "match_score"] = matches_df["match_score"]
    lookup_prefixed = lookup_df.add_prefix(prefix).copy()

    enriched = enriched.merge(
        matches_df[["lookup_index"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    enriched = enriched.merge(
        lookup_prefixed,
        left_on="lookup_index",
        right_index=True,
        how="left",
    )
    enriched = enriched.drop(columns=["lookup_index"])

    matched_count = matches_df.index.nunique()
    log(
        f"Fuzzy enriched '{prefix}' columns for {matched_count:,} out of {len(base_df):,} rows."
    )
    return enriched


# --- IGDB enrichment -------------------------------------------------------


IGDB_GAMES_URL = "https://api.igdb.com/v4/games"
IGDB_CONTENT_URL = "https://api.igdb.com/v4/age_rating_content_descriptions"
IGDB_GENRE_URL = "https://api.igdb.com/v4/genres"
IGDB_AGE_RATINGS_URL = "https://api.igdb.com/v4/age_ratings"


def igdb_request(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    body: str,
) -> Optional[List[Dict]]:
    """Make a single IGDB API request and return JSON payload."""
    try:
        response = session.post(url, headers=headers, data=body, timeout=10)
        if response.status_code != 200:
            log(f"IGDB request failed ({response.status_code}): {response.text[:120]}")
            return None
        return response.json()
    except requests.RequestException as exc:
        log(f"IGDB request error: {exc}")
        return None


def fetch_age_descriptor_names(
    session: requests.Session, headers: Dict[str, str], descriptor_ids: Sequence[int]
) -> List[str]:
    """Translate age rating descriptor IDs into human-readable names."""
    valid_ids = [int(i) for i in descriptor_ids if pd.notna(i)]
    if not valid_ids:
        return []
    ids_list = ", ".join(str(i) for i in valid_ids)
    body = f"fields name; where id = ({ids_list});"
    payload = igdb_request(session, IGDB_CONTENT_URL, headers, body)
    if not payload:
        return []
    return [item.get("name") for item in payload if item.get("name")]


def fetch_genre_names(
    session: requests.Session, headers: Dict[str, str], genre_ids: Sequence[int]
) -> List[str]:
    """Translate genre IDs into names."""
    valid_ids = [int(i) for i in genre_ids if pd.notna(i)]
    if not valid_ids:
        return []
    ids_list = ", ".join(str(i) for i in valid_ids)
    body = f"fields name; where id = ({ids_list});"
    payload = igdb_request(session, IGDB_GENRE_URL, headers, body)
    if not payload:
        return []
    return [item.get("name") for item in payload if item.get("name")]


def fetch_age_rating_details(
    session: requests.Session, headers: Dict[str, str], rating_ids: Sequence[int]
) -> Dict[int, Dict]:
    """Fetch rating metadata (rating, category, descriptor IDs) for age ratings."""
    valid_ids = [int(i) for i in rating_ids if pd.notna(i)]
    if not valid_ids:
        return {}
    ids_list = ", ".join(str(i) for i in valid_ids)
    body = "fields rating, category, content_descriptions; "
    body += f"where id = ({ids_list});"
    payload = igdb_request(session, IGDB_AGE_RATINGS_URL, headers, body)
    if not payload:
        return {}
    return {item["id"]: item for item in payload if item.get("id") is not None}


def add_igdb_descriptors(
    df: pd.DataFrame,
    max_requests: int = 200,
) -> pd.DataFrame:
    """Enrich dataframe with IGDB content descriptors using API credentials."""
    if df.empty:
        return df

    client_id = os.getenv("IGDB_CLIENT_ID")
    bearer_token = os.getenv("IGDB_BEARER_TOKEN")
    if not client_id or not bearer_token:
        log("IGDB credentials missing in .env; skipping descriptor enrichment.")
        return df

    session = requests.Session()
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {bearer_token}",
    }

    title_lookup = OrderedDict()
    for normalized, title in zip(df["steam_title_normalized"], df["steam_name"]):
        if normalized and normalized not in title_lookup:
            title_lookup[normalized] = title
        if len(title_lookup) >= max_requests:
            break

    igdb_records = []
    failure_streak = 0
    for normalized, title in title_lookup.items():
        body = f'search "{title}"; fields name,summary,genres,age_ratings; limit 1;'
        payload = igdb_request(session, IGDB_GAMES_URL, headers, body)
        if not payload:
            failure_streak += 1
            if failure_streak >= 5:
                log(
                    "Multiple consecutive IGDB errors encountered. "
                    "Stopping IGDB enrichment early."
                )
                break
            continue
        failure_streak = 0

        game = payload[0]
        genre_ids = game.get("genres", [])
        rating_ids = game.get("age_ratings", [])
        rating_details = fetch_age_rating_details(session, headers, rating_ids)

        descriptor_ids: List[int] = []
        rating_labels: List[str] = []
        for rating in rating_details.values():
            descriptor_ids.extend(rating.get("content_descriptions") or [])
            rating_value = rating.get("rating")
            category_value = rating.get("category")
            label_bits = []
            if category_value is not None:
                label_bits.append(f"cat:{category_value}")
            if rating_value is not None:
                label_bits.append(f"rating:{rating_value}")
            if label_bits:
                rating_labels.append("_".join(label_bits))

        descriptor_names = fetch_age_descriptor_names(session, headers, descriptor_ids)
        genre_names = fetch_genre_names(session, headers, genre_ids)
        igdb_records.append(
            {
                "steam_title_normalized": normalized,
                "igdb_name": game.get("name"),
                "igdb_summary": game.get("summary"),
                "igdb_descriptors": "; ".join(sorted(set(descriptor_names))),
                "igdb_genres": "; ".join(sorted(set(genre_names))),
                "igdb_age_ratings": "; ".join(sorted(set(rating_labels))),
            }
        )
        time.sleep(0.25)  # Politeness delay to avoid rate limits.

    if not igdb_records:
        log("IGDB enrichment yielded no records; leaving dataframe unchanged.")
        return df

    igdb_df = pd.DataFrame(igdb_records)
    merged = df.merge(igdb_df, on="steam_title_normalized", how="left")
    log(f"IGDB enrichment applied to {igdb_df['igdb_name'].notna().sum():,} rows.")
    return merged


# --- Orchestration ---------------------------------------------------------


def merge_datasets(
    steam_df: pd.DataFrame,
    esrb_df: pd.DataFrame,
    achievements_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all datasets using fuzzy matching and available keys."""
    merged = steam_df.copy()
    merged = fuzzy_enrich(
        merged,
        esrb_df,
        base_col="steam_title_normalized",
        lookup_col="esrb_title_normalized",
        prefix="esrb_",
        score_cutoff=86,
    )
    merged = fuzzy_enrich(
        merged,
        achievements_df,
        base_col="steam_title_normalized",
        lookup_col="achievement_title_normalized",
        prefix="achievement_",
        score_cutoff=82,
    )
    return merged


def resolve_achievement_row_limit(default: int = 50000) -> Optional[int]:
    """Read ACHIEVEMENT_ROW_LIMIT env var and convert to int."""
    raw_value = os.getenv("ACHIEVEMENT_ROW_LIMIT")
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        log(
            "Invalid ACHIEVEMENT_ROW_LIMIT value. Falling back to "
            f"default ({default})."
        )
        return default
    if parsed <= 0:
        return None
    return parsed


def run_pipeline(paths: Optional[DataPaths] = None) -> Path:
    """Execute the full data pipeline from raw inputs to processed output."""
    load_dotenv()
    paths = paths or DataPaths()
    ensure_directory(paths.processed_dir)

    log("Starting data pipeline run.")
    steam_df = load_steam_games(paths)
    esrb_df = load_esrb_ratings(paths)
    achievement_limit = resolve_achievement_row_limit()
    achievements_df = load_steam_achievement_rankings(paths, max_rows=achievement_limit)
    merged_df = merge_datasets(steam_df, esrb_df, achievements_df)
    enriched_df = add_igdb_descriptors(merged_df)

    enriched_df.to_csv(paths.output_file, index=False)
    log(
        f"Pipeline finished. Final dataset saved to {paths.output_file} "
        f"({len(enriched_df):,} rows)."
    )
    return paths.output_file


if __name__ == "__main__":
    run_pipeline()
