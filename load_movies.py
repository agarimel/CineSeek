"""Load and clean the movie metadata used by CineSeek."""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path


MOVIES_CSV = Path(__file__).parent / "movies" / "movies_metadata.csv"
KEYWORDS_CSV = Path(__file__).parent / "movies" / "keywords.csv"


def parse_genres(raw_genres: str) -> list[str]:
    """Convert the dataset's serialized genre objects into genre names."""
    if not raw_genres:
        return []

    try:
        genres = ast.literal_eval(raw_genres)
    except (SyntaxError, ValueError):
        return []

    if not isinstance(genres, list):
        return []

    return [
        genre["name"].strip()
        for genre in genres
        if isinstance(genre, dict)
        and isinstance(genre.get("name"), str)
        and genre["name"].strip()
    ]


def parse_keywords(raw_keywords: str) -> list[str]:
    """Convert the dataset's serialized keyword objects into names."""
    if not raw_keywords:
        return []

    try:
        keywords = ast.literal_eval(raw_keywords)
    except (SyntaxError, ValueError):
        return []

    if not isinstance(keywords, list):
        return []

    return [
        keyword["name"].strip()
        for keyword in keywords
        if isinstance(keyword, dict)
        and isinstance(keyword.get("name"), str)
        and keyword["name"].strip()
    ]


def load_movie_keywords(csv_path: Path = KEYWORDS_CSV) -> dict[str, list[str]]:
    """Return keyword names keyed by the metadata dataset's movie ID."""
    keywords_by_movie_id = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            movie_id = (row.get("id") or "").strip()
            if movie_id:
                keywords_by_movie_id[movie_id] = parse_keywords(
                    row.get("keywords") or ""
                )
    return keywords_by_movie_id


def load_movies(csv_path: Path = MOVIES_CSV) -> list[dict]:
    """Return all valid movies with the useful fields cleaned."""
    ranked_movies = []

    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            title = (row.get("title") or "").strip()
            overview = (row.get("overview") or "").strip()
            if not title or not overview:
                continue

            release_date = (row.get("release_date") or "").strip()
            release_year = (
                int(release_date[:4])
                if len(release_date) >= 4 and release_date[:4].isdigit()
                else None
            )

            try:
                popularity = float(row.get("popularity") or 0)
            except (TypeError, ValueError):
                popularity = 0.0
            if not math.isfinite(popularity):
                popularity = 0.0

            ranked_movies.append(
                (
                    popularity,
                    {
                    "movie_id": (row.get("id") or "").strip(),
                    "title": title,
                    "overview": overview,
                    "release_date": release_date or None,
                    "release_year": release_year,
                    "genres": parse_genres(row.get("genres") or ""),
                    },
                )
            )

    ranked_movies.sort(key=lambda item: item[0], reverse=True)
    return [movie for _, movie in ranked_movies]


if __name__ == "__main__":
    cleaned_movies = load_movies()
    print(f"Loaded {len(cleaned_movies):,} cleaned movies from {MOVIES_CSV}")
    print("First 5 cleaned records:")
    for movie in cleaned_movies[:5]:
        print(json.dumps(movie, ensure_ascii=False))
