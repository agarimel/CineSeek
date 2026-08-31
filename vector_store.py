"""Shared configuration and helpers for the CineSeek vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "movies"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def movie_to_searchable_text(movie: dict) -> str:
    """Build the text whose meaning will represent a movie in the index."""
    year = movie.get("release_year") or "Unknown"
    genres = ", ".join(movie.get("genres") or []) or "Unknown"
    keywords = ", ".join(movie.get("keywords") or []) or "Unknown"
    return (
        f"Title: {movie['title']}\n"
        f"Overview: {movie['overview']}\n"
        f"Release year: {year}\n"
        f"Genres: {genres}\n"
        f"Keywords: {keywords}"
    )


def movie_to_metadata(movie: dict) -> dict[str, str | int]:
    """Return the compact fields stored alongside a movie embedding."""
    return {
        "movie_id": movie.get("movie_id") or "",
        "title": movie["title"],
        "release_year": movie.get("release_year") or "Unknown",
        "genres": ", ".join(movie.get("genres") or []),
    }


def load_embedding_model(*, local_files_only: bool = False) -> Any:
    """Load the free local Sentence Transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        local_files_only=local_files_only,
    )


def get_collection() -> Any:
    """Open the locally persisted Chroma movie collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "CineSeek movie semantic search index",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "hnsw:space": "cosine",
        },
    )
