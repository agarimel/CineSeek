"""Search CineSeek's local movie vector database from the terminal."""

from __future__ import annotations

import argparse
from typing import Any

from vector_store import CHROMA_PATH, get_collection, load_embedding_model


DEFAULT_RESULT_COUNT = 5


def search_movies(
    query: str,
    top_k: int = DEFAULT_RESULT_COUNT,
    *,
    collection: Any | None = None,
    model: Any | None = None,
) -> list[dict]:
    """Return normalized movie results for a natural-language query."""
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")

    if collection is None:
        collection = get_collection()
    if model is None:
        model = load_embedding_model(local_files_only=True)
    available = collection.count()
    if available == 0:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, available),
        include=["metadatas", "distances"],
    )

    metadatas = raw_results.get("metadatas") or [[]]
    distances = raw_results.get("distances") or [[]]
    return [
        {**metadata, "similarity": 1.0 - distance}
        for metadata, distance in zip(metadatas[0], distances[0])
    ]


def print_results(results: list[dict]) -> None:
    """Print search results in a readable terminal format."""
    if not results:
        print(f"No indexed movies found. Run build_vector_db.py first ({CHROMA_PATH}).")
        return

    for rank, movie in enumerate(results, start=1):
        year = movie.get("release_year", "Unknown")
        genres = movie.get("genres") or "Unknown"
        print(f"{rank}. {movie['title']} ({year})")
        print(f"   Genres: {genres}")
        print(f"   Movie ID: {movie.get('movie_id', '')}")
        print(f"   Similarity: {movie['similarity']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        nargs="*",
        help="Movie description. If omitted, you will be prompted for one.",
    )
    args = parser.parse_args()
    query = " ".join(args.query).strip() or input("Describe a movie: ").strip()

    print_results(search_movies(query, top_k=DEFAULT_RESULT_COUNT))


if __name__ == "__main__":
    main()
