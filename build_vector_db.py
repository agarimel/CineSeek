"""Build CineSeek's persistent local Chroma movie index."""

from __future__ import annotations

from load_movies import load_movie_keywords, load_movies
from vector_store import (
    CHROMA_PATH,
    EMBEDDING_MODEL_NAME,
    get_collection,
    load_embedding_model,
    movie_to_metadata,
    movie_to_searchable_text,
)


BATCH_SIZE = 128


def unique_movies_by_id(movies: list[dict]) -> list[dict]:
    """Keep one movie per ID, preserving the loader's popularity order."""
    unique_movies = []
    seen_movie_ids = set()
    for movie in movies:
        movie_id = movie["movie_id"]
        if movie_id in seen_movie_ids:
            continue
        seen_movie_ids.add(movie_id)
        unique_movies.append(movie)
    return unique_movies


def build_vector_db() -> int:
    """Embed the cleaned movies and upsert them into local Chroma storage."""
    cleaned_movies = load_movies()
    movies = unique_movies_by_id(cleaned_movies)
    keywords_by_movie_id = load_movie_keywords()
    movies = [
        {
            **movie,
            "keywords": keywords_by_movie_id.get(movie["movie_id"], []),
        }
        for movie in movies
    ]
    collection = get_collection()
    model = load_embedding_model()

    print(f"Loaded {len(cleaned_movies):,} cleaned movie rows.")
    print(f"Indexing {len(movies):,} unique movie IDs.")
    print(f"Embedding with {EMBEDDING_MODEL_NAME}...")

    for start in range(0, len(movies), BATCH_SIZE):
        batch = movies[start : start + BATCH_SIZE]
        documents = [movie_to_searchable_text(movie) for movie in batch]
        embeddings = model.encode(
            documents,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        collection.upsert(
            ids=[movie["movie_id"] for movie in batch],
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=[movie_to_metadata(movie) for movie in batch],
        )
        print(f"Indexed {start + len(batch):,}/{len(movies):,} movies", end="\r")

    print()
    print(f"Vector database contains {collection.count():,} movies at {CHROMA_PATH}")
    return collection.count()


if __name__ == "__main__":
    build_vector_db()
