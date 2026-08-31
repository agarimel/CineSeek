"""Ask context-grounded movie questions from the terminal."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from groq_client import generate_answer
from load_movies import load_movies
from search_movies import DEFAULT_RESULT_COUNT, search_movies


INSUFFICIENT_CONTEXT_ANSWER = (
    "I do not have enough information in the retrieved movie context to answer that."
)


def retrieve_movie_context(question: str) -> list[dict]:
    """Retrieve five movies, then attach their full cleaned loader records."""
    print(f"\nChromaDB query: {question}")
    search_results = search_movies(question, top_k=DEFAULT_RESULT_COUNT)

    print("Top 5 retrieved movies:")
    if search_results:
        for rank, result in enumerate(search_results, start=1):
            print(f"{rank}. {result['title']} — similarity {result['similarity']:.3f}")
    else:
        print("No movies retrieved.")

    movies_by_id = {movie["movie_id"]: movie for movie in load_movies()}
    return [
        movies_by_id[result["movie_id"]]
        for result in search_results
        if result.get("movie_id") in movies_by_id
    ]


def format_movie_context(movies: list[dict]) -> str:
    """Format retrieved movies as clear, bounded context for the LLM."""
    sections = []
    for index, movie in enumerate(movies, start=1):
        year = movie.get("release_year") or "Unknown"
        genres = ", ".join(movie.get("genres") or []) or "Unknown"
        sections.append(
            f"Movie {index}\n"
            f"Title: {movie['title']}\n"
            f"Release year: {year}\n"
            f"Genres: {genres}\n"
            f"Overview: {movie['overview']}"
        )
    return "\n\n".join(sections)


def answer_movie_question(
    question: str,
    retrieve: Callable[[str], list[dict]] | None = None,
    ask_llm: Callable[[str, str], str] | None = None,
) -> tuple[str, list[str]]:
    """Retrieve movie context, ask the LLM, and return answer plus sources."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    retrieve = retrieve or retrieve_movie_context
    ask_llm = ask_llm or generate_answer
    movies = retrieve(question)
    if not movies:
        return INSUFFICIENT_CONTEXT_ANSWER, []

    answer = ask_llm(question, format_movie_context(movies))
    sources = list(dict.fromkeys(movie["title"] for movie in movies))
    return answer, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="*",
        help="Movie question or description. If omitted, you will be prompted.",
    )
    args = parser.parse_args()
    question = " ".join(args.question).strip() or input("Ask about movies: ").strip()

    answer, sources = answer_movie_question(question)
    print(f"\nAnswer:\n{answer}")
    print("\nSources:")
    if sources:
        for title in sources:
            print(f"- {title}")
    else:
        print("- No relevant movies retrieved")


if __name__ == "__main__":
    main()
