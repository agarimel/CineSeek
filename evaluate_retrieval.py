"""Evaluate CineSeek semantic retrieval against labeled movie queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from search_movies import search_movies
from vector_store import get_collection, load_embedding_model


DEFAULT_DATASET_PATH = Path(__file__).parent / "evaluation_queries.json"
REQUIRED_CASE_FIELDS = {
    "id",
    "category",
    "query",
    "expected_movie_id",
    "expected_title",
}


def load_evaluation_cases(path: Path = DEFAULT_DATASET_PATH) -> list[dict]:
    """Load and validate labeled retrieval cases from JSON."""
    with path.open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)

    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must contain a non-empty 'cases' list.")

    seen_case_ids = set()
    for case in cases:
        missing = REQUIRED_CASE_FIELDS - case.keys()
        if missing:
            raise ValueError(
                f"Evaluation case is missing fields: {', '.join(sorted(missing))}"
            )
        if case["id"] in seen_case_ids:
            raise ValueError(f"Duplicate evaluation case ID: {case['id']}")
        if not all(str(case[field]).strip() for field in REQUIRED_CASE_FIELDS):
            raise ValueError(f"Evaluation case contains an empty field: {case['id']}")
        seen_case_ids.add(case["id"])

    return cases


def find_expected_rank(results: list[dict], expected_movie_id: str) -> int | None:
    """Return the expected movie's one-based rank, or None when absent."""
    return next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if str(result.get("movie_id")) == expected_movie_id
        ),
        None,
    )


def evaluate_cases(
    cases: list[dict],
    *,
    top_k: int = 5,
    collection: Any | None = None,
    model: Any | None = None,
    show_progress: bool = False,
) -> tuple[list[dict], dict]:
    """Run labeled cases and return per-case results plus summary metrics."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if collection is None:
        collection = get_collection()
    if model is None:
        model = load_embedding_model(local_files_only=True)

    evaluations = []
    for case_number, case in enumerate(cases, start=1):
        if show_progress:
            print(
                f"Evaluating {case_number}/{len(cases)}: {case['id']}",
                flush=True,
            )
        results = search_movies(
            case["query"],
            top_k=top_k,
            collection=collection,
            model=model,
        )
        rank = find_expected_rank(results, case["expected_movie_id"])
        expected_result = results[rank - 1] if rank is not None else None
        evaluations.append(
            {
                **case,
                "rank": rank,
                "similarity": (
                    expected_result.get("similarity")
                    if expected_result is not None
                    else None
                ),
                "retrieved_titles": [result["title"] for result in results],
            }
        )

    total = len(evaluations)
    top_1_hits = sum(evaluation["rank"] == 1 for evaluation in evaluations)
    top_k_hits = sum(evaluation["rank"] is not None for evaluation in evaluations)
    summary = {
        "total": total,
        "top_1_hits": top_1_hits,
        "top_k_hits": top_k_hits,
        "top_1_accuracy": top_1_hits / total if total else 0.0,
        "top_k_accuracy": top_k_hits / total if total else 0.0,
        "top_k": top_k,
    }
    return evaluations, summary


def print_report(evaluations: list[dict], summary: dict) -> None:
    """Print individual retrieval outcomes and aggregate accuracy."""
    for evaluation in evaluations:
        rank = evaluation["rank"] or "miss"
        similarity = evaluation["similarity"]
        score = f"{similarity:.3f}" if similarity is not None else "n/a"
        print(
            f"[{evaluation['id']}] expected={evaluation['expected_title']!r} "
            f"rank={rank} similarity={score}"
        )
        if evaluation["rank"] is None:
            print(f"  retrieved: {', '.join(evaluation['retrieved_titles'])}")

    print("\nRetrieval summary")
    print(f"Cases: {summary['total']}")
    print(
        f"Top-1 accuracy: {summary['top_1_hits']}/{summary['total']} "
        f"({summary['top_1_accuracy']:.1%})"
    )
    print(
        f"Top-{summary['top_k']} accuracy: "
        f"{summary['top_k_hits']}/{summary['total']} "
        f"({summary['top_k_accuracy']:.1%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the labeled evaluation JSON file.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading evaluation cases from {args.dataset}...", flush=True)
    cases = load_evaluation_cases(args.dataset)
    print("Opening the local ChromaDB collection...", flush=True)
    collection = get_collection()
    print("Loading the local embedding model...", flush=True)
    model = load_embedding_model(local_files_only=True)
    evaluations, summary = evaluate_cases(
        cases,
        top_k=args.top_k,
        collection=collection,
        model=model,
        show_progress=True,
    )
    print_report(evaluations, summary)


if __name__ == "__main__":
    main()
