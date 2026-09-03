"""Tests for the retrieval evaluation dataset and metric calculation."""

from __future__ import annotations

import unittest

from evaluate_retrieval import (
    evaluate_cases,
    find_expected_rank,
    load_evaluation_cases,
)


class FakeCollection:
    def count(self) -> int:
        return 10


class FakeModel:
    pass


class RetrievalEvaluationTests(unittest.TestCase):
    def test_repository_dataset_is_valid_and_has_unique_cases(self) -> None:
        cases = load_evaluation_cases()

        self.assertGreaterEqual(len(cases), 20)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(all(case["expected_movie_id"].isdigit() for case in cases))

    def test_find_expected_rank_uses_movie_id(self) -> None:
        results = [
            {"movie_id": "597", "title": "Titanic"},
            {"movie_id": "16535", "title": "Titanic"},
        ]

        self.assertEqual(find_expected_rank(results, "16535"), 2)
        self.assertIsNone(find_expected_rank(results, "missing"))

    def test_evaluate_cases_calculates_top_1_and_top_k_accuracy(self) -> None:
        cases = [
            {
                "id": "case_one",
                "category": "test",
                "query": "first query",
                "expected_movie_id": "1",
                "expected_title": "First Movie",
            },
            {
                "id": "case_two",
                "category": "test",
                "query": "second query",
                "expected_movie_id": "2",
                "expected_title": "Second Movie",
            },
        ]

        def fake_search(query, **kwargs):
            if query == "first query":
                return [
                    {"movie_id": "1", "title": "First Movie", "similarity": 0.9},
                    {"movie_id": "9", "title": "Other", "similarity": 0.7},
                ]
            return [
                {"movie_id": "9", "title": "Other", "similarity": 0.8},
                {"movie_id": "2", "title": "Second Movie", "similarity": 0.6},
            ]

        from unittest.mock import patch

        with patch("evaluate_retrieval.search_movies", side_effect=fake_search):
            evaluations, summary = evaluate_cases(
                cases,
                top_k=5,
                collection=FakeCollection(),
                model=FakeModel(),
            )

        self.assertEqual([result["rank"] for result in evaluations], [1, 2])
        self.assertEqual(summary["top_1_hits"], 1)
        self.assertEqual(summary["top_k_hits"], 2)
        self.assertEqual(summary["top_1_accuracy"], 0.5)
        self.assertEqual(summary["top_k_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
