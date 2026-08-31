"""Tests for full-corpus vector database preparation."""

from __future__ import annotations

import unittest

from build_vector_db import unique_movies_by_id


class BuildVectorDbTests(unittest.TestCase):
    def test_duplicate_movie_ids_are_removed_in_loader_order(self) -> None:
        movies = [
            {"movie_id": "10", "title": "Higher popularity version"},
            {"movie_id": "20", "title": "Another movie"},
            {"movie_id": "10", "title": "Duplicate version"},
        ]

        unique_movies = unique_movies_by_id(movies)

        self.assertEqual(
            [movie["movie_id"] for movie in unique_movies],
            ["10", "20"],
        )
        self.assertEqual(unique_movies[0]["title"], "Higher popularity version")


if __name__ == "__main__":
    unittest.main()
