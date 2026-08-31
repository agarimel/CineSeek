"""Tests for cleaning and popularity-based movie selection."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from load_movies import load_movie_keywords, load_movies, parse_keywords


class LoadMoviesTests(unittest.TestCase):
    def write_movies_csv(self, rows: list[dict]) -> Path:
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            delete=False,
        )
        self.addCleanup(Path(temporary_file.name).unlink, missing_ok=True)
        with temporary_file:
            writer = csv.DictWriter(
                temporary_file,
                fieldnames=[
                    "id",
                    "title",
                    "overview",
                    "release_date",
                    "genres",
                    "popularity",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return Path(temporary_file.name)

    def test_returns_all_movies_ordered_by_numeric_popularity(self) -> None:
        csv_path = self.write_movies_csv(
            [
                {"id": "1", "title": "First", "overview": "Plot one", "popularity": "2.5"},
                {"id": "2", "title": "Second", "overview": "Plot two", "popularity": "100.1"},
                {"id": "3", "title": "Third", "overview": "Plot three", "popularity": "12.0"},
            ]
        )

        movies = load_movies(csv_path=csv_path)

        self.assertEqual(
            [movie["title"] for movie in movies],
            ["Second", "Third", "First"],
        )
        self.assertNotIn("popularity", movies[0])

    def test_keeps_existing_title_and_overview_cleaning(self) -> None:
        csv_path = self.write_movies_csv(
            [
                {"id": "1", "title": "", "overview": "Has a plot", "popularity": "999"},
                {"id": "2", "title": "No Plot", "overview": "   ", "popularity": "998"},
                {"id": "3", "title": "Valid", "overview": "A real plot", "popularity": "1"},
            ]
        )

        movies = load_movies(csv_path=csv_path)

        self.assertEqual([movie["title"] for movie in movies], ["Valid"])

    def test_invalid_popularity_is_ranked_as_zero(self) -> None:
        csv_path = self.write_movies_csv(
            [
                {"id": "1", "title": "Unknown", "overview": "Plot", "popularity": "invalid"},
                {"id": "2", "title": "Known", "overview": "Plot", "popularity": "0.5"},
            ]
        )

        movies = load_movies(csv_path=csv_path)

        self.assertEqual([movie["title"] for movie in movies], ["Known", "Unknown"])

    def test_loads_keywords_keyed_by_metadata_movie_id(self) -> None:
        temporary_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv",
            delete=False,
        )
        self.addCleanup(Path(temporary_file.name).unlink, missing_ok=True)
        with temporary_file:
            writer = csv.DictWriter(temporary_file, fieldnames=["id", "keywords"])
            writer.writeheader()
            writer.writerow(
                {
                    "id": "157336",
                    "keywords": "[{'id': 3417, 'name': 'wormhole'}, "
                    "{'id': 3801, 'name': 'space travel'}]",
                }
            )

        keywords = load_movie_keywords(Path(temporary_file.name))

        self.assertEqual(keywords["157336"], ["wormhole", "space travel"])

    def test_invalid_keywords_are_empty(self) -> None:
        self.assertEqual(parse_keywords("not serialized keyword data"), [])


if __name__ == "__main__":
    unittest.main()
