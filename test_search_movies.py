"""Small unit tests for movie text creation and retrieval behavior."""

from __future__ import annotations

import unittest

from search_movies import search_movies
from vector_store import movie_to_metadata, movie_to_searchable_text


class FakeEmbedding(list):
    def tolist(self) -> list[float]:
        return list(self)


class FakeModel:
    def __init__(self) -> None:
        self.encoded_texts = None

    def encode(self, texts, **kwargs):
        self.encoded_texts = texts
        return [FakeEmbedding([0.1, 0.2, 0.3])]


class FakeCollection:
    def count(self) -> int:
        return 2

    def query(self, **kwargs):
        self.query_arguments = kwargs
        return {
            "metadatas": [[
                {"movie_id": "862", "title": "Toy Story", "release_year": 1995,
                 "genres": "Animation, Comedy, Family"},
                {"movie_id": "8844", "title": "Jumanji", "release_year": 1995,
                 "genres": "Adventure, Fantasy, Family"},
            ]],
            "distances": [[0.1, 0.25]],
        }


class MovieVectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.movie = {
            "movie_id": "862",
            "title": "Toy Story",
            "overview": "Toys come to life and go on an adventure.",
            "release_year": 1995,
            "genres": ["Animation", "Comedy", "Family"],
            "keywords": ["toy", "friendship", "toy comes to life"],
        }

    def test_searchable_text_contains_all_requested_fields(self) -> None:
        text = movie_to_searchable_text(self.movie)
        self.assertIn("Toy Story", text)
        self.assertIn("Toys come to life", text)
        self.assertIn("1995", text)
        self.assertIn("Animation, Comedy, Family", text)
        self.assertIn("Keywords: toy, friendship, toy comes to life", text)

    def test_metadata_contains_requested_fields(self) -> None:
        self.assertEqual(
            movie_to_metadata(self.movie),
            {
                "movie_id": "862",
                "title": "Toy Story",
                "release_year": 1995,
                "genres": "Animation, Comedy, Family",
            },
        )

    def test_search_embeds_query_and_returns_ranked_metadata(self) -> None:
        model = FakeModel()
        collection = FakeCollection()

        results = search_movies(
            "animated toys",
            top_k=5,
            collection=collection,
            model=model,
        )

        self.assertEqual(model.encoded_texts, ["animated toys"])
        self.assertEqual(collection.query_arguments["n_results"], 2)
        self.assertEqual(results[0]["title"], "Toy Story")
        self.assertAlmostEqual(results[0]["similarity"], 0.9)
        self.assertAlmostEqual(results[1]["similarity"], 0.75)

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            search_movies(
                "   ",
                collection=FakeCollection(),
                model=FakeModel(),
            )

    def test_dream_implant_query_retrieves_inception(self) -> None:
        query = "people enter other people's dreams to steal and implant ideas"
        model = FakeModel()
        collection = FakeCollection()
        collection.query = lambda **kwargs: {
            "metadatas": [[{
                "movie_id": "27205",
                "title": "Inception",
                "release_year": 2010,
                "genres": "Action, Science Fiction",
            }]],
            "distances": [[0.2]],
        }

        results = search_movies(
            query,
            top_k=5,
            collection=collection,
            model=model,
        )

        self.assertEqual(model.encoded_texts, [query])
        self.assertEqual(results[0]["title"], "Inception")


if __name__ == "__main__":
    unittest.main()
