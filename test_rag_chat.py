"""Tests for RAG orchestration without real Groq API calls."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from groq_client import SYSTEM_PROMPT, generate_answer
from rag_chat import (
    INSUFFICIENT_CONTEXT_ANSWER,
    answer_movie_question,
    format_movie_context,
    retrieve_movie_context,
)


MOVIES = [
    {
        "movie_id": "27205",
        "title": "Inception",
        "overview": "A thief enters dreams and is asked to implant an idea.",
        "release_date": "2010-07-14",
        "release_year": 2010,
        "genres": ["Action", "Science Fiction"],
    },
    {
        "movie_id": "157336",
        "title": "Interstellar",
        "overview": "Explorers travel through a wormhole in space.",
        "release_date": "2014-11-05",
        "release_year": 2014,
        "genres": ["Adventure", "Drama", "Science Fiction"],
    },
]


class FakeMessage:
    content = "The retrieved context points to Inception."


class FakeCompletion:
    choices = [type("Choice", (), {"message": FakeMessage()})()]


class FakeCompletions:
    def create(self, **kwargs):
        self.arguments = kwargs
        return FakeCompletion()


class FakeGroqClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


class RagChatTests(unittest.TestCase):
    def test_context_contains_movie_details(self) -> None:
        context = format_movie_context(MOVIES)
        self.assertIn("Title: Inception", context)
        self.assertIn("Release year: 2010", context)
        self.assertIn("Action, Science Fiction", context)
        self.assertIn("A thief enters dreams", context)

    def test_rag_uses_retrieval_and_mocked_llm(self) -> None:
        captured = {}

        def fake_retrieve(question):
            captured["retrieval_question"] = question
            return MOVIES

        def fake_llm(question, context):
            captured["llm_question"] = question
            captured["context"] = context
            return "Inception best matches that description."

        answer, sources = answer_movie_question(
            "Which movie involves entering dreams?",
            retrieve=fake_retrieve,
            ask_llm=fake_llm,
        )

        self.assertEqual(captured["retrieval_question"], captured["llm_question"])
        self.assertIn("Overview:", captured["context"])
        self.assertEqual(answer, "Inception best matches that description.")
        self.assertEqual(sources, ["Inception", "Interstellar"])

    @patch("rag_chat.load_movies", return_value=MOVIES)
    @patch("rag_chat.search_movies")
    def test_rag_passes_exact_question_to_shared_search(
        self,
        mocked_search,
        mocked_load_movies,
    ) -> None:
        question = "What movie involves entering dreams to implant an idea?"
        mocked_search.return_value = [
            {
                "movie_id": "27205",
                "title": "Inception",
                "release_year": 2010,
                "genres": "Action, Science Fiction",
                "similarity": 0.8,
            }
        ]

        output = StringIO()
        with redirect_stdout(output):
            movies = retrieve_movie_context(question)

        mocked_search.assert_called_once_with(question, top_k=5)
        self.assertEqual(movies[0]["title"], "Inception")
        self.assertIn(f"ChromaDB query: {question}", output.getvalue())
        self.assertIn("Inception — similarity 0.800", output.getvalue())

    def test_no_context_returns_clear_message_without_llm_call(self) -> None:
        def unexpected_llm(question, context):
            self.fail("The LLM must not be called without retrieved context")

        answer, sources = answer_movie_question(
            "unknown",
            retrieve=lambda question: [],
            ask_llm=unexpected_llm,
        )

        self.assertEqual(answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertEqual(sources, [])

    def test_groq_call_is_grounded_by_system_prompt(self) -> None:
        client = FakeGroqClient()
        answer = generate_answer("What happens?", "Title: Inception", client=client)
        arguments = client.chat.completions.arguments

        self.assertEqual(answer, "The retrieved context points to Inception.")
        self.assertIn("only", SYSTEM_PROMPT.lower())
        self.assertIn("not have enough information", SYSTEM_PROMPT)
        self.assertIn("Title: Inception", arguments["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
