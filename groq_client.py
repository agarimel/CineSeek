"""Groq client setup and context-grounded answer generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parent
DEFAULT_MODEL = "openai/gpt-oss-120b"
SYSTEM_PROMPT = """You are CineSeek, a movie question-answering assistant.
Answer the user's question using only the retrieved movie context provided.
Do not use outside knowledge or invent details.
If the retrieved context does not contain enough information to answer, clearly say:
"I do not have enough information in the retrieved movie context to answer that."
Keep the answer concise and mention relevant movie titles when useful."""


def get_groq_client() -> Any:
    """Load GROQ_API_KEY from .env and return an authenticated Groq client."""
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to the project's .env file."
        )
    return Groq(api_key=api_key)


def generate_answer(question: str, context: str, client: Any | None = None) -> str:
    """Ask Groq to answer a question using only the supplied movie context."""
    client = client or get_groq_client()
    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Retrieved movie context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    answer = completion.choices[0].message.content
    return answer.strip() if answer else "No answer was returned by Groq."
