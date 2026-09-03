# CineSeek

CineSeek is a command-line movie discovery and question-answering application built to explore Retrieval-Augmented Generation (RAG) and, eventually, agentic AI. It combines a local movie dataset, local text embeddings, ChromaDB retrieval, and a Groq-hosted language model to identify movies from natural-language descriptions and answer questions using retrieved movie records.

The repository currently implements a fixed RAG pipeline. It does **not** yet implement an autonomous or tool-using agent.

## Current Capabilities

CineSeek currently supports:

- Loading and cleaning movie metadata from `movies/movies_metadata.csv`.
- Removing records without a usable title or overview.
- Parsing release dates, release years, genres, IDs, and popularity values.
- Joining `movies/keywords.csv` to movie metadata through the shared movie ID.
- Deduplicating records by movie ID before indexing.
- Indexing approximately **44,471 unique movies** from 44,501 valid cleaned rows.
- Creating embeddings locally with `sentence-transformers/all-MiniLM-L6-v2`; no embedding API key is required.
- Persisting embeddings and metadata in a local ChromaDB collection.
- Finding movies from natural-language descriptions using cosine semantic similarity.
- Retrieving five movie records and using them as context for question answering.
- Generating context-grounded answers through Groq.
- Printing retrieved similarity scores and movie titles used as sources.
- Returning an insufficient-context message when no movie records are available, while also instructing the LLM to acknowledge when retrieved context cannot answer the question.
- Running unit tests without contacting Groq by using mocked LLM behavior.

## Current Architecture

```text
User query
    │
    ▼
Sentence Transformers embedding
    │
    ▼
Local ChromaDB cosine search
    │
    ▼
Top 5 movie IDs and metadata
    │
    ▼
Full cleaned movie records used as context
    │
    ▼
Groq chat completion
    │
    ▼
Grounded answer + retrieved source titles
```

There are two user-facing command-line paths:

- `search_movies.py` performs semantic retrieval and prints ranked matches.
- `rag_chat.py` uses the same search function, enriches the results with full movie records, and asks Groq to answer from that context.

## How RAG Works in CineSeek

RAG consists of retrieval, augmentation, and generation.

### Retrieval

`search_movies.py` embeds the user's query with the same local Sentence Transformers model used during indexing. It queries the persisted ChromaDB collection and returns the five closest movies with similarity scores.

Each indexed document is constructed in `vector_store.py` from the movie's:

- title,
- overview,
- release year,
- genres, and
- keywords.

### Augmentation

`rag_chat.py` matches the retrieved movie IDs back to the cleaned records produced by `load_movies.py`. It formats each selected movie's title, release year, genres, and overview into a bounded context block. The user's original question and this context are then supplied to the language model.

### Generation

`groq_client.py` sends the augmented prompt to Groq. Its system prompt requires the model to use only the retrieved context, avoid inventing details, and state when the available context is insufficient. The terminal output includes the generated answer and the titles of the retrieved movies used as sources.

## Project Structure

| File or directory | Responsibility |
|---|---|
| `movies/movies_metadata.csv` | Main source of movie titles, overviews, dates, genres, IDs, and popularity values. |
| `movies/keywords.csv` | Keyword records joined to metadata through the movie ID. |
| `load_movies.py` | Parses and cleans metadata, sorts valid records by popularity, and loads keywords by movie ID. |
| `vector_store.py` | Defines ChromaDB/model configuration and formats searchable documents and stored metadata. |
| `build_vector_db.py` | Deduplicates movie IDs, joins keywords, creates embeddings in batches, and persists the ChromaDB index. |
| `search_movies.py` | Provides the shared semantic-search function and standalone search CLI. |
| `rag_chat.py` | Retrieves five movies, constructs LLM context, calls the grounded-answer function, and prints sources. |
| `groq_client.py` | Loads Groq configuration from `.env` and performs the chat-completion request. |
| `test_load_movies.py` | Tests cleaning, ordering, keyword parsing, and keyword-ID joining. |
| `test_build_vector_db.py` | Tests movie-ID deduplication. |
| `test_search_movies.py` | Tests searchable text, Chroma query handling, result normalization, and retrieval behavior. |
| `test_rag_chat.py` | Tests RAG orchestration and grounding with mocked retrieval and Groq calls. |
| `evaluation_queries.json` | Hand-labeled movie-description queries and expected movie IDs for retrieval evaluation. |
| `evaluate_retrieval.py` | Runs the labeled benchmark and reports top-1 and top-k retrieval accuracy. |
| `test_evaluate_retrieval.py` | Tests evaluation-data validation and metric calculation without running the embedding model. |
| `requirements.txt` | Lists the current Python dependencies. |
| `.env.example` | Provides a safe template for local Groq configuration. |
| `~/.cineseek/chroma_db/` | Generated local vector database, stored outside the repository and created by the build script. |

Other CSV files are present in `movies/`, but the current pipeline uses `movies_metadata.csv` and `keywords.csv` for indexing and retrieval.

## Technology Stack

- **Python** — data processing, indexing, retrieval, RAG orchestration, and tests.
- **Sentence Transformers** — local `all-MiniLM-L6-v2` document and query embeddings.
- **ChromaDB** — persistent local vector storage and cosine similarity search.
- **Groq Python SDK** — hosted chat-completion inference for grounded answers.
- **python-dotenv** — local environment-variable loading from `.env`.
- **Movie metadata CSV dataset** — movie metadata and keyword source files.
- **Git and GitHub** — source control and repository hosting.
- **`unittest`** — standard-library test framework.

## Running CineSeek Locally

### Prerequisites

- Python 3.12 or another version supported by the dependencies.
- The `movies/` directory containing `movies_metadata.csv` and `keywords.csv`.
- A Groq API key for `rag_chat.py`. Semantic search does not require a Groq key.

The raw CSV dataset and generated ChromaDB index are intentionally excluded from Git because of their size. Place `movies_metadata.csv` and `keywords.csv` under `movies/` before building the index. By default, the index is stored at `~/.cineseek/chroma_db`; set `CINESEEK_CHROMA_PATH` in the shell to override this location.

### 1. Create and activate a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure Groq

Copy the example configuration:

```bash
cp .env.example .env
```

Then edit `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

The optional `GROQ_MODEL` variable overrides the default model configured in `groq_client.py`. Never commit `.env`; it is excluded by `.gitignore`.

### 4. Build the local vector database

```bash
python build_vector_db.py
```

The first build downloads the embedding model. The script then embeds movies in batches, prints progress, and persists the collection under `~/.cineseek/chroma_db/`. Rebuild the database whenever the dataset or indexed text format changes.

To perform a clean rebuild:

```bash
rm -rf ~/.cineseek/chroma_db
python build_vector_db.py
```

### 5. Run semantic search

Interactive mode:

```bash
python search_movies.py
```

Direct query:

```bash
python search_movies.py "people enter other people's dreams to steal and implant ideas"
```

### 6. Run RAG question answering

Interactive mode:

```bash
python rag_chat.py
```

Direct question:

```bash
python rag_chat.py "What is Interstellar about?"
```

### 7. Run the tests

```bash
python -m unittest -v
```

The RAG tests mock the LLM call and do not consume Groq API credits.

### 8. Evaluate semantic retrieval

After building `chroma_db`, run the labeled retrieval benchmark:

```bash
python evaluate_retrieval.py
```

The report shows the expected movie's rank and similarity for every query, followed by top-1 and top-5 accuracy. To evaluate a different result depth:

```bash
python evaluate_retrieval.py --top-k 10
```

On macOS systems that synchronize Desktop through iCloud, keep the virtual environment outside the project directory to prevent package files from being offloaded:

```bash
python3 -m venv /Users/your-name/.venvs/cineseek
source /Users/your-name/.venvs/cineseek/bin/activate
python -m pip install -r requirements.txt
```

## Example Queries

Identify *Inception* from its premise:

```bash
python search_movies.py "people enter other people's dreams to steal and implant ideas"
```

Identify *Groundhog Day* from a repeated-day description:

```bash
python search_movies.py "a TV weatherman becomes trapped in a time loop and repeats the same day"
```

Identify *Interstellar* from its space and family themes:

```bash
python search_movies.py "astronauts travel through a wormhole while a father tries to save humanity and return to his family"
```

Ask about a known film with grounded generation:

```bash
python rag_chat.py "What is The Lion King about?"
```

Retrieval is probabilistic, so rankings can vary with query wording and library/model versions.

## Current Limitations

CineSeek is a **fixed RAG pipeline**, not an autonomous agent.

- Every RAG request follows essentially the same retrieval → context construction → generation path.
- Semantic retrieval is not always ideal for exact titles or structured constraints such as genre and year combinations.
- There is no intent classification or tool selection.
- There are no exact-title, fuzzy-title, metadata-filter, comparison, or details tools.
- Retrieval does not use a similarity threshold to reject weak results.
- The system does not evaluate evidence quality or retry with a reformulated query.
- It does not ask clarification questions when a request is ambiguous.
- There is no conversation memory or session state; each invocation is independent.
- Source reporting lists retrieved titles, not claim-level citations proving which record supports each sentence.
- Keywords improve retrieval embeddings but are not currently included in the context sent to Groq.
- There is no personalized recommendation model or user-preference profile.
- There is no FastAPI service, frontend, account system, or deployment configuration.

## Roadmap: From RAG Application to RAG Agent

The phases below describe planned work, not current functionality.

### Phase 1 — Retrieval evaluation ✅

- Create a labeled evaluation dataset of representative queries and expected movies.
- Measure top-1 and top-5 retrieval accuracy.
- Establish similarity thresholds for strong, weak, and insufficient evidence.
- Add structured retrieval logging for queries, ranks, scores, and outcomes.

### Phase 2 — Hybrid retrieval

- Add exact and fuzzy title lookup.
- Retain semantic search for plot and theme descriptions.
- Add genre and year filters.
- Support structured metadata queries and combine their results with vector search.

### Phase 3 — Agent tools

Expose deterministic, independently tested capabilities such as:

```text
semantic_movie_search
find_movie_by_title
filter_movies
get_movie_details
```

Each tool should use a validated input schema and return structured results.

### Phase 4 — Agent routing

Add an LLM-powered structured router that identifies user intent and selects an appropriate tool. Example intents include:

- identify a movie,
- ask about a known movie,
- filter movies,
- compare movies, and
- clarify an unclear request.

Tool names and arguments should be validated before execution.

### Phase 5 — Evaluation and retry

Add a controlled evidence-evaluation step that can:

- answer when evidence is strong,
- reformulate and retry retrieval once when evidence is weak,
- ask a clarification question when appropriate, and
- refuse to invent information when evidence remains insufficient.

Retries should be bounded and observable.

### Phase 6 — Conversation state

Store the minimum state needed for follow-up questions, including recent turns and selected movie IDs. This would support requests such as “Which of those was released first?” without rerunning an unrelated search.

### Phase 7 — Application layer

Once agent behavior is evaluated and reliable:

- expose the system through FastAPI,
- build a simple frontend,
- add sessions and error handling, and
- deploy the application with appropriate secret management and monitoring.

## Target Agent Architecture

The following diagram is the intended future design; it is **not implemented yet**.

```text
User
  │
  ▼
Agent / structured intent router
  │
  ├── semantic_movie_search
  ├── find_movie_by_title
  ├── filter_movies
  └── get_movie_details
          │
          ▼
   Inspect retrieved evidence
          │
          ├── strong evidence ──────────► grounded response
          ├── weak evidence ────────────► one bounded retry
          └── ambiguous request ────────► clarification question
```

## Project Purpose

CineSeek is a learning and portfolio project intended to demonstrate practical development with:

- Retrieval-Augmented Generation,
- local text embeddings,
- vector databases,
- hosted LLM integration,
- grounded prompting and source reporting,
- retrieval evaluation,
- tool-based agent design, and
- production-oriented AI application development.

The repository deliberately separates implemented RAG behavior from its future agent architecture so that later capabilities can be added and evaluated incrementally.
