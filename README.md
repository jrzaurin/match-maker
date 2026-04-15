<p align="center">
  <img src="figures/mm_logo.png" alt="Match Maker logo" width="220"/>
</p>

# Match Maker

A two-stage CV–job matching system powered by LLMs and TF-IDF retrieval.

Given a folder of job descriptions and a candidate's CV in PDF format, the system:

1. **Offline** — parses every job spec with an LLM, extracts structured records, and fits a TF-IDF encoder on job titles.
2. **Online** — accepts a CV PDF, extracts the candidate profile, retrieves the top-N most similar jobs by title similarity, and asks an LLM to score and rank them with a written rationale. Optional candidate preferences (salary, remote policy, seniority …) can further reorder the final ranking.

---

## Architecture

```
data/
  job_descriptions/   ← raw .txt job specs (one file per role)
  cvs/                ← CV PDFs

         ┌─────────────────────────────────┐
         │       OFFLINE PIPELINE          │
         │                                 │
         │  1. LLM extraction              │
         │     .txt → JobRecord (JSONL)    │
         │                                 │
         │  2. TF-IDF indexing             │
         │     job titles → vectorizer.pkl │
         └──────────────┬──────────────────┘
                        │ produces
              processed_data/jobs.jsonl
              artifacts/title_vectorizer.pkl

         ┌─────────────────────────────────┐
         │        ONLINE PIPELINE          │
         │                                 │
         │  1. Read CV PDF                 │
         │  2. LLM extraction → CVRecord   │
         │  3. TF-IDF cosine → top-N jobs  │
         │  4. LLM ranking (score+reason)  │
         │  5. Preference reranking        │
         └──────────────┬──────────────────┘
                        │ produces
              results/<stem>_<timestamp>.json
```

## Installation

```bash
# set an environment with your tool of choice and...
pip install -e ".[dev]"
```

Set your OpenAI key:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Offline pipeline

Run this once whenever you add or change job descriptions.

```
data/job_descriptions/
    acme_ml_engineer.txt
    bigco_data_scientist.txt
    ...
```

### Step 1 + 2 together (default)

```bash
python -m matchmaker.offline_pipeline.pipeline
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt-4o-mini` | LLM to use for extraction |
| `--api-key` | env `OPENAI_API_KEY` | Raw key string or path to a key file |
| `--n-workers` | `3` | Concurrent LLM calls |
| `--index-only` | `False` | Skip LLM extraction; only refit the TF-IDF encoder |

```bash
# Use a local Ollama model
python -m matchmaker.offline_pipeline.pipeline --model qwen3.5:9b

# Key from a file
python -m matchmaker.offline_pipeline.pipeline --api-key ~/.openai/key.txt

# Only refit the title encoder (jobs.jsonl already exists) NOTE: not tested
python -m matchmaker.offline_pipeline.pipeline --index-only
```

Outputs written to:

```
processed_data/jobs.jsonl           ← structured JobRecords
artifacts/title_vectorizer.pkl      ← fitted TF-IDF encoder
```

---

## Online pipeline

### CLI

```bash
python -m matchmaker.online_pipeline.pipeline path/to/cv.pdf
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt-4o-mini` | LLM for extraction and ranking |
| `--api-key` | env `OPENAI_API_KEY` | Raw key string or path to a key file |
| `--n-workers` | `3` | Concurrent LLM calls |
| `--role-type` | — | Filter by role type (repeatable) |
| `--remote` | `no_preference` | `remote` / `hybrid` / `onsite` / `no_preference` |
| `--min-salary` | — | Minimum acceptable salary (GBP) |
| `--seniority` | — | Preferred seniority levels (repeatable) |
| `--company-stage` | — | Preferred company stages (repeatable) |
| `--company-size` | — | Preferred company sizes (repeatable) |
| `--location` | — | Preferred locations — partial match (repeatable) |
| `--no-print` | `False` | Suppress terminal output; just save to `results/` |

```bash
# Plain ranking
python -m matchmaker.online_pipeline.pipeline cv.pdf

# With preferences
python -m matchmaker.online_pipeline.pipeline cv.pdf \
  --model gpt-4o-mini \
  --role-type ml_engineer \
  --remote hybrid \
  --min-salary 100000 \
  --seniority senior --seniority lead \
  --company-stage scale_up \
  --location London

# Local Ollama model, no terminal output
python -m matchmaker.online_pipeline.pipeline cv.pdf \
  --model qwen3.5:9b \
  --no-print
```

Result saved to `results/<cv_stem>_<timestamp>.json`:

```json
{
  "ranked_jobs": [
    {
      "job_id": "xantura_ml_engineer",
      "score": 92.0,
      "rank": 1,
      "rationale": "Strong alignment on ML leadership and general AI experience ..."
    },
    ...
  ],
  "summary": {
    "best_fit_job_id": "xantura_ml_engineer",
    "overall_comment": "Best suited for senior ML roles with a leadership component ..."
  }
}
```

**IMPORTANT NOTE**: at the moment there is nothing implemented regarding evals
for the LLM output or the evaluation of the overall pipeline via implicit or
explicit feedback from the user. This would be in itself the most important next
step to take.

### HTTP API (local)

```bash
uvicorn api:app --reload
```

```bash
# Minimal
curl -X POST http://localhost:8000/match \
  -F "cv=@cv.pdf"

# With preferences
curl -X POST http://localhost:8000/match \
  -F "cv=@cv.pdf" \
  -F "model=gpt-4o-mini" \
  -F "remote=hybrid" \
  -F "min_salary=90000" \
  -F "role_type=ml_engineer" \
  -F "location=London"
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Docker

Create a `.env` file at the project root or set yourself the OPENAI_API_KEY:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### Step 1 — run the ingest container

```bash
docker compose run --rm ingest
```

```bash
# Ollama model (Ollama must be reachable from inside the container)
docker compose run --rm ingest --model qwen3.5:9b

# Skip LLM extraction, refit vectorizer only
docker compose run --rm ingest --index-only

# Custom worker count
docker compose run --rm ingest --n-workers 5
```

The `ingest` service uses the `ingest` Docker Compose profile, so it does **not** start automatically with `docker compose up`.

### Step 2 — start the API

```bash
docker compose up api

# or to rebuild after a code change:
docker compose up --build api
```

### Step 3 — send requests

```bash
# Basic match
curl -X POST http://localhost:8000/match \
  -F "cv=@/path/to/cv.pdf"

# With preferences
curl -X POST http://localhost:8000/match \
  -F "cv=@/path/to/cv.pdf" \
  -F "remote=hybrid" \
  -F "min_salary=90000" \
  -F "role_type=ml_engineer" \
  -F "seniority=senior" \
  -F "location=London"
```

---

## Tests

```bash
# maybe set PYTHONPATH and then just run
pytest
```
