# AILiteracy

A concise, reproducible pipeline for analyzing academic library guides on artificial intelligence using large language models, vector embeddings, and thematic synthesis.

## Overview
This repository implements a three‑stage computational qualitative analysis:
1. **Structured LLM extraction** – parses guide content into a JSON schema covering definitions, outcomes, policies, instructional frameworks, and supporting quotes.
2. **Embedding & similarity** – creates high‑dimensional embeddings (`gemini‑embedding‑2` or `nomic‑embed‑text`) and computes pairwise cosine similarity to cluster institutions.
3. **Thematic synthesis** – feeds the clustered data back to an LLM for cross‑institutional thematic reporting.

For detailed background on pedagogical frameworks and methodology, see the Quarto chapters:
- `chapters/05-frameworks.qmd` – instructional frameworks & evaluation heuristics.
- `methods.qmd` – full methodological pipeline description.

## Installation
```bash
# Install dependencies via uv 
uv sync
```
The project uses Python 3.11+.

## Usage
Run the end‑to‑end analysis with the CLI provided in `analysis.py`:
```bash
# Basic execution with a provider/model
uv run python analysis.py --model <provider>/<model>

# Local Ollama execution (e.g., Qwen2.5)
uv run python analysis.py --model ollama/qwen2.5:3b --embed-provider ollama --embed-model nomic-embed-text

# Benchmark multiple models
uv run python analysis.py --compare
```
Key arguments:
- `--model` – `<provider>/<model>` identifier.
- `--embed-provider` – source for embeddings.
- `--embed-model` – specific embedding model name.
- `--compare` – run a multi‑model benchmark.
- `--force` – refresh extractions, ignoring cached results.

## Extending
To add new model providers, update `config.py` with the service endpoint and modify the routing table in `analysis.py` as described in `methods.qmd`.

## Reproducing Results
All results are generated under the `results/` directory. The pipeline is deterministic given the same model and seed.

## License
This work is released under the MIT License.
