# EviMem: Evidence-Gap-Driven Iterative Retrieval for Long-Term Conversational Memory

This is the official repository for the paper:

> **EviMem: Evidence-Gap-Driven Iterative Retrieval for Long-Term Conversational Memory**
>
> Yuyang Li\*, Yime He\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/), Dong Gong†
>
> \*Equal contribution. †Corresponding author.
>
> ### [Paper](paper.pdf)


## Overview

EviMem is a long-term conversational memory framework designed to retrieve evidence scattered across multiple sessions. Single-pass retrieval fails on temporal and multi-hop questions, and existing iterative methods refine queries via generated content or document-level signals without explicitly diagnosing what evidence is *missing* from the accumulated retrieval set. EviMem closes the retrieval loop via explicit *evidence-gap diagnosis* and combines two mutually enabling components:

- **IRIS** (Iterative Retrieval via Insufficiency Signals): a closed-loop framework that detects evidence gaps through three-tier sufficiency evaluation (`EXACT` / `INFERRABLE` / `PARTIAL`), produces a natural-language diagnosis of what information remains missing, and drives targeted query refinement.
- **LaceMem** (Layered Architecture for Conversational Evidence Memory): a coarse-to-fine memory hierarchy of three layers — Index (atomic semantic tuples for search), Edge (graph links for multi-hop expansion), and Raw (verbatim dialogue for grounding).

On the LoCoMo benchmark, EviMem improves Judge Accuracy over the multi-agent baseline MIRIX on temporal (73.3% → 81.6%) and multi-hop (65.9% → 85.2%) questions, while running 4.5× faster.


## What is in this repository

- `src/memory_retrieval_system/IRIS_retriever_v5.py`: the paper implementation of IRIS, including the dual-path iterative loop, three-tier sufficiency evaluation, calibrated confidence, entity tracking, and "not mentioned" abstention.
- `src/memory_retrieval_system/IRIS_runner_v5.py`: CLI runner that evaluates IRIS on a LoCoMo conversation and writes per-question results to JSON.
- `src/memory_retrieval_system/models.py`: SQLAlchemy schema for LaceMem's three-layer memory (Raw / Index / Edge).
- `src/memory_retrieval_system/retriever.py`: base retrieval primitives — vector search over Index, graph expansion over Edge, raw text fetching.
- `src/memory_retrieval_system/precompute_embeddings.py`: offline embedding precomputation for the Index layer.


## Method Summary

EviMem combines two mutually enabling components:

- **IRIS** runs a closed-loop iterative process. At each step it evaluates whether accumulated evidence is sufficient, classifies it into one of three tiers (`EXACT` / `INFERRABLE` / `PARTIAL`), diagnoses what is missing, and uses the diagnosis to refine the next query. The loop terminates when evidence is sufficient or the iteration budget is exhausted; otherwise the system abstains.

- **LaceMem** organises dialogue into a coarse-to-fine three-layer hierarchy: an *Index layer* of atomic semantic tuples for fine-grained search, an *Edge layer* of graph links for multi-hop expansion, and a *Raw layer* of verbatim dialogue for grounded generation. The atomic granularity is what lets IRIS detect *which specific fact* is missing.

See the paper for full details on confidence calibration, entity tracking, and rule-based refinement strategies.


## Setup

### 1. Create the environment

```bash
conda create -n evimem python=3.10
conda activate evimem
pip install -r requirements.txt
```

### 2. Configure the OpenAI API key

```bash
export OPENAI_API_KEY=sk-your-key-here
```

Or pass it explicitly to each script with `--api-key sk-...`. The Google AI provider uses `GEMINI_API_KEY` analogously.

### 3. Prepare the LoCoMo benchmark

Download LoCoMo from its [official repository](https://github.com/snap-research/locomo) and place `locomo10.json` at `locomo/data/locomo10.json` (or specify an alternate path with `--locomo-path`).

### 4. Build the LaceMem memory database

The runner expects a SQLite database containing the LaceMem layers (`raw_memory`, `memory_index`, `memory_index_edge`) for each conversation. Database construction factorises LoCoMo dialogue turns into atomic Index tuples via an LLM extractor; embedding precomputation for the Index layer is handled by `precompute_embeddings.py`.

By default the runner expects databases at `cache/mirix_dbs/{sample_id}.sqlite.db`.


## Usage

### Run IRIS on a LoCoMo conversation

```bash
python src/memory_retrieval_system/IRIS_runner_v5.py \
    --db cache/mirix_dbs/conv26.sqlite.db \
    --locomo \
    --indices conv-26 \
    --category 2 \
    --model gpt-4o \
    --provider openai
```

Key arguments:

- `--db`: path to the LaceMem SQLite database. Use `{sample_id}` as a placeholder when batching across multiple conversations.
- `--indices`: comma-separated sample IDs (e.g., `conv-26,conv-30`).
- `--category`: question category (`1`=single-hop, `2`=multi-hop, `3`=temporal, `4`=open-domain, `5`=adversarial). Omit to evaluate all.
- `--model`: LLM for answer generation (default `gpt-4o`).
- `--provider`: `openai` or `google_ai`.
- `--max-iterations`: maximum IRIS iterations (default `3`).

Ablation flags (disable individual components):

- `--no-temporal-inference`: disable INFERRABLE temporal reasoning.
- `--no-entity-tracking`: disable per-entity fact tracking.
- `--no-dual-retrieval`: disable the anchor-path retrieval (refinement path only).

Output JSON files are written to `--output-dir` (default `results/memory_retrieval/irr_v5/`).

## Inputs and Outputs

### Inputs

- **LoCoMo benchmark**: long-term conversational memory benchmark with 10 multi-session conversations and 1,986 QA pairs across 5 categories.
- **LaceMem SQLite DB**: per-conversation database with `raw_memory`, `memory_index`, and `memory_index_edge` tables.
- **OpenAI API key** (or Gemini key for `--provider google_ai`).

### Outputs

- **Per-question result JSON** (`IRRv5_conv-XX_CatY_*.json`): generated answer, sufficiency tier, confidence, iteration count, retrieval traces, reasoning chain, retrieved facts, and timing breakdown.


## Notes on the Current Implementation

- The runner script is a research codebase. Default paths assume the conventional layout (`cache/mirix_dbs/`, `results/memory_retrieval/`); adjust via CLI flags if your layout differs.
- Memory database construction (factorising LoCoMo dialogue into LaceMem Index tuples and building Edge connections) is currently performed offline; comprehensive documentation for this pipeline is forthcoming.
- The `provider=google_ai` path is supported but tested less thoroughly than the OpenAI path.
- `IRIS_retriever_v5.py`'s factory `create_irr_retriever_v5()` exposes `calib_exact_floor`, `calib_inferrable_cap`, and `calib_partial_cap` parameters for confidence-calibration sensitivity studies.


## Acknowledgements

This work builds on:

- [LoCoMo](https://github.com/snap-research/locomo) — the long-term conversational memory benchmark used throughout our experiments.
- OpenAI GPT-4o and GPT-4o-mini — used as answer generator, sufficiency evaluator, query refinement model, and primary judge.
- DeepSeek-V3.2 — used for cross-family judge validation and LLM-backbone robustness experiments.
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — open-weights multilingual embedding model used for embedding-axis robustness experiments.
