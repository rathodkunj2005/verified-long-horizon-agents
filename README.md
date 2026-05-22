# Verified Long-Horizon Language Agents

Paper, experiments, and results for a long-horizon language-agent architecture built around persistent memory, bounded workspace management, lightweight planning, and latent predictive dynamics.

Repository:
https://github.com/rathodkunj2005/verified-long-horizon-agents

## What this repo contains

This project evaluates six concrete claims:

1. Hybrid retrieval over persistent memory on LoCoMo
2. Bounded-workspace reconstruction on a real literature-synthesis workflow
3. Stronger open-weight planning baselines, including Qwen2.5-1.5B-Instruct and a 3-shot CoT prompt
4. Embedding-space latent dynamics over 9,512 symbolic transitions
5. An integrated memory + workspace + verification pipeline
6. An architectural comparison between MLP, causal transformer, and minimal LRU predictors

## Repository layout

- `main.tex` — paper source
- `references.bib` — bibliography
- `experiments/run_strong_experiments.py` — main experiment runner
- `experiments/run_all.py` — lighter experiment runner
- `results/strong/results.json` — latest strong-run outputs
- `results/` — aggregated metrics and intermediate artifacts
- `EXPERIMENT_ROADMAP.md` — experiment notes and future directions
- `main.pdf` — compiled paper artifact

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make experiments
```

For the lighter experiment pass:

```bash
make experiments-lite
```

## Build the paper

From the repository root:

```bash
make paper
```

This runs:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Current empirical story

- Hybrid retrieval improves LoCoMo Hits@5 over lexical-only retrieval.
- Bounded workspace reduces cumulative prompt load by roughly 58%.
- Greedy planning still fails after modest model scaling and 3-shot CoT prompting.
- The integrated pipeline reduces prompt load, improves retrieval overlap, and prevents malformed note commits.
- Latent prediction is geometrically easy but exact decode remains weak.
- A minimal recurrent proxy slightly exceeds a small causal transformer on local decode accuracy while using fewer parameters.

## Status

This repo is a research artifact, not a polished production framework. The main value is the paper + experiment bundle and the concrete negative/positive findings documented in `results/`.
