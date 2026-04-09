# Verified Long-Horizon Language Agents

Paper, experiments, and results for the LLM world-model / agent architecture project.

Repository:
https://github.com/rathodkunj2005/verified-long-horizon-agents

## Scope

This version evaluates six concrete claims:

1. Hybrid retrieval over persistent memory on LoCoMo
2. Bounded-workspace reconstruction on a real literature-synthesis workflow
3. Stronger open-weight planning baselines, including Qwen2.5-1.5B-Instruct and a 3-shot CoT prompt
4. Embedding-space latent dynamics over 9,512 symbolic transitions
5. An integrated memory + workspace + verification pipeline
6. An architectural comparison between MLP, causal transformer, and minimal LRU predictors

## Key files

- `main.tex` — paper source
- `references.bib` — bibliography
- `experiments/run_strong_experiments.py` — experiment runner
- `results/strong/results.json` — latest experiment outputs
- `main.pdf` — compiled paper

## Run experiments

```bash
source /Users/kunjrathod/.hermes/hermes-agent/venv/bin/activate
python /Users/kunjrathod/.hermes/hermes-agent/research/llm_world_model_paper/experiments/run_strong_experiments.py
```

## Build paper

```bash
cd /Users/kunjrathod/.hermes/hermes-agent/research/llm_world_model_paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Current empirical story

- Hybrid retrieval improves LoCoMo Hits@5 over lexical-only retrieval.
- Bounded workspace reduces cumulative prompt load by roughly 58%.
- Greedy planning still fails after modest model scaling and 3-shot CoT prompting.
- The integrated pipeline reduces prompt load, improves retrieval overlap, and prevents malformed note commits.
- Latent prediction is geometrically easy but exact decode remains weak.
- A minimal recurrent proxy slightly exceeds a small causal transformer on local decode accuracy while using fewer parameters.
