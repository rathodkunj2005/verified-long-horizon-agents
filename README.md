# Toward Verified Long-Horizon Language Agents

**Kunj Rathod**

> *Real-model evidence for hybrid memory, bounded workspace state, open-weight planning, and latent predictive dynamics — all reproducible on a laptop.*

---

## Overview

Transcript-only language agents are a poor substrate for long-horizon autonomy. They lack durable state, accumulate irrelevant context, and offer no principled mechanism for validating state-changing actions. This paper proposes a concrete modular architecture and tests five architectural hypotheses with reproducible experiments on a laptop-class machine (Apple M1 Pro).

**Key results:**

| Experiment | Finding |
|---|---|
| LoCoMo retrieval (1,977 queries, 5,882 memories) | Hybrid lexical+dense retrieval improves Hits@5 from 0.4385 → **0.4891** over lexical baseline |
| Bounded workspace (8-document synthesis task) | Reduces cumulative prompt load by **57.96%** with full title & keyword coverage |
| Open-weight planning (distilgpt2, Qwen2.5-0.5B) | Greedy LM: **0% success**; Verifier-backed search: **100% success** |
| Latent world model (MiniLM embeddings + MLP) | One-step cosine similarity **0.9989**; exact decode accuracy low → latent state insufficient alone |
| Sequence model comparison | Linear recurrent model matches transformer on decode accuracy with **2.5× fewer parameters** |

---

## Paper

The full paper is in [`main.tex`](main.tex) / [`main.pdf`](main.pdf).

**Abstract:** We formalize a modular architecture for long-horizon language agents separating language interaction from persistent memory, bounded workspace state, predictive state estimation, and verifier-mediated commits. We replace purely synthetic toy evidence with real-model experiments: LoCoMo retrieval with sentence embeddings, a real-data bounded-workspace workflow, open-weight planning with distilgpt2 and Qwen2.5-0.5B-Instruct, an embedding-space latent dynamics baseline, and a sequence-model comparison across transformer, recurrent, and non-sequential baselines.

---

## Repository Structure

```
.
├── main.tex                          # Paper source (LaTeX)
├── main.pdf                          # Compiled paper
├── references.bib                    # Bibliography
├── Makefile                          # Build pipeline
├── requirements.txt                  # Python dependencies
├── experiments/
│   ├── run_strong_experiments.py     # Main experiment script (real models)
│   └── run_all.py                    # Lightweight synthetic baseline
├── results/
│   ├── strong/
│   │   ├── results.json              # Full results (strong experiments)
│   │   └── locomo_retrieval_details.json
│   └── results.json                  # Lightweight baseline results
└── EXPERIMENT_ROADMAP.md             # Roadmap for future experiments
```

---

## Reproducing Experiments

### Requirements

```bash
pip install -r requirements.txt
```

### Run experiments

```bash
python experiments/run_strong_experiments.py
```

Runs in ~3.5 minutes on Apple M1 Pro. Writes results to `results/strong/results.json`.

### Build paper

```bash
make paper
```

Requires `pdflatex` and `bibtex` (install via MacTeX or TeX Live).

### Lightweight baseline only (no downloads)

```bash
python experiments/run_all.py
```

Runs in seconds, stdlib only, no model downloads.

---

## Experiments

### E1 — Hybrid Memory Retrieval on LoCoMo

Evaluates TF-IDF, dense (MiniLM), and hybrid retrieval on 1,977 questions from the LoCoMo conversational memory benchmark with 5,882 memory items.

| Method | Hits@1 | Hits@5 | Hits@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| TF-IDF | 0.2352 | 0.4385 | 0.5250 | 0.3337 | 0.3513 |
| Dense (MiniLM) | 0.1507 | 0.3409 | 0.4294 | 0.2456 | 0.2586 |
| **Hybrid** | 0.2190 | **0.4891** | **0.5802** | **0.3426** | **0.3687** |

### E2 — Bounded Workspace vs. Transcript Accumulation

Real-data literature synthesis over 8 research abstracts. Bounded workspace tracks structured state (`task`, `findings`, `next_steps`, `citations`) rather than the full transcript.

- Cumulative prompt reduction: **57.96%**
- Title coverage: **100%** | Keyword coverage: **100%**

### E3 — Open-Weight Planning with Verification

distilgpt2 and Qwen2.5-0.5B-Instruct as planners in a constrained discrete environment.

| Condition | Success Rate | Mean Steps |
|---|---|---|
| Greedy LM (distilgpt2) | 0.0% | 3.9 |
| Greedy LM (Qwen2.5-0.5B) | 0.0% | 4.5 |
| LM + Verifier-backed BFS | **100%** | 8.2 |
| Pure BFS (oracle) | 100% | 8.2 |

### E4 — Latent World Model

MLP transition model on MiniLM state embeddings (9,471 transitions, 1,774 unique states).

| Metric | Value |
|---|---|
| One-step cosine similarity | 0.9989 |
| Top-1 decode accuracy (local) | 10.42% |
| Top-5 decode accuracy (local) | 39.83% |
| Rollout cosine at horizon 5 | 0.9990 |

### E5 — Sequence Model Comparison

| Model | Top-1 Decode | Top-5 Decode | Parameters | Train Time |
|---|---|---|---|---|
| Causal Transformer | 6.03% | 29.29% | 368,640 | 1.42s |
| **Linear Recurrent** | **6.31%** | **30.97%** | **149,376** | 0.84s |
| MLP Control | 6.03% | 29.64% | 659,840 | 0.46s |

---

## Models Used

| Model | Purpose | Size |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | State encoding, retrieval | ~80 MB |
| `distilgpt2` | Planning prior | ~350 MB |
| `Qwen/Qwen2.5-0.5B-Instruct` | Planning prior | ~1 GB |

---

## Citation

```bibtex
@article{rathod2026verified,
  title={Toward Verified Long-Horizon Language Agents: Real-Model Evidence for
         Hybrid Memory, Bounded State, Open-Weight Planning, and Latent Predictive Dynamics},
  author={Rathod, Kunj},
  year={2026}
}
```

---

## References

Key citations: LoCoMo, MemGPT, CoALA, LATS, ReAct, InfiAgent, ToolGate, V-JEPA 2, VL-JEPA, PlaNet, DreamerV3, Mamba, Griffin. See [`references.bib`](references.bib) for full bibliography.
