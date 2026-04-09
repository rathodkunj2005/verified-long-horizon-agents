#!/usr/bin/env python3
"""Run lightweight, MacBook-friendly experiments for the LLM + world-model paper.

All experiments use only the Python standard library and run in a few seconds.
Outputs are written to ./results/ as JSON and CSV.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SEED = 7
random.seed(SEED)

SYNONYMS = {
    "purchase": ["buy", "acquire", "order"],
    "meeting": ["call", "sync", "standup"],
    "research": ["study", "investigation", "analysis"],
    "robot": ["arm", "manipulator", "agent"],
    "kitchen": ["galley", "cookspace"],
    "deadline": ["due", "submission", "cutoff"],
    "error": ["bug", "failure", "fault"],
    "schedule": ["plan", "calendar", "timeline"],
    "memory": ["recall", "history", "context"],
    "world": ["environment", "physical", "latent"],
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with",
    "at", "from", "by", "is", "was", "be", "this", "that", "it", "as", "into",
}


def tokenize(text: str) -> List[str]:
    chars = []
    for ch in text.lower():
        chars.append(ch if ch.isalnum() or ch.isspace() else " ")
    return [t for t in "".join(chars).split() if t and t not in STOPWORDS]


@dataclass
class MemoryDoc:
    doc_id: str
    kind: str
    timestamp: int
    text: str
    gold_terms: Tuple[str, ...]


def build_memory_corpus(n_docs: int = 240) -> Tuple[List[MemoryDoc], List[dict]]:
    subjects = [
        "robot kitchen pickup", "calendar deadline planning", "research paper drafting",
        "world model planning", "memory consolidation pipeline", "retrieval benchmark analysis",
        "tool verification loop", "agent workspace reconstruction", "streaming video encoder",
        "vision language alignment",
    ]
    verbs = ["purchase", "schedule", "debug", "research", "organize", "verify"]
    objects = [
        "apples", "submission", "tabletop policy", "planner", "retriever",
        "report", "benchmark", "decoder", "workspace", "controller",
    ]
    locations = ["kitchen", "lab", "office", "garage", "desk"]

    docs: List[MemoryDoc] = []
    for i in range(n_docs):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        loc = random.choice(locations)
        timestamp = 1_700_000_000 + i * 1800
        kind = random.choice(["world_fact", "experience", "observation", "mental_model"])
        text = (
            f"{kind.replace('_', ' ')}: the agent must {verb} the {obj} related to {subject} "
            f"before the next deadline in the {loc}. temporal marker {i}."
        )
        docs.append(MemoryDoc(f"doc_{i}", kind, timestamp, text, (verb, obj, loc)))

    queries = []
    for i in range(60):
        doc = random.choice(docs)
        verb, obj, loc = doc.gold_terms
        alt_verb = random.choice(SYNONYMS.get(verb, [verb]))
        query = f"find the note about {alt_verb} {obj} in the {loc}"
        queries.append({"query": query, "target": doc.doc_id, "time": doc.timestamp})
    return docs, queries


class BM25Retriever:
    def __init__(self, docs: List[MemoryDoc]):
        self.docs = docs
        self.doc_tokens = [tokenize(d.text) for d in docs]
        self.doc_freq = Counter()
        for toks in self.doc_tokens:
            for tok in set(toks):
                self.doc_freq[tok] += 1
        self.avgdl = statistics.mean(len(toks) for toks in self.doc_tokens)
        self.N = len(docs)

    def score(self, query: str, idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        q_tokens = tokenize(query)
        tf = Counter(self.doc_tokens[idx])
        dl = len(self.doc_tokens[idx])
        score = 0.0
        for tok in q_tokens:
            df = self.doc_freq.get(tok, 0)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            freq = tf.get(tok, 0)
            numer = freq * (k1 + 1)
            denom = freq + k1 * (1 - b + b * dl / self.avgdl)
            score += idf * (numer / denom if denom else 0.0)
        return score

    def topk(self, query: str, k: int = 5) -> List[str]:
        scored = [(self.score(query, i), self.docs[i].doc_id) for i in range(len(self.docs))]
        scored.sort(reverse=True)
        return [doc_id for _, doc_id in scored[:k]]


class CharNGramRetriever:
    def __init__(self, docs: List[MemoryDoc], n: int = 3):
        self.docs = docs
        self.n = n
        self.doc_vecs = [self._vec(d.text) for d in docs]

    def _vec(self, text: str) -> Counter:
        txt = f"  {text.lower()}  "
        return Counter(txt[i : i + self.n] for i in range(max(0, len(txt) - self.n + 1)))

    def _cos(self, a: Counter, b: Counter) -> float:
        inter = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in inter)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def topk(self, query: str, k: int = 5) -> List[str]:
        qv = self._vec(query)
        scored = [(self._cos(qv, dv), self.docs[i].doc_id) for i, dv in enumerate(self.doc_vecs)]
        scored.sort(reverse=True)
        return [doc_id for _, doc_id in scored[:k]]


class HybridRetriever:
    def __init__(self, docs: List[MemoryDoc]):
        self.docs = docs
        self.bm25 = BM25Retriever(docs)
        self.char = CharNGramRetriever(docs)
        self.doc_index = {d.doc_id: d for d in docs}

    def topk(self, query: str, query_time: int, k: int = 5) -> List[str]:
        # Combine lexical, fuzzy semantic, and temporal proximity.
        bm = self.bm25.topk(query, k=len(self.docs))
        ch = self.char.topk(query, k=len(self.docs))
        scores = defaultdict(float)
        for rank, doc_id in enumerate(bm[:40]):
            scores[doc_id] += 1.0 / (rank + 1)
        for rank, doc_id in enumerate(ch[:40]):
            scores[doc_id] += 1.0 / (rank + 1)
        for doc_id in list(scores.keys()):
            dt = abs(self.doc_index[doc_id].timestamp - query_time) / 86400.0
            scores[doc_id] += math.exp(-dt)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:k]]


def evaluate_retrieval() -> dict:
    docs, queries = build_memory_corpus()
    bm25 = BM25Retriever(docs)
    char = CharNGramRetriever(docs)
    hybrid = HybridRetriever(docs)
    methods = {
        "bm25": lambda q: bm25.topk(q["query"]),
        "char_ngram": lambda q: char.topk(q["query"]),
        "hybrid": lambda q: hybrid.topk(q["query"], q["time"]),
    }
    summary = {}
    rows = []
    for name, fn in methods.items():
        hits1 = hits5 = mrr = 0.0
        start = time.perf_counter()
        for q in queries:
            topk = fn(q)
            target = q["target"]
            if topk and topk[0] == target:
                hits1 += 1
            if target in topk:
                hits5 += 1
                mrr += 1.0 / (topk.index(target) + 1)
            rows.append({"method": name, "query": q["query"], "target": target, "ranked": " | ".join(topk)})
        elapsed = time.perf_counter() - start
        n = len(queries)
        summary[name] = {
            "hits_at_1": round(hits1 / n, 3),
            "hits_at_5": round(hits5 / n, 3),
            "mrr_at_5": round(mrr / n, 3),
            "seconds": round(elapsed, 4),
        }
    with open(RESULTS / "retrieval_details.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "query", "target", "ranked"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def simulate_context_bloat(steps: int = 150) -> dict:
    transcript = []
    workspace_state = {
        "plan": "Investigate memory + world-model architecture and produce verified artifacts.",
        "notes": [],
        "open_questions": ["How much context is enough?", "When does verification pay off?"],
    }
    full_prompt_chars = []
    bounded_prompt_chars = []
    for step in range(1, steps + 1):
        observation = (
            f"step {step}: observed new source, intermediate result, and experiment notes. "
            f"This chunk intentionally adds enough characters to mimic a verbose tool trace."
        )
        action = f"action {step}: summarize finding and update state file."
        transcript.append(observation)
        transcript.append(action)
        workspace_state["notes"].append(f"s{step}: compact note")
        serialized_state = json.dumps({
            "plan": workspace_state["plan"],
            "recent_notes": workspace_state["notes"][-8:],
            "open_questions": workspace_state["open_questions"],
        })
        full_prompt_chars.append(sum(len(x) for x in transcript))
        bounded_prompt_chars.append(len(serialized_state) + sum(len(x) for x in transcript[-6:]))
    reduction = 1 - (sum(bounded_prompt_chars) / sum(full_prompt_chars))
    return {
        "steps": steps,
        "final_full_prompt_chars": full_prompt_chars[-1],
        "final_bounded_prompt_chars": bounded_prompt_chars[-1],
        "cumulative_full_chars_processed": sum(full_prompt_chars),
        "cumulative_bounded_chars_processed": sum(bounded_prompt_chars),
        "relative_reduction": round(reduction, 3),
    }


DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def generate_grid(size: int = 8, obstacle_prob: float = 0.18) -> Tuple[Tuple[int, int], Tuple[int, int], set]:
    cells = [(r, c) for r in range(size) for c in range(size)]
    start = (0, 0)
    goal = (size - 1, size - 1)
    obstacles = {cell for cell in cells if cell not in {start, goal} and random.random() < obstacle_prob}
    return start, goal, obstacles


def greedy_plan(size: int, start: Tuple[int, int], goal: Tuple[int, int], obstacles: set) -> Tuple[bool, int]:
    pos = start
    visited = {pos}
    steps = 0
    while pos != goal and steps < size * size * 2:
        candidates = []
        for dr, dc in DIRS:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in obstacles:
                dist = abs(goal[0] - nr) + abs(goal[1] - nc)
                candidates.append((dist, (nr, nc)))
        if not candidates:
            return False, steps
        candidates.sort(key=lambda x: x[0])
        moved = False
        for _, nxt in candidates:
            if nxt not in visited:
                pos = nxt
                visited.add(pos)
                moved = True
                break
        if not moved:
            return False, steps
        steps += 1
    return pos == goal, steps


def bfs_plan(size: int, start: Tuple[int, int], goal: Tuple[int, int], obstacles: set) -> Tuple[bool, int]:
    queue = [start]
    parent = {start: None}
    while queue:
        cur = queue.pop(0)
        if cur == goal:
            length = 0
            while parent[cur] is not None:
                length += 1
                cur = parent[cur]
            return True, length
        for dr, dc in DIRS:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if 0 <= nr < size and 0 <= nc < size and nxt not in obstacles and nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    return False, 0


def evaluate_planning(n_trials: int = 120, size: int = 8) -> dict:
    greedy_success = 0
    bfs_success = 0
    greedy_lengths = []
    bfs_lengths = []
    for _ in range(n_trials):
        start, goal, obstacles = generate_grid(size=size)
        g_ok, g_len = greedy_plan(size, start, goal, obstacles)
        b_ok, b_len = bfs_plan(size, start, goal, obstacles)
        greedy_success += int(g_ok)
        bfs_success += int(b_ok)
        if g_ok:
            greedy_lengths.append(g_len)
        if b_ok:
            bfs_lengths.append(b_len)
    return {
        "trials": n_trials,
        "greedy_success_rate": round(greedy_success / n_trials, 3),
        "verified_search_success_rate": round(bfs_success / n_trials, 3),
        "greedy_mean_path_len": round(statistics.mean(greedy_lengths), 2) if greedy_lengths else None,
        "verified_mean_path_len": round(statistics.mean(bfs_lengths), 2) if bfs_lengths else None,
    }


def main() -> None:
    results = {
        "seed": SEED,
        "retrieval": evaluate_retrieval(),
        "context_bloat": simulate_context_bloat(),
        "planning": evaluate_planning(),
    }
    with open(RESULTS / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
