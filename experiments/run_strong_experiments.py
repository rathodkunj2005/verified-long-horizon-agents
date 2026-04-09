#!/usr/bin/env python3
"""Stronger, submission-grade small-scale experiments for the LLM + world-model paper.

Experiments:
1. Real conversational-memory retrieval on LoCoMo using TF-IDF, dense embeddings, and hybrid fusion.
2. Verified planning with a small open-weight LM (distilgpt2) on a symbolic key-door gridworld.
3. JEPA-inspired latent dynamics experiment: predict next-state embeddings in latent space and decode by nearest neighbor.

All outputs are written under ../results/strong/.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import xml.etree.ElementTree as ET

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "strong"
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_TRANSITIONS = 9512
MAX_REASONABLE_MODEL_BYTES = 4 * 1024**3
PLANNER_CONDITIONS = [
    {
        "key": "distilgpt2",
        "model_name": "distilgpt2",
        "label": "distilgpt2",
        "n_tasks": 20,
        "prompt_style": "direct",
    },
    {
        "key": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen-0.5B-Instruct",
        "n_tasks": 10,
        "prompt_style": "direct",
    },
    {
        "key": "Qwen/Qwen2.5-0.5B-Instruct::cot3",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen-0.5B-Instruct CoT-3shot",
        "n_tasks": 10,
        "prompt_style": "cot3",
    },
    {
        "key": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen-1.5B-Instruct",
        "n_tasks": 10,
        "prompt_style": "direct",
        "size_limit_bytes": MAX_REASONABLE_MODEL_BYTES,
    },
]
ARXIV_IDS = [
    "2506.09985",
    "2512.10942",
    "2310.08560",
    "2309.02427",
    "2310.04406",
    "2402.17753",
    "2601.03204",
    "2601.04688",
]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[int]:
    scores: Dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank + 1)
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def top_keywords(text: str, k: int = 6) -> List[str]:
    tokens = []
    word = []
    for ch in text.lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                tokens.append("".join(word))
                word = []
    if word:
        tokens.append("".join(word))
    stop = {
        "the", "and", "for", "that", "with", "this", "from", "into", "their", "have", "using",
        "they", "these", "through", "over", "than", "across", "long", "agent", "language",
        "models", "model", "agents", "paper", "show", "shows", "results", "based",
    }
    counts = defaultdict(int)
    for t in tokens:
        if len(t) < 4 or t in stop or t.isdigit():
            continue
        counts[t] += 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:k]]


def estimate_repo_bytes(repo_id: str) -> Optional[int]:
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception:
        return None
    total = 0
    seen = False
    for sibling in getattr(info, "siblings", []) or []:
        size = getattr(sibling, "size", None)
        if size is not None:
            total += int(size)
            seen = True
    return total if seen else None


def summarize_skip(reason: str, estimated_bytes: Optional[int] = None) -> dict:
    out = {"reason": reason}
    if estimated_bytes is not None:
        out["estimated_model_bytes"] = int(estimated_bytes)
    return out


# ---------------------------------------------------------------------------
# Experiment 1: LoCoMo retrieval
# ---------------------------------------------------------------------------

def load_locomo() -> list:
    cache = RESULTS / "locomo10.json"
    if not cache.exists():
        data = requests.get(LOCOMO_URL, timeout=60)
        data.raise_for_status()
        cache.write_text(data.text)
    return json.loads(cache.read_text())


def build_locomo_retrieval_dataset() -> Tuple[List[str], List[dict]]:
    raw = load_locomo()
    memories: List[str] = []
    memory_meta: List[dict] = []
    queries: List[dict] = []
    id_to_idx: Dict[str, int] = {}

    for conv in raw:
        c = conv["conversation"]
        sample_id = conv["sample_id"]
        sessions = [k for k in c.keys() if k.startswith("session_") and k.split("_")[-1].isdigit()]
        sessions = sorted(sessions, key=lambda x: int(x.split("_")[-1]))
        for session_name in sessions:
            session_idx = int(session_name.split("_")[-1])
            dt_key = f"session_{session_idx}_date_time"
            dt_value = c.get(dt_key, f"session {session_idx}")
            for turn in c[session_name]:
                text = turn["text"].strip()
                memory_text = f"[{sample_id} | {dt_value} | {turn['speaker']}] {text}"
                idx = len(memories)
                memories.append(memory_text)
                id_to_idx[turn["dia_id"]] = idx
                memory_meta.append(
                    {
                        "sample_id": sample_id,
                        "dia_id": turn["dia_id"],
                        "speaker": turn["speaker"],
                        "session": session_idx,
                        "text": text,
                    }
                )

        for qa in conv["qa"]:
            gold = [id_to_idx[e] for e in qa["evidence"] if e in id_to_idx]
            if not gold:
                continue
            queries.append(
                {
                    "sample_id": sample_id,
                    "question": qa["question"],
                    "answer": qa.get("answer", qa.get("adversarial_answer", "")),
                    "category": qa["category"],
                    "gold": gold,
                }
            )
    return memories, queries


def retrieval_metrics(ranked: List[int], gold: List[int], ks=(1, 5, 10)) -> Dict[str, float]:
    gold_set = set(gold)
    out = {}
    for k in ks:
        out[f"hits@{k}"] = 1.0 if any(idx in gold_set for idx in ranked[:k]) else 0.0
    rr = 0.0
    for pos, idx in enumerate(ranked, start=1):
        if idx in gold_set:
            rr = 1.0 / pos
            break
    out["mrr"] = rr
    # nDCG@10
    dcg = 0.0
    for pos, idx in enumerate(ranked[:10], start=1):
        if idx in gold_set:
            dcg += 1.0 / math.log2(pos + 1)
    ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(len(gold), 10) + 1))
    out["ndcg@10"] = dcg / ideal if ideal else 0.0
    return out


def run_locomo_retrieval(embedder: SentenceTransformer) -> dict:
    memories, queries = build_locomo_retrieval_dataset()

    tfidf = TfidfVectorizer(stop_words="english", min_df=1)
    X_lex = tfidf.fit_transform(memories)

    memory_emb = embedder.encode(memories, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    memory_emb = l2_normalize(memory_emb.astype(np.float32))

    summary = {}
    details = []
    methods = ["tfidf", "dense", "hybrid"]
    scores_by_method = {m: defaultdict(list) for m in methods}

    start = time.perf_counter()
    query_emb = embedder.encode([q["question"] for q in queries], batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    query_emb = l2_normalize(query_emb.astype(np.float32))
    dense_elapsed = time.perf_counter() - start

    for qi, q in enumerate(queries):
        # lexical ranking
        q_lex = tfidf.transform([q["question"]])
        lex_scores = (X_lex @ q_lex.T).toarray().ravel()
        lex_rank = np.argsort(-lex_scores)

        dense_scores = memory_emb @ query_emb[qi]
        dense_rank = np.argsort(-dense_scores)

        hybrid_rank = reciprocal_rank_fusion([lex_rank.tolist(), dense_rank.tolist()])

        for method, rank in {
            "tfidf": lex_rank.tolist(),
            "dense": dense_rank.tolist(),
            "hybrid": hybrid_rank,
        }.items():
            metrics = retrieval_metrics(rank, q["gold"])
            for k, v in metrics.items():
                scores_by_method[method][k].append(v)

        details.append(
            {
                "question": q["question"],
                "gold_dia_ids": [memories[g] for g in q["gold"]],
                "tfidf_top1": memories[lex_rank[0]],
                "dense_top1": memories[dense_rank[0]],
                "hybrid_top1": memories[hybrid_rank[0]],
                "category": q["category"],
            }
        )

    for method in methods:
        summary[method] = {metric: round(float(np.mean(vals)), 4) for metric, vals in scores_by_method[method].items()}
        summary[method]["num_queries"] = len(queries)
        summary[method]["num_memories"] = len(memories)
    summary["dense_encoding_seconds"] = round(dense_elapsed, 3)

    (RESULTS / "locomo_retrieval_details.json").write_text(json.dumps(details[:100], indent=2))
    return summary


# ---------------------------------------------------------------------------
# Shared gridworld for planner + latent world model
# ---------------------------------------------------------------------------

ACTIONS = ["north", "south", "east", "west", "pickup key", "open door"]
DIRMAP = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
}


@dataclass(frozen=True)
class State:
    row: int
    col: int
    has_key: bool
    door_open: bool


@dataclass
class Task:
    size: int
    start: State
    goal: Tuple[int, int]
    key_pos: Tuple[int, int]
    door_pos: Tuple[int, int]
    walls: frozenset


def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    return [(pos[0] + dr, pos[1] + dc) for dr, dc in DIRMAP.values()]


def make_task(size: int = 5, rng=None) -> Task:
    rng = rng or random
    while True:
        cells = [(r, c) for r in range(size) for c in range(size)]
        start_pos = (0, 0)
        goal = (size - 1, size - 1)
        key = rng.choice([c for c in cells if c not in {start_pos, goal}])
        door = rng.choice([c for c in cells if c not in {start_pos, goal, key}])
        wall_count = rng.randint(2, 5)
        walls = set(rng.sample([c for c in cells if c not in {start_pos, goal, key, door}], wall_count))
        # Ensure the door is adjacent to goal to make it semantically meaningful.
        if abs(door[0] - goal[0]) + abs(door[1] - goal[1]) != 1:
            continue
        start = State(start_pos[0], start_pos[1], False, False)
        task = Task(size=size, start=start, goal=goal, key_pos=key, door_pos=door, walls=frozenset(walls))
        if bfs(task, use_lm_order=False)[0]:
            return task


def legal_actions(task: Task, state: State) -> List[str]:
    acts = []
    for action, (dr, dc) in DIRMAP.items():
        nr, nc = state.row + dr, state.col + dc
        if not (0 <= nr < task.size and 0 <= nc < task.size):
            continue
        if (nr, nc) in task.walls:
            continue
        if (nr, nc) == task.door_pos and not state.door_open:
            continue
        acts.append(action)
    if (state.row, state.col) == task.key_pos and not state.has_key:
        acts.append("pickup key")
    if state.has_key and not state.door_open and abs(state.row - task.door_pos[0]) + abs(state.col - task.door_pos[1]) == 1:
        acts.append("open door")
    return acts


def transition(task: Task, state: State, action: str) -> State:
    if action not in legal_actions(task, state):
        raise ValueError(f"illegal action {action}")
    if action in DIRMAP:
        dr, dc = DIRMAP[action]
        return State(state.row + dr, state.col + dc, state.has_key, state.door_open)
    if action == "pickup key":
        return State(state.row, state.col, True, state.door_open)
    if action == "open door":
        return State(state.row, state.col, state.has_key, True)
    raise ValueError(action)


def is_goal(task: Task, state: State) -> bool:
    return (state.row, state.col) == task.goal


def render_state(task: Task, state: State) -> str:
    return (
        f"Grid size {task.size}. Agent at ({state.row},{state.col}). "
        f"Goal at {task.goal}. Key at {task.key_pos}. Door at {task.door_pos}. "
        f"Walls at {sorted(task.walls)}. Has_key={state.has_key}. Door_open={state.door_open}."
    )


def bfs(task: Task, use_lm_order: bool, planner=None) -> Tuple[bool, List[str], int]:
    from collections import deque

    frontier = deque([(task.start, [])])
    visited = {task.start}
    expansions = 0
    while frontier:
        state, path = frontier.popleft()
        if is_goal(task, state):
            return True, path, expansions
        acts = legal_actions(task, state)
        if use_lm_order and planner is not None:
            acts = planner.rank_actions(task, state, acts)
        expansions += 1
        for act in acts:
            nxt = transition(task, state, act)
            if nxt not in visited:
                visited.add(nxt)
                frontier.append((nxt, path + [act]))
    return False, [], expansions


class CausalLMPlanner:
    def __init__(self, model_name: str, prompt_style: str = "direct"):
        self.model_name = model_name
        self.prompt_style = prompt_style
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = torch.float16 if DEVICE == "mps" else None
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.to(DEVICE)
        self.model.eval()

    def build_prompt(self, task: Task, state: State) -> str:
        state_text = render_state(task, state)
        if self.prompt_style == "cot3":
            return (
                "You are choosing the next action in a gridworld. Think briefly about key possession, door status, and the shortest path, then output one legal action.\n\n"
                "Example 1\n"
                "State: Grid size 5. Agent at (0,0). Goal at (4,4). Key at (0,1). Door at (3,4). Walls at [(1, 1)]. Has_key=False. Door_open=False.\n"
                "Reasoning: The key is adjacent and I do not have it, so I should move east first instead of drifting toward the locked door.\n"
                "Action: east\n\n"
                "Example 2\n"
                "State: Grid size 5. Agent at (0,1). Goal at (4,4). Key at (0,1). Door at (3,4). Walls at [(1, 1)]. Has_key=False. Door_open=False.\n"
                "Reasoning: I am standing on the key, so the next move should update the hidden boolean state has_key before any long-horizon navigation.\n"
                "Action: pickup key\n\n"
                "Example 3\n"
                "State: Grid size 5. Agent at (2,4). Goal at (4,4). Key at (0,1). Door at (3,4). Walls at [(1, 1)]. Has_key=True. Door_open=False.\n"
                "Reasoning: The door is directly adjacent, and opening it is necessary before stepping into the goal corridor.\n"
                "Action: open door\n\n"
                f"State: {state_text}\n"
                "Reasoning: Track whether the agent already has the key, whether the door is open, and which legal action most directly preserves progress toward the goal.\n"
                "Action: "
            )
        return (
            "You are choosing the next action in a gridworld. "
            "Choose the action that best helps reach the goal.\n"
            f"State: {state_text}\n"
            "Action: "
        )

    @torch.no_grad()
    def score_action(self, prompt: str, action: str) -> float:
        full = prompt + action
        enc_full = self.tokenizer(full, return_tensors="pt")
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc_full["input_ids"].to(DEVICE)
        logits = self.model(input_ids).logits[:, :-1, :]
        target = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        prompt_len = enc_prompt["input_ids"].shape[1] - 1
        return float(token_log_probs[0, prompt_len:].sum().item())

    def rank_actions(self, task: Task, state: State, actions: List[str]) -> List[str]:
        prompt = self.build_prompt(task, state)
        scored = [(self.score_action(prompt, a), a) for a in actions]
        scored.sort(reverse=True)
        return [a for _, a in scored]



def run_planner_experiment(planner: CausalLMPlanner, n_tasks: int = 60) -> dict:
    results = defaultdict(list)
    for _ in range(n_tasks):
        task = make_task(size=5)

        # Greedy LM policy
        state = task.start
        traj = []
        visited = set()
        for _step in range(40):
            if is_goal(task, state):
                break
            acts = legal_actions(task, state)
            if not acts:
                break
            ranked = planner.rank_actions(task, state, acts)
            action = ranked[0]
            traj.append(action)
            state = transition(task, state, action)
            if state in visited:
                break
            visited.add(state)
        results["greedy_success"].append(1.0 if is_goal(task, state) else 0.0)
        results["greedy_length"].append(len(traj))

        # LM-guided verifier search
        ok_lm, path_lm, exp_lm = bfs(task, use_lm_order=True, planner=planner)
        results["lm_verify_success"].append(1.0 if ok_lm else 0.0)
        results["lm_verify_length"].append(len(path_lm) if ok_lm else 0)
        results["lm_verify_expansions"].append(exp_lm)

        # Pure BFS
        ok_bfs, path_bfs, exp_bfs = bfs(task, use_lm_order=False)
        results["bfs_success"].append(1.0 if ok_bfs else 0.0)
        results["bfs_length"].append(len(path_bfs) if ok_bfs else 0)
        results["bfs_expansions"].append(exp_bfs)

    return {
        "num_tasks": n_tasks,
        "greedy_lm_success_rate": round(float(np.mean(results["greedy_success"])), 4),
        "greedy_lm_mean_steps": round(float(np.mean(results["greedy_length"])), 3),
        "lm_guided_verified_success_rate": round(float(np.mean(results["lm_verify_success"])), 4),
        "lm_guided_verified_mean_steps": round(float(np.mean([x for x in results["lm_verify_length"] if x > 0])), 3),
        "lm_guided_verified_mean_expansions": round(float(np.mean(results["lm_verify_expansions"])), 3),
        "pure_bfs_success_rate": round(float(np.mean(results["bfs_success"])), 4),
        "pure_bfs_mean_steps": round(float(np.mean([x for x in results["bfs_length"] if x > 0])), 3),
        "pure_bfs_mean_expansions": round(float(np.mean(results["bfs_expansions"])), 3),
    }


# ---------------------------------------------------------------------------
# Experiment 3: latent dynamics
# ---------------------------------------------------------------------------


class LatentDynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, state_vec: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        a = self.action_embed(action_idx)
        out = self.net(torch.cat([state_vec, a], dim=-1))
        return F.normalize(out, dim=-1)


def task_signature(task: Task) -> str:
    return f"goal={task.goal}|key={task.key_pos}|door={task.door_pos}|walls={sorted(task.walls)}"


def generate_transition_dataset(target_transitions: int = TARGET_TRANSITIONS, max_steps: int = 20, seed: int = 123):
    rng = random.Random(seed)
    records = []
    all_states = []
    episodes = []
    while len(records) < target_transitions:
        task = make_task(size=5, rng=rng)
        sig = task_signature(task)
        for _ in range(8):
            if len(records) >= target_transitions:
                break
            state = task.start
            episode = []
            all_states.append((sig, task, state))
            for _ in range(max_steps):
                if len(records) >= target_transitions:
                    break
                acts = legal_actions(task, state)
                if not acts:
                    break
                act = rng.choice(acts)
                nxt = transition(task, state, act)
                records.append((sig, task, state, act, nxt))
                episode.append((sig, task, state, act, nxt))
                all_states.append((sig, task, nxt))
                state = nxt
                if is_goal(task, state):
                    break
            if episode:
                episodes.append(episode)
    return records[:target_transitions], all_states, episodes


def encode_texts(embedder: SentenceTransformer, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
    vecs = embedder.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=show_progress_bar)
    return l2_normalize(vecs.astype(np.float32))


def prepare_latent_bundle(embedder: SentenceTransformer, seed: int = 123) -> dict:
    records, all_states, episodes = generate_transition_dataset(seed=seed)
    unique_state_texts = []
    unique_state_lookup = {}
    task_to_state_indices = defaultdict(list)
    for sig, task, state in all_states:
        txt = render_state(task, state)
        if txt not in unique_state_lookup:
            unique_state_lookup[txt] = len(unique_state_texts)
            unique_state_texts.append(txt)
        task_to_state_indices[sig].append(unique_state_lookup[txt])
    for sig in list(task_to_state_indices.keys()):
        task_to_state_indices[sig] = sorted(set(task_to_state_indices[sig]))

    state_bank = encode_texts(embedder, unique_state_texts, show_progress_bar=True)
    action_to_idx = {a: i for i, a in enumerate(ACTIONS)}

    X_states = np.stack([state_bank[unique_state_lookup[render_state(task, state)]] for sig, task, state, _, _ in records])
    Y_states = np.stack([state_bank[unique_state_lookup[render_state(task, nxt)]] for sig, task, _, _, nxt in records])
    A = np.array([action_to_idx[act] for _, _, _, act, _ in records], dtype=np.int64)
    SIGS = [sig for sig, _, _, _, _ in records]

    seq_tokens = []
    seq_targets = []
    seq_sigs = []
    for episode in episodes:
        tokens = []
        targets = []
        sig = episode[0][0]
        for _, task, state, act, nxt in episode:
            state_vec = state_bank[unique_state_lookup[render_state(task, state)]]
            next_vec = state_bank[unique_state_lookup[render_state(task, nxt)]]
            action_onehot = np.zeros(len(ACTIONS), dtype=np.float32)
            action_onehot[action_to_idx[act]] = 1.0
            token = np.concatenate([state_vec, action_onehot], axis=0)
            tokens.append(token.astype(np.float32))
            targets.append(next_vec.astype(np.float32))
        seq_tokens.append(np.stack(tokens))
        seq_targets.append(np.stack(targets))
        seq_sigs.append(sig)

    return {
        "records": records,
        "all_states": all_states,
        "episodes": episodes,
        "state_bank": state_bank,
        "unique_state_lookup": unique_state_lookup,
        "task_to_state_indices": task_to_state_indices,
        "X_states": X_states,
        "Y_states": Y_states,
        "A": A,
        "SIGS": SIGS,
        "action_to_idx": action_to_idx,
        "seq_tokens": seq_tokens,
        "seq_targets": seq_targets,
        "seq_sigs": seq_sigs,
        "state_dim": int(state_bank.shape[1]),
        "token_dim": int(seq_tokens[0].shape[1]),
    }


def run_latent_dynamics(bundle: dict, epochs: int = 8) -> dict:
    state_bank = bundle["state_bank"]
    X_states = bundle["X_states"]
    Y_states = bundle["Y_states"]
    A = bundle["A"]
    SIGS = bundle["SIGS"]
    task_to_state_indices = bundle["task_to_state_indices"]
    action_to_idx = bundle["action_to_idx"]
    n = len(X_states)

    perm = np.random.RandomState(SEED).permutation(n)
    split = int(0.85 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    model = LatentDynamics(state_dim=X_states.shape[1], action_dim=len(ACTIONS), hidden=512).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def batchify(indices, batch_size=128):
        for i in range(0, len(indices), batch_size):
            idx = indices[i : i + batch_size]
            yield (
                torch.tensor(X_states[idx], dtype=torch.float32, device=DEVICE),
                torch.tensor(A[idx], dtype=torch.long, device=DEVICE),
                torch.tensor(Y_states[idx], dtype=torch.float32, device=DEVICE),
            )

    for _epoch in range(epochs):
        model.train()
        for xb, ab, yb in batchify(train_idx):
            pred = model(xb, ab)
            loss = 1 - F.cosine_similarity(pred, yb, dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    cosines = []
    top1 = []
    top5 = []
    local_top1 = []
    local_top5 = []
    with torch.no_grad():
        for xb, ab, yb in batchify(test_idx, batch_size=256):
            pred = model(xb, ab).cpu().numpy()
            gold = yb.cpu().numpy()
            cos = np.sum(pred * gold, axis=1)
            cosines.extend(cos.tolist())
            sims = pred @ state_bank.T
            ranked = np.argsort(-sims, axis=1)
            gold_rank_target = gold @ state_bank.T
            gold_nearest = np.argmax(gold_rank_target, axis=1)
            top1.extend((ranked[:, 0] == gold_nearest).astype(float).tolist())
            top5.extend([1.0 if gold_nearest[i] in ranked[i, :5] else 0.0 for i in range(len(gold_nearest))])

    for idx in test_idx:
        sig = SIGS[idx]
        with torch.no_grad():
            pred = model(
                torch.tensor(X_states[idx:idx+1], dtype=torch.float32, device=DEVICE),
                torch.tensor(A[idx:idx+1], dtype=torch.long, device=DEVICE),
            ).cpu().numpy()[0]
        candidate_indices = task_to_state_indices[sig]
        local_sims = state_bank[candidate_indices] @ pred
        local_rank = np.argsort(-local_sims)
        gold_global = int(np.argmax(Y_states[idx] @ state_bank.T))
        ranked_globals = [candidate_indices[i] for i in local_rank[:5]]
        local_top1.append(1.0 if ranked_globals[0] == gold_global else 0.0)
        local_top5.append(1.0 if gold_global in ranked_globals else 0.0)

    horizons = [1, 3, 5]
    rollout_scores = defaultdict(list)
    seq_tokens = bundle["seq_tokens"]
    seq_targets = bundle["seq_targets"]
    max_eval = min(30, len(seq_tokens))
    for seq_x, seq_y in zip(seq_tokens[:max_eval], seq_targets[:max_eval]):
        init_state = seq_x[0, : X_states.shape[1]]
        action_indices = [int(np.argmax(tok[X_states.shape[1]:])) for tok in seq_x]
        for horizon in horizons:
            if len(action_indices) < horizon:
                continue
            pred_vec = torch.tensor(init_state[None, :], dtype=torch.float32, device=DEVICE)
            for act_idx in action_indices[:horizon]:
                pred_vec = model(pred_vec, torch.tensor([act_idx], dtype=torch.long, device=DEVICE))
            pred_vec = pred_vec.detach().cpu().numpy()[0]
            gold_vec = seq_y[horizon - 1]
            sim = float(np.dot(pred_vec, gold_vec))
            rollout_scores[f"h{horizon}_cosine"].append(sim)

    return {
        "num_transitions": int(n),
        "state_bank_size": int(state_bank.shape[0]),
        "one_step_cosine": round(float(np.mean(cosines)), 4),
        "one_step_top1_decode_global": round(float(np.mean(top1)), 4),
        "one_step_top5_decode_global": round(float(np.mean(top5)), 4),
        "one_step_top1_decode_local": round(float(np.mean(local_top1)), 4),
        "one_step_top5_decode_local": round(float(np.mean(local_top5)), 4),
        **{k: round(float(np.mean(v)), 4) for k, v in rollout_scores.items()},
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class SequenceMLPControl(nn.Module):
    def __init__(self, token_dim: int, state_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class CausalSequenceTransformer(nn.Module):
    def __init__(self, token_dim: int, state_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, max_len: int = 32):
        super().__init__()
        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, state_dim)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.input_proj(x) + self.pos_embed[:t].unsqueeze(0)
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        padding_mask = None if mask is None else ~mask
        out = self.encoder(h, mask=causal, src_key_padding_mask=padding_mask)
        return F.normalize(self.out_proj(out), dim=-1)


class MinimalLRU(nn.Module):
    def __init__(self, token_dim: int, state_dim: int, hidden: int = 96):
        super().__init__()
        self.hidden = hidden
        self.in_proj = nn.Linear(token_dim, hidden * 2)
        self.log_radius = nn.Parameter(torch.full((hidden,), -0.7))
        self.phase = nn.Parameter(torch.linspace(0.0, math.pi / 2, hidden))
        self.out_proj = nn.Linear(hidden * 2, state_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        device = x.device
        inp = self.in_proj(x)
        inp_c = torch.complex(inp[..., : self.hidden], inp[..., self.hidden :])
        radius = torch.sigmoid(self.log_radius).to(device)
        phase = self.phase.to(device)
        lam = torch.complex(radius * torch.cos(phase), radius * torch.sin(phase))
        h = torch.zeros(b, self.hidden, dtype=torch.cfloat, device=device)
        outputs = []
        for i in range(t):
            if mask is not None:
                valid = mask[:, i].unsqueeze(-1)
                h = torch.where(valid, lam.unsqueeze(0) * h + inp_c[:, i], h)
            else:
                h = lam.unsqueeze(0) * h + inp_c[:, i]
            feat = torch.cat([h.real, h.imag], dim=-1)
            outputs.append(self.out_proj(feat))
        out = torch.stack(outputs, dim=1)
        return F.normalize(out, dim=-1)


def run_sequence_architecture_experiment(bundle: dict, epochs: int = 4) -> dict:
    seq_device = "cpu"
    seq_tokens = bundle["seq_tokens"]
    seq_targets = bundle["seq_targets"]
    seq_sigs = bundle["seq_sigs"]
    state_bank = bundle["state_bank"]
    task_to_state_indices = bundle["task_to_state_indices"]
    state_dim = bundle["state_dim"]
    token_dim = bundle["token_dim"]
    action_dim = len(ACTIONS)

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(seq_tokens))
    split = int(0.85 * len(order))
    train_ids = order[:split]
    test_ids = order[split:]

    def pad_batch(ids):
        max_len = max(len(seq_tokens[i]) for i in ids)
        xb = np.zeros((len(ids), max_len, token_dim), dtype=np.float32)
        yb = np.zeros((len(ids), max_len, state_dim), dtype=np.float32)
        mb = np.zeros((len(ids), max_len), dtype=bool)
        for row, idx in enumerate(ids):
            l = len(seq_tokens[idx])
            xb[row, :l] = seq_tokens[idx]
            yb[row, :l] = seq_targets[idx]
            mb[row, :l] = True
        return (
            torch.tensor(xb, dtype=torch.float32, device=seq_device),
            torch.tensor(yb, dtype=torch.float32, device=seq_device),
            torch.tensor(mb, dtype=torch.bool, device=seq_device),
        )

    def iterate_minibatches(ids, batch_size=24):
        ids = list(ids)
        rng.shuffle(ids)
        for i in range(0, len(ids), batch_size):
            yield pad_batch(ids[i : i + batch_size])

    def evaluate_model(model):
        model.eval()
        cosines, local_top1, local_top5 = [], [], []
        with torch.no_grad():
            for idx in test_ids:
                xb, yb, mb = pad_batch([idx])
                pred = model(xb, mb)[0, : mb[0].sum()].cpu().numpy()
                gold = yb[0, : mb[0].sum()].cpu().numpy()
                cosines.extend(np.sum(pred * gold, axis=1).tolist())
                sig = seq_sigs[idx]
                candidate_indices = task_to_state_indices[sig]
                local_bank = state_bank[candidate_indices]
                for p, g in zip(pred, gold):
                    local_rank = np.argsort(-(local_bank @ p))
                    gold_global = int(np.argmax(g @ state_bank.T))
                    ranked_globals = [candidate_indices[i] for i in local_rank[:5]]
                    local_top1.append(1.0 if ranked_globals[0] == gold_global else 0.0)
                    local_top5.append(1.0 if gold_global in ranked_globals else 0.0)

        rollout = defaultdict(list)
        with torch.no_grad():
            for idx in test_ids[:30]:
                actions = [int(np.argmax(tok[state_dim:])) for tok in seq_tokens[idx]]
                init_state = seq_tokens[idx][0][:state_dim]
                gold_states = seq_targets[idx]
                for horizon in [1, 3, 5]:
                    if len(actions) < horizon:
                        continue
                    history = []
                    current_state = init_state.copy()
                    for step in range(horizon):
                        onehot = np.zeros(action_dim, dtype=np.float32)
                        onehot[actions[step]] = 1.0
                        history.append(np.concatenate([current_state, onehot], axis=0))
                        hx = torch.tensor(np.stack(history)[None, :, :], dtype=torch.float32, device=seq_device)
                        hm = torch.ones((1, hx.shape[1]), dtype=torch.bool, device=seq_device)
                        pred = model(hx, hm)[0, -1].cpu().numpy()
                        current_state = pred
                    rollout[f"h{horizon}_cosine"].append(float(np.dot(current_state, gold_states[horizon - 1])))

        return {
            "one_step_cosine": round(float(np.mean(cosines)), 4),
            "one_step_top1_decode_local": round(float(np.mean(local_top1)), 4),
            "one_step_top5_decode_local": round(float(np.mean(local_top5)), 4),
            **{k: round(float(np.mean(v)), 4) for k, v in rollout.items()},
        }

    specs = {
        "causal_transformer": lambda: CausalSequenceTransformer(token_dim, state_dim),
        "linear_recurrent": lambda: MinimalLRU(token_dim, state_dim),
        "mlp_control": lambda: SequenceMLPControl(token_dim, state_dim),
    }

    results = {"device": seq_device, "num_episodes": len(seq_tokens), "num_transitions": int(sum(len(x) for x in seq_tokens))}
    for name, builder in specs.items():
        model = builder().to(seq_device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        start = time.perf_counter()
        for _ in range(epochs):
            model.train()
            for xb, yb, mb in iterate_minibatches(train_ids):
                pred = model(xb, mb)
                cos = F.cosine_similarity(pred, yb, dim=-1)
                loss = 1 - cos[mb].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        train_seconds = time.perf_counter() - start
        metrics = evaluate_model(model)
        metrics["parameter_count"] = count_parameters(model)
        metrics["training_seconds"] = round(train_seconds, 3)
        results[name] = metrics
    return results


def fetch_arxiv_abstracts(arxiv_ids: List[str]) -> List[dict]:
    joined = ",".join(arxiv_ids)
    url = f"http://export.arxiv.org/api/query?id_list={joined}"
    xml_text = requests.get(url, timeout=60).text
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    docs = []
    for entry in root.findall("a:entry", ns):
        title = entry.findtext("a:title", default="", namespaces=ns).strip().replace("\n", " ")
        summary = entry.findtext("a:summary", default="", namespaces=ns).strip().replace("\n", " ")
        raw_id = entry.findtext("a:id", default="", namespaces=ns).split("/")[-1]
        arxiv_id = raw_id.split("v")[0]
        docs.append({"id": arxiv_id, "title": title, "summary": summary})
    docs.sort(key=lambda x: arxiv_ids.index(x["id"]))
    return docs


def run_workspace_experiment() -> dict:
    docs = fetch_arxiv_abstracts(ARXIV_IDS)
    transcript_chunks: List[str] = []
    full_sizes = []
    bounded_sizes = []
    workspace = {"objective": "survey modular long-horizon language-agent architectures", "papers": []}
    for step, doc in enumerate(docs, start=1):
        note = {
            "id": doc["id"],
            "title": doc["title"],
            "keywords": top_keywords(doc["title"] + " " + doc["summary"], k=5),
            "claim": doc["summary"].split(". ")[0].strip(),
        }
        observation = f"paper {step}: {doc['title']} :: {doc['summary']}"
        action = f"note {step}: {json.dumps(note, sort_keys=True)}"
        transcript_chunks.extend([observation, action])
        workspace["papers"].append(note)
        full_sizes.append(sum(len(x) for x in transcript_chunks))
        bounded_prompt = json.dumps(
            {
                "objective": workspace["objective"],
                "paper_count": len(workspace["papers"]),
                "papers": workspace["papers"],
                "recent_actions": transcript_chunks[-2:],
            },
            sort_keys=True,
        )
        bounded_sizes.append(len(bounded_prompt))

    title_coverage = sum(1 for d in docs if any(p["title"] == d["title"] for p in workspace["papers"])) / len(docs)
    keyword_coverage = sum(1 for p in workspace["papers"] if len(p["keywords"]) >= 3) / len(workspace["papers"])
    cumulative_full = [int(x) for x in np.cumsum(full_sizes)]
    cumulative_bounded = [int(x) for x in np.cumsum(bounded_sizes)]
    return {
        "num_documents": len(docs),
        "steps": list(range(1, len(docs) + 1)),
        "full_prompt_chars_per_step": [int(x) for x in full_sizes],
        "bounded_prompt_chars_per_step": [int(x) for x in bounded_sizes],
        "cumulative_full_prompt_chars_per_step": cumulative_full,
        "cumulative_bounded_prompt_chars_per_step": cumulative_bounded,
        "final_full_prompt_chars": int(full_sizes[-1]),
        "final_bounded_prompt_chars": int(bounded_sizes[-1]),
        "cumulative_full_chars_processed": int(sum(full_sizes)),
        "cumulative_bounded_chars_processed": int(sum(bounded_sizes)),
        "relative_reduction": round(1 - (sum(bounded_sizes) / sum(full_sizes)), 4),
        "title_coverage": round(float(title_coverage), 4),
        "keyword_coverage": round(float(keyword_coverage), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    started = time.time()
    embedder = SentenceTransformer(DENSE_MODEL_NAME, device=DEVICE)

    retrieval = run_locomo_retrieval(embedder)
    workspace = run_workspace_experiment()
    planning = {}
    for model_name, n_tasks in PLANNER_MODELS:
        planner = CausalLMPlanner(model_name)
        planning[model_name] = run_planner_experiment(planner, n_tasks=n_tasks)
        del planner
        if DEVICE == "mps":
            torch.mps.empty_cache()
    latent_bundle = prepare_latent_bundle(embedder)
    latent = run_latent_dynamics(latent_bundle)
    sequence_models = run_sequence_architecture_experiment(latent_bundle)

    results = {
        "seed": SEED,
        "device": DEVICE,
        "models": {
            "dense_encoder": DENSE_MODEL_NAME,
            "planner_lms": [m for m, _ in PLANNER_MODELS],
        },
        "locomo_retrieval": retrieval,
        "workspace": workspace,
        "planner": planning,
        "latent_world_model": latent,
        "sequence_models": sequence_models,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    results = sanitize(results)
    out_path = RESULTS / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
