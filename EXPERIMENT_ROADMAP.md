# Stronger MacBook-Feasible Experiments for the LLM + World-Model Paper

Target machine assumptions:
- Apple M1 Pro
- 16 GB unified memory
- Python 3.11
- local packages already present: `torch`, `transformers`, `sentence_transformers`, `sklearn`, `datasets`
- goal: materially stronger than the current toy experiments without multi-GB checkpoints or long training jobs

This roadmap focuses on experiments that can be run locally on Apple Silicon with modest downloads and that map cleanly onto the paper's architecture claims: latent prediction, persistent memory, bounded workspace, and verifier-backed planning.

---

## Recommended shortlist

If you only add three experiments, add these:

1. `Real embedding memory benchmark` using `sentence-transformers/all-MiniLM-L6-v2`
2. `Constrained planner with a small open-weight LM` using `distilgpt2` or `gpt2` plus symbolic action masking / BFS verifier
3. `Latent dynamics rollout` using sentence embeddings as compact state and a small MLP transition model on a text-gridworld or ALFWorld-style abstracted traces

These three are coherent together: experiment 1 validates the memory layer, experiment 2 validates verified planning over explicit state, and experiment 3 gives you a real but still laptop-feasible latent world-model story.

---

## Experiment 1: Real embedding retrieval on conversational memory

### Claim it supports
Hybrid persistent memory beats lexical-only retrieval when queries are paraphrastic, noisy, or temporally ambiguous.

### Stronger version of the current toy retrieval test
Replace character n-gram similarity with a real sentence embedding model and evaluate on a benchmark-like conversational memory retrieval task.

### Exact model / packages
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Optional stronger embedding baseline: `BAAI/bge-small-en-v1.5`
- Retrieval stack:
  - lexical: `sklearn.feature_extraction.text.TfidfVectorizer`
  - dense: `sentence_transformers.SentenceTransformer`
  - fusion: reciprocal rank fusion or weighted score fusion
- Optional reranker if needed later: `cross-encoder/ms-marco-MiniLM-L-6-v2` (not required for first pass)

### Download / memory profile
- `all-MiniLM-L6-v2`: roughly 80-100 MB
- `bge-small-en-v1.5`: roughly 130-150 MB
- Fits easily in 16 GB RAM; inference on CPU or MPS is fine

### Dataset options
Best local-first options:
1. Build a realistic benchmark from dialogue datasets already available through `datasets`, then derive memory queries.
   - Candidate sources: `daily_dialog`, `blended_skill_talk`, `personachat`
   - Convert each conversation into memory items such as preferences, commitments, locations, plans, and facts.
   - Create paraphrastic queries manually or semi-automatically.
2. If you want less preprocessing, use `personachat` because persona statements naturally form memory facts.

### Concrete protocol
For each conversation:
- Extract 5-20 memory records with type labels: `fact`, `preference`, `plan`, `experience`
- Create 1-3 queries per target memory:
  - exact wording
  - paraphrase
  - temporally ambiguous or distractor-heavy variant
- Compare:
  1. TF-IDF only
  2. dense embeddings only
  3. hybrid dense + lexical
  4. hybrid + temporal prior

### Metrics
- Hits@1
- Hits@5
- MRR
- nDCG@10
- Retrieval latency per query
- Breakdown by query type: exact, paraphrase, distractor-heavy

### Expected output shape
You should expect:
- dense retrieval to beat lexical retrieval on paraphrases
- lexical retrieval to remain competitive on exact-match facts
- hybrid fusion to be best overall
- temporal priors to help mostly on repeated entities or evolving plans

### Estimated runtime
- Data prep for 500-2,000 memory items: under 10 min once scripted
- Embedding inference for a few thousand texts on M1 Pro: about 2-10 min depending on batching and model
- Full evaluation: usually under 15 min total

### Implementation notes
- Keep everything in a single JSONL file with fields like:
  - `memory_id`, `dialog_id`, `type`, `timestamp`, `text`, `entity`, `query_text`, `query_type`, `target_memory_id`
- Cache embeddings to `.npy`
- Normalize vectors and use cosine similarity
- You do not need FAISS; plain NumPy / sklearn nearest-neighbor search is enough at this scale
- A very publishable ablation is: dense only vs hybrid vs hybrid+time

### Why this is good for the paper
It upgrades the memory section from a synthetic proxy to a real embedding-backed retrieval experiment with minimal engineering and almost no download burden.

---

## Experiment 2: Verified planning with a small open-weight language model

### Claim it supports
A small language model can propose actions, but explicit search / verification is what makes long-horizon planning reliable.

### Core idea
Use a tiny or GPT-2-style LM to score or propose actions in a discrete environment, then compare pure greedy LM planning against verifier-backed search over the same world states.

### Exact model / packages
Primary local option:
- `distilgpt2` via `transformers`

Secondary local option if you want slightly stronger text behavior:
- `gpt2`

Optional better small instruct model, still laptop-feasible but requires GGUF + llama.cpp rather than pure transformers:
- `Qwen2.5-0.5B-Instruct-GGUF` q4 or q5 quantization
- or `SmolLM2-360M-Instruct`

### Download / memory profile
- `distilgpt2`: roughly 300-350 MB
- `gpt2`: roughly 500-550 MB
- `Qwen2.5-0.5B` GGUF quantized: roughly 0.4-0.7 GB depending on quantization

### Environment choices
Best low-friction option:
- Text gridworld with keys, doors, obstacles, and inventory constraints

Better but still manageable option:
- Mini text game abstraction with actions from a fixed vocabulary:
  - `go north/south/east/west`
  - `pick key`
  - `open door`
  - `push box`

Do not rely on free-form generation alone. Restrict decoding to a legal action vocabulary.

### Concrete protocol
Construct 200-500 planning problems with:
- hidden dead ends
- need for backtracking
- small inventory dependencies
- irreversible bad actions in some cases

Compare:
1. Greedy LM: choose the highest-probability next action from the legal action list
2. Beam LM: short-horizon beam search over legal actions without explicit state verifier beyond transition legality
3. LM + verifier: LM provides action prior, symbolic planner performs BFS or A* over legal states and goal conditions
4. Pure symbolic search baseline: BFS or A* without LM prior

### Metrics
- success rate
- plan length
- invalid action rate
- search expansions
- wall-clock latency per task
- robustness under perturbed prompts or renamed entities

### Expected output shape
Likely pattern:
- greedy LM will fail when backtracking or delayed rewards are needed
- beam search may improve modestly but still be brittle
- LM + verifier should match or approach symbolic search on success while reducing expansions if the LM prior is informative
- pure symbolic search may be slow but strongest on feasibility guarantees

### Estimated runtime
- Environment generation: seconds
- Running 200-500 tasks with `distilgpt2`: about 10-30 min depending on generation method and beam width
- Faster if you compute one-step action logits only instead of full text continuation

### Implementation notes
Recommended implementation pattern:
- Represent each state symbolically in Python
- Render state as compact text description for the LM
- Score candidate actions by computing next-token or sequence log-probability for each legal action string
- Do not allow arbitrary text generation; action scoring is much more stable and much faster
- Use the symbolic environment as the world model / verifier
- Add one ablation where the LM sees only a text transcript and another where it sees a structured state summary

### Why this is good for the paper
It preserves your core thesis: language is a useful interface and heuristic prior, but robust planning comes from explicit state transition models and verification.

---

## Experiment 3: Latent world-model rollout using real embeddings as compact state

### Claim it supports
Predicting future latent state embeddings can support planning or retrieval without decoding every intermediate step into text.

### Why this matters
This is the cleanest laptop-feasible bridge between the paper's world-model argument and something materially more realistic than BFS on a toy grid.

### Core idea
Treat environment observations as text, embed them into a compact vector space, and train a small transition model:
- input: current state embedding + action embedding
- output: next-state embedding

Then compare planning in latent space versus text-only or symbolic-only alternatives.

### Exact model / packages
- State encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Action encoder: same model or learned embeddings for the finite action set
- Transition model: small PyTorch MLP, for example 2 layers of width 512 or 768
- Optional decoder-free setup: nearest-neighbor decode from predicted latent to a known state bank

### Download / memory profile
- Only the sentence-transformer download is needed beyond existing packages
- Transition model is tiny and trains in memory

### Environment choices
Best option for the paper:
- Textual gridworld / household-task abstraction where each state is rendered as a sentence or short paragraph
  - example: `agent in kitchen; has key; red door locked; goal is apple in pantry`
- Generate trajectories automatically with a symbolic simulator

Alternative if you want a more standard sequential benchmark:
- Use short abstract action traces derived from ALFWorld-style task templates, but keep the actual simulator symbolic and local

### Concrete protocol
1. Generate 10k-50k transitions from your simulator
2. Encode each state text into a dense vector
3. Train transition model to predict next-state embedding from `(state_embedding, action)`
4. Evaluate:
   - cosine similarity to true next-state embedding
   - top-k nearest-state retrieval accuracy
   - multi-step rollout drift over horizons 1, 3, 5, 10
5. Planning comparison:
   - text-only greedy LM planner
   - latent rollout planner using learned transition model and value heuristic
   - symbolic search oracle

### Metrics
- one-step cosine similarity
- top-1 / top-5 nearest-state decoding accuracy
- rollout consistency across horizon length
- task success when planning with predicted latents
- selective decoding count: how often full text decoding is needed

### Expected output shape
You should expect:
- one-step latent prediction to work well on locally smooth transitions
- multi-step rollouts to degrade gradually rather than catastrophically
- latent planning to outperform purely greedy text planning on short-to-medium horizons
- symbolic oracle to remain strongest on exact feasibility

### Estimated runtime
- 10k-20k transitions with MiniLM embeddings: about 5-15 min preprocessing
- MLP training on CPU/MPS: about 5-20 min
- evaluation: a few minutes
- very feasible in under an hour end-to-end

### Implementation notes
- Keep state descriptions templated and compositional; do not use highly varied prose
- Predict normalized embeddings with cosine loss or MSE on normalized vectors
- For planning, use latent rollout to score candidate action sequences of length 3-6
- A strong paper figure is rollout error versus horizon length
- Another strong figure is selective decoding frequency: decode only at the final chosen trajectory, not every intermediate state

### Why this is good for the paper
This gives you an actual latent predictive model rather than only arguing for one. It is still simple enough to run on a MacBook and directly tests the selective-decoding thesis.

---

## Experiment 4: Workspace reconstruction in a real agent loop

### Claim it supports
Bounded workspace reconstruction lowers context cost in real task execution, not just in synthetic transcript accounting.

### Exact model / packages
Local-only option:
- `distilgpt2` or `gpt2` as a weak local language baseline for note synthesis / extraction

Optional materially better API-backed option:
- GPT-4.1-mini or equivalent cheap API model for extraction and report generation

### Task choice
- literature triage over 20-50 abstracts
- or bug-fix / code-edit task with file summaries

### Protocol
Compare:
1. transcript accumulation
2. bounded workspace with explicit structured state (`task`, `findings`, `next_steps`, `open_questions`, `citations`)

Measure:
- cumulative prompt tokens/chars
- wall-clock latency
- task completion quality
- factual consistency of final answer

### Why keep this optional
A local tiny LM is weak enough that task quality may be dominated by model weakness rather than architecture. If you use an API model, this experiment becomes much stronger as a systems result.

---

## Optional API-backed upgrade worth mentioning separately

If the paper can tolerate one optional paid experiment, the best value upgrade is:

### API-assisted planner with the same symbolic world model
Use a small API model as the proposal policy while keeping the verifier, state representation, and evaluation protocol identical.

Compare:
- `distilgpt2` local planner prior
- small API planner prior
- pure symbolic search

This isolates the systems claim cleanly:
- better language priors help
- but verification and explicit state still do the heavy lifting

This is scientifically stronger than replacing the whole stack with an API black box.

---

## Best package / model recommendations

### Embeddings
Best default:
- `sentence-transformers/all-MiniLM-L6-v2`

If you want slightly stronger retrieval and can tolerate a somewhat bigger download:
- `BAAI/bge-small-en-v1.5`

### Local open-weight LMs
Safest with current installed stack:
- `distilgpt2`
- `gpt2`

If you are willing to use `llama.cpp` and GGUF quantization:
- `Qwen2.5-0.5B-Instruct`
- `SmolLM2-360M-Instruct`

### Search / retrieval infrastructure
- `sklearn` is enough for TF-IDF and nearest neighbors at this scale
- plain NumPy cosine search is enough for up to tens of thousands of vectors
- skip FAISS unless the dataset grows substantially

---

## What I would put in the paper first

Priority order for strongest paper-per-hour return:

1. Real embedding memory benchmark with MiniLM or BGE-small
2. Verified planner with `distilgpt2` action scoring + symbolic search
3. Latent embedding transition model on a symbolic text environment
4. Optional API-backed bounded-workspace experiment

This combination gives you:
- one real embedding experiment
- one planner experiment with a small open-weight LM
- one actual latent world-model experiment
- one optional stronger practical systems experiment

---

## Minimal reproducible experiment bundle

If you want a tight artifact that still runs comfortably on the target MacBook, the reproducible bundle should be:

- `exp_memory_real_embeddings.py`
  - loads `personachat` or derived conversational memory set
  - compares TF-IDF vs MiniLM vs hybrid
- `exp_planner_lm_verifier.py`
  - discrete symbolic environment
  - `distilgpt2` scores legal actions
  - compare greedy / beam / verifier-guided search
- `exp_latent_rollout.py`
  - generate text-state trajectories
  - embed states with MiniLM
  - train tiny MLP transition model
  - evaluate one-step and multi-step latent rollout

With cached embeddings and small models, this should remain practical on an M1 Pro 16 GB and require only modest downloads.
