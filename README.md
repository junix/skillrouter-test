# skillrouter-test

A small CLI to exercise the **SkillRouter** retrieve-and-rerank pipeline from
the paper *SkillRouter: Skill Routing for LLM Agents at Scale*
([arXiv:2603.22455](https://arxiv.org/abs/2603.22455)).

It loads the two released 0.6B checkpoints and lets you route a task query
against a pool of skills:

| Stage    | Model                                  | Role                              |
|----------|----------------------------------------|-----------------------------------|
| encoder  | `pipizhao/SkillRouter-Embedding-0.6B`  | retrieve top-K from the full pool |
| reranker | `pipizhao/SkillRouter-Reranker-0.6B`   | re-score the top-K with full text |

## Setup

```bash
cd ~/projects/models/skillrouter
uv sync          # installs torch + transformers (large, one-time)
```

Weights download from Hugging Face on first run. To use local checkpoints
instead, point these at a directory:

```bash
export SR_EMB_PATH=/path/to/SkillRouter-Embedding-0.6B
export SR_RANK_PATH=/path/to/SkillRouter-Reranker-0.6B
```

### Behind hf-mirror.com (China)

The `hf-mirror.com` mirror is **broken for these repos**: its `/api/models/...`
metadata endpoint returns empty and LFS blobs redirect to Xet storage, so
`from_pretrained` / `snapshot_download` fail with `LocalEntryNotFoundError`.
The plain `resolve/main/<file>` paths *do* work, so fetch with curl and load
from disk:

```bash
scripts/fetch_models.sh          # curls files into ./checkpoints/{emb,rank}
export SR_EMB_PATH=$PWD/checkpoints/emb
export SR_RANK_PATH=$PWD/checkpoints/rank
export HF_HUB_OFFLINE=1           # don't let transformers re-check the Hub
uv run skillrouter route "..."
```

`just demo-local` wraps these env vars for you.

Device is auto-detected (`cuda` > `mps` > `cpu`); override with `--device`.

## Usage

```bash
# full pipeline on the built-in skill pool
uv run skillrouter route "transcribe a local lecture recording with word timings"

# encoder-only retrieval
uv run skillrouter retrieve "set up a PR workflow with required CI checks" --top-k 5

# rerank an explicit candidate set
uv run skillrouter rerank "rewrite git history" --skills my_skills.json

# run every built-in sample query end-to-end
uv run skillrouter demo
```

### Custom skill pool

Pass `--skills file.json` where the file is a JSON array:

```json
[
  {"name": "speech-to-text", "description": "Local Whisper transcription.", "body": "Runs faster-whisper ..."},
  {"name": "git-feature-branch", "description": "Open a PR with CI checks.", "body": "Trunk-based flow ..."}
]
```

The built-in pool (`sample_data.py`) deliberately contains near-duplicate skills
(three transcription variants, two git tools) to reproduce the homogeneous-pool
setting the paper targets — where metadata alone is ambiguous and the full skill
body is the deciding signal.

## Notes

- The encoder uses last-token pooling + L2 norm with the paper's
  `Instruct: ... \nQuery:` prefix; documents are flattened as
  `name | description | body`.
- The reranker score is `logit("yes") − logit("no")` at the final position,
  using the Qwen reranker chat template.
- This is an evaluation/inspection harness, not the official training code
  (see <https://github.com/zhengyanzhao1997/SkillRouter>).
