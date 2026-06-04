#!/usr/bin/env bash
# Fetch the SkillRouter checkpoints into ./checkpoints/{emb,rank}.
#
# Why this exists: behind the hf-mirror.com mirror the `/api/models/...`
# metadata endpoint returns empty and LFS blobs redirect to Xet storage, so
# `huggingface_hub` (and thus `from_pretrained` / `snapshot_download`) fails
# with LocalEntryNotFound. The plain `resolve/main/<file>` paths work, so we
# fetch them directly.
#
# Two gotchas this script handles:
#   1. A single curl connection to the Xet CDN is throttled to ~40 KB/s and
#      frequently truncates large files mid-stream. We use aria2c with several
#      connections instead.
#   2. aria2c preallocates the full file and, if a segment ultimately fails,
#      leaves it as a zero-hole -- producing a *right-sized but corrupt* file
#      (zeroed RMSNorm weights => the model emits all-zero logits). We download
#      with unlimited retries (--max-tries=0) and then VERIFY that no tensor
#      decoded to all-zeros, retrying the whole file if any did.
#
# Usage:  scripts/fetch_models.sh
# Then:   export SR_EMB_PATH=$PWD/checkpoints/emb SR_RANK_PATH=$PWD/checkpoints/rank
#         HF_HUB_OFFLINE=1 uv run skillrouter route "..."
set -uo pipefail

BASE="${HF_ENDPOINT:-https://huggingface.co}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

command -v aria2c >/dev/null || { echo "error: aria2c is required (brew install aria2)"; exit 1; }

small_files=(
  config.json configuration.json generation_config.json
  model.safetensors.index.json
  tokenizer.json tokenizer_config.json vocab.json merges.txt
  special_tokens_map.json added_tokens.json chat_template.jinja
  modeling_qwen3.py 1_Pooling/config.json
)

# Count tensors that decoded to all-zeros (excluding embeddings) => corruption.
count_zeroed() {
  uv run python - "$1" <<'PY' 2>/dev/null | tail -1
import sys
from safetensors import safe_open
with safe_open(sys.argv[1], "pt") as st:
    print(sum(1 for k in st.keys()
              if "embed" not in k and float(st.get_tensor(k).abs().sum()) == 0))
PY
}

fetch_small() {
  local repo="$1" dest="$2" f
  for f in "${small_files[@]}"; do
    aria2c -x4 -s4 --max-tries=0 --retry-wait=5 --connect-timeout=20 \
           --console-log-level=error --file-allocation=none --allow-overwrite=true \
           -d "$dest" -o "$f" "$BASE/$repo/resolve/main/$f" >/dev/null 2>&1 \
      && echo "  [ok] $f" || echo "  [skip] $f (not found)"
  done
}

fetch_weights() {
  local repo="$1" dest="$2" f="$2/model.safetensors"
  local url="$BASE/$repo/resolve/main/model.safetensors"
  for attempt in 1 2 3 4 5; do
    rm -f "$f" "$f.aria2"
    aria2c -x8 -s8 -k1M --max-tries=0 --retry-wait=5 --connect-timeout=20 \
           --console-log-level=error --file-allocation=none --allow-overwrite=true \
           -d "$dest" -o model.safetensors "$url" >/dev/null 2>&1
    local z; z=$(count_zeroed "$f")
    if [ "$z" = "0" ]; then echo "  [ok] model.safetensors (verified)"; return 0; fi
    echo "  ...$z zeroed tensors, retrying model.safetensors (attempt $attempt)"
  done
  echo "  [FAIL] model.safetensors still corrupt"; return 1
}

for pair in "emb pipizhao/SkillRouter-Embedding-0.6B" "rank pipizhao/SkillRouter-Reranker-0.6B"; do
  set -- $pair; key="$1"; repo="$2"
  echo "=== $repo -> checkpoints/$key ==="
  mkdir -p "checkpoints/$key"
  fetch_small "$repo" "checkpoints/$key"
  fetch_weights "$repo" "checkpoints/$key"
done
echo "done. export SR_EMB_PATH=\$PWD/checkpoints/emb SR_RANK_PATH=\$PWD/checkpoints/rank (and HF_HUB_OFFLINE=1)."
