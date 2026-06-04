#!/usr/bin/env bash
# Fetch the SkillRouter checkpoints into ./checkpoints/{emb,rank}.
#
# Why this exists: behind the hf-mirror.com mirror the `/api/models/...`
# metadata endpoint returns empty and LFS blobs redirect to Xet storage, so
# `huggingface_hub` (and thus `from_pretrained`) fails with LocalEntryNotFound.
# The plain `resolve/main/<file>` paths work, so we curl them directly.
#
# The Xet CDN also tends to truncate the big `model.safetensors` mid-stream
# (curl still exits 200 with a partial file), so weights are downloaded with
# resume (`-C -`) and retried until the byte count matches the size declared
# in the safetensors header.
#
# Usage:  scripts/fetch_models.sh
# Then:   export SR_EMB_PATH=$PWD/checkpoints/emb SR_RANK_PATH=$PWD/checkpoints/rank
#         HF_HUB_OFFLINE=1 uv run skillrouter route "..."
set -uo pipefail

BASE="${HF_ENDPOINT:-https://huggingface.co}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

small_files=(
  config.json configuration.json generation_config.json
  model.safetensors.index.json
  tokenizer.json tokenizer_config.json vocab.json merges.txt
  special_tokens_map.json added_tokens.json chat_template.jinja
  modeling_qwen3.py 1_Pooling/config.json
)

size_of() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0; }

# Declared payload size from a safetensors header: 8-byte LE length + header + data.
declared_size() {
  python3 - "$1" <<'PY'
import json, struct, sys
with open(sys.argv[1], 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
end = max((v['data_offsets'][1] for k, v in hdr.items() if k != '__metadata__'), default=0)
print(8 + n + end)
PY
}

fetch_small() {
  local repo="$1" dest="$2" f code
  for f in "${small_files[@]}"; do
    code=$(curl -sS -L -m 600 --create-dirs --retry 5 --retry-all-errors \
                -o "$dest/$f" -w "%{http_code}" "$BASE/$repo/resolve/main/$f" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then echo "  [ok] $f"; else rm -f "$dest/$f"; fi
  done
}

fetch_weights() {
  local repo="$1" dest="$2"
  local out="$dest/model.safetensors"
  local url="$BASE/$repo/resolve/main/model.safetensors"
  mkdir -p "$dest"
  for attempt in $(seq 1 60); do
    curl -sS -L -C - -m 300 --retry 5 --retry-all-errors -o "$out" "$url" 2>/dev/null || true
    local cur; cur=$(size_of "$out")
    local tgt; tgt=$(declared_size "$out" 2>/dev/null || echo 0)
    if [ "$tgt" -gt 0 ] && [ "$cur" -ge "$tgt" ]; then
      echo "  [ok] model.safetensors ($cur bytes)"; return 0
    fi
    echo "  ...resuming model.safetensors ($cur/${tgt:-?} bytes, attempt $attempt)"
    sleep 1
  done
  echo "  [FAIL] model.safetensors did not complete"; return 1
}

for pair in "emb pipizhao/SkillRouter-Embedding-0.6B" "rank pipizhao/SkillRouter-Reranker-0.6B"; do
  set -- $pair; key="$1"; repo="$2"
  echo "=== $repo -> checkpoints/$key ==="
  fetch_small "$repo" "checkpoints/$key"
  fetch_weights "$repo" "checkpoints/$key"
done
echo "done. set SR_EMB_PATH=\$PWD/checkpoints/emb SR_RANK_PATH=\$PWD/checkpoints/rank (and HF_HUB_OFFLINE=1)."
