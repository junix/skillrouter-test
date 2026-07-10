install_bin := home_directory() / "sync" / "bin"

default:
    @just --list

sync:
    uv sync

build: sync
    uv build

test:
    uv run pytest

# encoder-only retrieval over the built-in pool
retrieve query:
    uv run skillrouter retrieve "{{query}}"

# full retrieve -> rerank pipeline
route query:
    uv run skillrouter route "{{query}}"

# run all built-in sample queries
demo:
    uv run skillrouter demo

# fetch checkpoints via curl (works behind broken hf-mirror.com)
fetch:
    scripts/fetch_models.sh

# run demo against locally-fetched checkpoints (offline, no Hub calls)
demo-local:
    SR_EMB_PATH={{justfile_directory()}}/checkpoints/emb \
    SR_RANK_PATH={{justfile_directory()}}/checkpoints/rank \
    HF_HUB_OFFLINE=1 uv run skillrouter demo

help:
    uv run skillrouter --help

install: build
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{ install_bin }}"
    cat > "{{ install_bin }}/skillrouter" << 'EOF'
    #!/bin/bash
    cd "{{ justfile_directory() }}" && uv run skillrouter "$@"
    EOF
    chmod +x "{{ install_bin }}/skillrouter"
    echo "Installed skillrouter to {{ install_bin }}"
