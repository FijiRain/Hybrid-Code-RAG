#!/usr/bin/env bash
set -e

TARGET_DIR="data/raw/vllm-0.10.1"

if [ -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR already exists. Skipping download."
    exit 0
fi

echo "Fetching sample codebase (vLLM v0.10.1) from official source..."
mkdir -p data/raw

git clone --depth 1 --branch v0.10.1 https://github.com/vllm-project/vllm.git "$TARGET_DIR"

rm -rf "$TARGET_DIR/.git"

echo "Done! Target codebase is ready in $TARGET_DIR."
