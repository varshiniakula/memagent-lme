#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
cd data
echo "Downloading LongMemEval cleaned files..."
wget -q --show-progress https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget -q --show-progress https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
wget -q --show-progress https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
echo "Done."
