#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ -x "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
else
    PYTHON_BIN="python3"
fi

CONFIG_FILE="${DEEPBEHAVIOR_CONFIG:-config.yaml}"

echo "============================================================"
echo " DeepBehavior End-to-End ML Pipeline"
echo " Python: ${PYTHON_BIN} | Config: ${CONFIG_FILE}"
echo "============================================================"

echo "[1/4] Silver Layer: Extracting Olist transactions..."
"$PYTHON_BIN" -m src.data.extract --config "$CONFIG_FILE"

echo "[2/4] Gold Layer: Building sequential user features..."
"$PYTHON_BIN" -m src.data.build_features --config "$CONFIG_FILE"

echo "[3/4] Model Training: Training 1D-CNN Autoencoder..."
"$PYTHON_BIN" -m src.models.train --config "$CONFIG_FILE"

echo "[4/4] RecSys Clustering: Extracting latent vectors & clustering..."
"$PYTHON_BIN" -m src.recsys.cluster --config "$CONFIG_FILE"

echo "============================================================"
echo " Pipeline completed successfully!"
echo " - Encoder weights: models/encoder.h5"
echo " - Cluster assignments: data/processed/user_clusters.parquet"
echo " - Cluster summary: models/cluster_summary.json"
echo "============================================================"
