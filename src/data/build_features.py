import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config.settings import load_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("deepbehavior.data.build_features")


def build_sequential_features(
    input_parquet: Path | None = None,
    output_dir: Path | None = None,
    min_transactions: int | None = None,
    max_len: int | None = None,
    padding: str | None = None,
    config_path: Path | str | None = None,
) -> dict:
    """Build Gold sequential user features (Nxmax_len) from Silver transactions."""
    cfg = load_settings(config_path)
    in_parquet = (
        Path(input_parquet) if input_parquet is not None else cfg.paths.interim_parquet
    )
    out_dir = Path(output_dir) if output_dir is not None else cfg.paths.processed_dir
    min_tx = (
        min_transactions if min_transactions is not None else cfg.data.min_transactions
    )
    seq_max_len = max_len if max_len is not None else cfg.data.max_len
    pad_mode = (padding if padding is not None else cfg.data.padding).lower().strip()

    sequences_path = (
        cfg.paths.sequences_file
        if output_dir is None
        else out_dir / "sequences.npz"
    )
    metadata_path = (
        cfg.paths.metadata_file
        if output_dir is None
        else out_dir / "metadata.json"
    )

    logger.info("Loading silver transactions from %s", in_parquet)
    df = pd.read_parquet(in_parquet)

    # Filter cold-start customer IDs with fewer than min_tx events to ensure valid inter-purchase deltas.
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_tx].index
    df = df[df["user_id"].isin(valid_users)].copy()

    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    # Compute causal inter-purchase delta \Delta t_{u,k} = (t_{u,k} - t_{u,k-1}) in days; initial event \Delta t_{u,0} = 0.
    df["time_delta"] = (
        df.groupby("user_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
        .fillna(0.0)
    )

    # Apply log1p variance stabilization prior to z-score standardization across heavy-tailed monetary and temporal domains.
    df["time_delta_log"] = np.log1p(df["time_delta"])
    df["amount_log"] = np.log1p(df["amount"])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[["amount_log", "time_delta_log"]])
    df["amount_scaled"] = scaled_features[:, 0]
    df["time_delta_scaled"] = scaled_features[:, 1]

    # Index 0 is strictly reserved for sequence padding; active category tokens occupy [1, |C|].
    unique_categories = sorted(df["mcc_category"].unique())
    cat_to_idx = {cat: idx + 1 for idx, cat in enumerate(unique_categories)}
    df["cat_idx"] = df["mcc_category"].map(cat_to_idx)
    vocab_size = len(unique_categories) + 1

    grouped = df.groupby("user_id")
    n_users = len(grouped)

    mcc_seqs = np.zeros((n_users, seq_max_len), dtype=np.int32)
    amount_seqs = np.zeros((n_users, seq_max_len), dtype=np.float32)
    time_seqs = np.zeros((n_users, seq_max_len), dtype=np.float32)
    user_ids = np.empty(n_users, dtype="U32")

    # Truncate sequences to the terminal seq_max_len interactions and zero-pad shorter histories.
    for i, (u_id, group) in enumerate(grouped):
        user_ids[i] = u_id

        cats = group["cat_idx"].values[-seq_max_len:]
        amounts = group["amount_scaled"].values[-seq_max_len:]
        times = group["time_delta_scaled"].values[-seq_max_len:]

        seq_len = len(cats)
        if pad_mode == "post":
            mcc_seqs[i, :seq_len] = cats
            amount_seqs[i, :seq_len] = amounts
            time_seqs[i, :seq_len] = times
        else:
            mcc_seqs[i, -seq_len:] = cats
            amount_seqs[i, -seq_len:] = amounts
            time_seqs[i, -seq_len:] = times

    out_dir.mkdir(parents=True, exist_ok=True)
    sequences_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        sequences_path,
        user_ids=user_ids,
        mcc_seqs=mcc_seqs,
        amount_seqs=amount_seqs,
        time_seqs=time_seqs,
    )

    metadata = {
        "max_len": seq_max_len,
        "padding": pad_mode,
        "vocab_size": vocab_size,
        "n_users": n_users,
        "categories": cat_to_idx,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(
        "Built Gold sequential features: n_users=%d, max_len=%d, vocab_size=%d, padding=%s -> %s",
        n_users,
        seq_max_len,
        vocab_size,
        pad_mode,
        sequences_path,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Gold sequential user features (sequences.npz)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    build_sequential_features(config_path=args.config)


if __name__ == "__main__":
    main()
