import argparse
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize
import tensorflow as tf

from src.config.settings import load_settings
from src.models.autoencoder import build_1d_cnn_autoencoder
from src.models.train import load_sequence_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("deepbehavior.recsys.cluster")


def load_trained_encoder(
    encoder_path: Path,
    seq_len: int,
    vocab_size: int,
    config_path: Path | str | None = None,
) -> tf.keras.Model:
    """Load trained SequenceEncoder model or weights from H5 file."""
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Trained encoder not found at {encoder_path}. Run src/models/train.py first."
        )
    try:
        encoder = tf.keras.models.load_model(encoder_path, compile=False)
        return encoder
    except Exception as exc:
        logger.warning(
            "Direct model load failed (%s); falling back to weight loading.", exc
        )
        cfg = load_settings(config_path)
        encoder, _ = build_1d_cnn_autoencoder(
            seq_len=seq_len,
            vocab_size=vocab_size,
            embedding_dim=cfg.model.embedding_dim,
            latent_dim=cfg.model.latent_dim,
            encoder_filters=cfg.model.encoder_filters,
            decoder_filters=cfg.model.decoder_filters,
            kernel_size=cfg.model.kernel_size,
        )
        encoder.load_weights(encoder_path)
        return encoder


def extract_user_embeddings(
    encoder: tf.keras.Model,
    mcc_seqs: np.ndarray,
    amount_seqs: np.ndarray,
    time_seqs: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Pass all user sequences through the Encoder to extract latent vectors."""
    inputs = {
        "mcc_input": mcc_seqs,
        "amount_input": np.expand_dims(amount_seqs, axis=-1),
        "time_input": np.expand_dims(time_seqs, axis=-1),
    }
    embeddings = encoder.predict(inputs, batch_size=batch_size, verbose=0)
    return np.asarray(embeddings, dtype=np.float32)


def run_clustering(
    config_path: Path | str | None = None,
    algorithm_override: str | None = None,
    n_clusters_override: int | None = None,
) -> dict[str, Any]:
    """Extract 16-dim user embeddings and cluster behavioral profiles."""
    cfg = load_settings(config_path)
    algorithm = (
        algorithm_override.lower().strip()
        if algorithm_override is not None
        else cfg.cluster.algorithm
    )
    n_clusters = (
        n_clusters_override
        if n_clusters_override is not None
        else cfg.cluster.n_clusters
    )

    logger.info("Loading sequence dataset from %s", cfg.paths.sequences_file)
    dataset = load_sequence_dataset(cfg.paths.sequences_file)
    user_ids = dataset["user_ids"]
    mcc_seqs = dataset["mcc_seqs"]
    amount_seqs = dataset["amount_seqs"]
    time_seqs = dataset["time_seqs"]

    _, seq_len = mcc_seqs.shape
    vocab_size = max(cfg.model.vocab_size, int(mcc_seqs.max()) + 1)
    idx_to_cat: dict[int, str] = {}
    if cfg.paths.metadata_file.exists():
        with open(cfg.paths.metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            vocab_size = max(vocab_size, int(meta.get("vocab_size", vocab_size)))
            cat_map = meta.get("categories", {})
            idx_to_cat = {int(v): str(k) for k, v in cat_map.items()}

    logger.info("Loading trained Encoder from %s", cfg.paths.encoder_path)
    encoder = load_trained_encoder(
        cfg.paths.encoder_path,
        seq_len=seq_len,
        vocab_size=vocab_size,
        config_path=config_path,
    )

    logger.info("Extracting %d-dim latent vectors for %d users...", cfg.model.latent_dim, len(user_ids))
    raw_embeddings = extract_user_embeddings(
        encoder,
        mcc_seqs=mcc_seqs,
        amount_seqs=amount_seqs,
        time_seqs=time_seqs,
        batch_size=max(cfg.train.batch_size, 256),
    )

    # Project latent vectors onto the unit L2 hypersphere so Euclidean clustering optimizes cosine similarity.
    if cfg.cluster.normalize_embeddings:
        features_for_clustering = normalize(raw_embeddings, norm="l2")
    else:
        features_for_clustering = raw_embeddings

    logger.info("Clustering user embeddings using algorithm='%s'...", algorithm)
    if algorithm == "kmeans":
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=cfg.cluster.random_seed,
            n_init=cfg.cluster.kmeans_n_init,
            max_iter=cfg.cluster.kmeans_max_iter,
        )
        cluster_labels = clusterer.fit_predict(features_for_clustering)
    elif algorithm == "dbscan":
        clusterer = DBSCAN(
            eps=cfg.cluster.dbscan_eps,
            min_samples=cfg.cluster.dbscan_min_samples,
        )
        cluster_labels = clusterer.fit_predict(features_for_clustering)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    unique_labels = set(cluster_labels.tolist())
    non_noise_labels = [lbl for lbl in unique_labels if lbl != -1]
    silhouette = None
    davies_bouldin = None
    if len(non_noise_labels) >= 2:
        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) > len(non_noise_labels):
            silhouette = float(
                silhouette_score(
                    features_for_clustering[valid_mask],
                    cluster_labels[valid_mask],
                )
            )
            davies_bouldin = float(
                davies_bouldin_score(
                    features_for_clustering[valid_mask],
                    cluster_labels[valid_mask],
                )
            )

    results_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "cluster_id": cluster_labels.astype(np.int32),
        }
    )
    for dim_idx in range(raw_embeddings.shape[1]):
        results_df[f"latent_{dim_idx}"] = raw_embeddings[:, dim_idx]

    cfg.paths.clusters_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.clusters_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.cluster_summary_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_parquet(cfg.paths.clusters_parquet_path, engine="pyarrow", index=False)
    results_df[["user_id", "cluster_id"]].to_csv(cfg.paths.clusters_csv_path, index=False)

    np.savez_compressed(
        cfg.paths.embeddings_path,
        user_ids=user_ids,
        embeddings=raw_embeddings,
        cluster_ids=cluster_labels.astype(np.int32),
    )

    # Exclude zero-padding tokens (cat_id == 0) when computing empirical category frequencies per cluster.
    cluster_profiles: dict[str, Any] = {}
    for cid in sorted(unique_labels):
        mask = cluster_labels == cid
        cluster_mcc = mcc_seqs[mask]
        non_zero_cats = cluster_mcc[cluster_mcc > 0]
        top_cats: list[dict[str, Any]] = []
        if len(non_zero_cats) > 0:
            vals, counts = np.unique(non_zero_cats, return_counts=True)
            top_order = np.argsort(-counts)[:5]
            for idx_pos in top_order:
                cat_id = int(vals[idx_pos])
                top_cats.append(
                    {
                        "category": idx_to_cat.get(cat_id, f"cat_{cat_id}"),
                        "count": int(counts[idx_pos]),
                    }
                )

        cluster_profiles[str(cid)] = {
            "user_count": int(np.sum(mask)),
            "user_share_pct": round(float(np.mean(mask) * 100.0), 2),
            "top_categories": top_cats,
        }

    summary = {
        "algorithm": algorithm,
        "n_users": int(len(user_ids)),
        "latent_dim": int(raw_embeddings.shape[1]),
        "n_clusters_found": len(non_noise_labels),
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
        "clusters": cluster_profiles,
    }

    with open(cfg.paths.cluster_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        "Clustering complete! Found %d clusters (silhouette=%s) | Saved assignments to %s and %s",
        len(non_noise_labels),
        f"{silhouette:.4f}" if silhouette is not None else "N/A",
        cfg.paths.clusters_parquet_path,
        cfg.paths.clusters_csv_path,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 16-dim user latent vectors and cluster users"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["kmeans", "dbscan"],
        help="Override clustering algorithm from config",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Override number of clusters for KMeans",
    )
    args = parser.parse_args()
    run_clustering(
        config_path=args.config,
        algorithm_override=args.algorithm,
        n_clusters_override=args.n_clusters,
    )


if __name__ == "__main__":
    main()
