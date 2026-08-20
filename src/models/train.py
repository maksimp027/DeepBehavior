import argparse
import json
import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src.config.settings import load_settings
from src.models.autoencoder import build_1d_cnn_autoencoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("deepbehavior.models.train")


def load_sequence_dataset(sequences_path: Path) -> dict[str, np.ndarray]:
    """Load preprocessed sequential user arrays from .npz archive."""
    if not sequences_path.exists():
        raise FileNotFoundError(
            f"Sequences file not found at {sequences_path}. Run feature engineering first."
        )
    data = np.load(sequences_path)
    return {
        "user_ids": data["user_ids"],
        "mcc_seqs": data["mcc_seqs"].astype(np.int32),
        "amount_seqs": data["amount_seqs"].astype(np.float32),
        "time_seqs": data["time_seqs"].astype(np.float32),
    }


def prepare_inputs_and_targets(
    mcc_seqs: np.ndarray,
    amount_seqs: np.ndarray,
    time_seqs: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Format model input tensors and reconstruction targets."""
    # Expand scalar sequences to (N, T, 1) and stack continuous targets into (N, T, 2) for joint MSE supervision.
    inputs = {
        "mcc_input": mcc_seqs,
        "amount_input": np.expand_dims(amount_seqs, axis=-1),
        "time_input": np.expand_dims(time_seqs, axis=-1),
    }
    targets = {
        "mcc_reconstruction": mcc_seqs,
        "num_reconstruction": np.stack([amount_seqs, time_seqs], axis=-1),
    }
    return inputs, targets


def train_model(config_path: Path | str | None = None) -> dict[str, list[float]]:
    """Train the 1D-CNN Autoencoder and save the trained Encoder."""
    cfg = load_settings(config_path)

    np.random.seed(cfg.train.random_seed)
    tf.random.set_seed(cfg.train.random_seed)

    logger.info("Loading sequence dataset from %s", cfg.paths.sequences_file)
    dataset = load_sequence_dataset(cfg.paths.sequences_file)
    mcc_seqs = dataset["mcc_seqs"]
    amount_seqs = dataset["amount_seqs"]
    time_seqs = dataset["time_seqs"]

    # Ensure embedding lookup table upper bound V strictly covers the maximum observed category token ID plus padding index 0.
    n_samples, seq_len = mcc_seqs.shape
    vocab_size = max(cfg.model.vocab_size, int(mcc_seqs.max()) + 1)

    if cfg.paths.metadata_file.exists():
        with open(cfg.paths.metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            vocab_size = max(vocab_size, int(meta.get("vocab_size", vocab_size)))

    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=cfg.train.validation_split,
        random_state=cfg.train.random_seed,
        shuffle=True,
    )

    train_inputs, train_targets = prepare_inputs_and_targets(
        mcc_seqs[train_idx], amount_seqs[train_idx], time_seqs[train_idx]
    )
    val_inputs, val_targets = prepare_inputs_and_targets(
        mcc_seqs[val_idx], amount_seqs[val_idx], time_seqs[val_idx]
    )

    logger.info(
        "Dataset split: train=%d users, val=%d users (seq_len=%d, vocab_size=%d)",
        len(train_idx),
        len(val_idx),
        seq_len,
        vocab_size,
    )

    encoder, autoencoder = build_1d_cnn_autoencoder(
        seq_len=seq_len,
        vocab_size=vocab_size,
        embedding_dim=cfg.model.embedding_dim,
        latent_dim=cfg.model.latent_dim,
        encoder_filters=cfg.model.encoder_filters,
        decoder_filters=cfg.model.decoder_filters,
        kernel_size=cfg.model.kernel_size,
        learning_rate=cfg.model.learning_rate,
        mcc_loss_weight=cfg.model.mcc_loss_weight,
        num_loss_weight=cfg.model.num_loss_weight,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.train.early_stopping_patience,
            min_delta=cfg.train.early_stopping_min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.train.reduce_lr_factor,
            patience=cfg.train.reduce_lr_patience,
            min_lr=cfg.train.min_lr,
            verbose=1,
        ),
    ]

    logger.info(
        "Starting training for up to %d epochs (batch_size=%d)...",
        cfg.train.epochs,
        cfg.train.batch_size,
    )
    history = autoencoder.fit(
        x=train_inputs,
        y=train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        callbacks=callbacks,
        verbose=cfg.train.verbose,
    )

    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.encoder_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.autoencoder_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.history_path.parent.mkdir(parents=True, exist_ok=True)

    encoder.save(cfg.paths.encoder_path)
    autoencoder.save(cfg.paths.autoencoder_path)

    history_dict = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    with open(cfg.paths.history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)

    best_val_loss = min(history_dict.get("val_loss", [0.0]))
    logger.info(
        "Training complete! Best val_loss=%.4f | Saved Encoder to %s",
        best_val_loss,
        cfg.paths.encoder_path,
    )

    return history_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the 1D-CNN Autoencoder and save Encoder weights"
