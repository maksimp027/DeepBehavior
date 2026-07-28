from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class PathsConfig:
    raw_dir: Path = Path("data/raw")
    interim_parquet: Path = Path("data/interim/transactions.parquet")
    processed_dir: Path = Path("data/processed")
    sequences_file: Path = Path("data/processed/sequences.npz")
    metadata_file: Path = Path("data/processed/metadata.json")
    models_dir: Path = Path("models")
    encoder_path: Path = Path("models/encoder.h5")
    autoencoder_path: Path = Path("models/autoencoder.h5")
    history_path: Path = Path("models/training_history.json")
    clusters_parquet_path: Path = Path("data/processed/user_clusters.parquet")
    clusters_csv_path: Path = Path("data/processed/user_clusters.csv")
    embeddings_path: Path = Path("data/processed/user_embeddings.npz")
    cluster_summary_path: Path = Path("models/cluster_summary.json")

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if not isinstance(val, Path):
                val = Path(val)
            if not val.is_absolute():
                val = PROJECT_ROOT / val
            setattr(self, field_name, val)


@dataclass
class DataConfig:
    target_order_status: str = "delivered"
    unknown_category_label: str = "unknown"
    min_transactions: int = 2
    max_len: int = 10
    padding: str = "post"

    def __post_init__(self) -> None:
        self.padding = self.padding.lower().strip()
        if self.padding not in {"post", "pre"}:
            raise ValueError(f"padding must be 'post' or 'pre', got '{self.padding}'")
        if self.min_transactions < 1:
            raise ValueError("min_transactions must be >= 1")
        if self.max_len < 1:
            raise ValueError("max_len must be >= 1")


@dataclass
class ModelConfig:
    seq_len: int = 10
    vocab_size: int = 72
    embedding_dim: int = 16
    latent_dim: int = 16
    encoder_filters: list[int] = field(default_factory=lambda: [32, 64])
    decoder_filters: int = 32
    kernel_size: int = 3
    learning_rate: float = 1e-3
    mcc_loss_weight: float = 1.0
    num_loss_weight: float = 1.0


@dataclass
class TrainConfig:
    validation_split: float = 0.2
    random_seed: int = 42
    batch_size: int = 64
    epochs: int = 30
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0005
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5
    verbose: int = 1


@dataclass
class ClusterConfig:
    algorithm: str = "kmeans"
    n_clusters: int = 5
    random_seed: int = 42
    normalize_embeddings: bool = True
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    dbscan_eps: float = 0.65
    dbscan_min_samples: int = 15

    def __post_init__(self) -> None:
        self.algorithm = self.algorithm.lower().strip()
        if self.algorithm not in {"kmeans", "dbscan"}:
            raise ValueError(
                f"cluster.algorithm must be 'kmeans' or 'dbscan', got '{self.algorithm}'"
            )


@dataclass
class Settings:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load typed Settings from YAML configuration file."""
    resolved_path: Path
    if config_path is not None:
        resolved_path = Path(config_path)
    else:
        env_path = os.getenv("DEEPBEHAVIOR_CONFIG")
        resolved_path = Path(env_path) if env_path else DEFAULT_CONFIG_PATH

    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    raw_cfg: dict[str, Any] = {}
    if resolved_path.exists():
        with open(resolved_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw_cfg = loaded

    paths_cfg = PathsConfig(**raw_cfg.get("paths", {}))
    data_cfg = DataConfig(**raw_cfg.get("data", {}))
    model_cfg = ModelConfig(**raw_cfg.get("model", {}))
    train_cfg = TrainConfig(**raw_cfg.get("train", {}))
    cluster_cfg = ClusterConfig(**raw_cfg.get("cluster", {}))

    return Settings(
        paths=paths_cfg,
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        cluster=cluster_cfg,
    )


settings = load_settings()
