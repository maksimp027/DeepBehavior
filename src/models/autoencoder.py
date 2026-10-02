from collections.abc import Sequence
import tensorflow as tf
from tensorflow.keras import Model, layers

from src.config.settings import settings


def build_1d_cnn_autoencoder(
    seq_len: int | None = None,
    vocab_size: int | None = None,
    embedding_dim: int | None = None,
    latent_dim: int | None = None,
    encoder_filters: Sequence[int] | None = None,
    decoder_filters: int | None = None,
    kernel_size: int | None = None,
    learning_rate: float | None = None,
    mcc_loss_weight: float | None = None,
    num_loss_weight: float | None = None,
) -> tuple[Model, Model]:
    """Build Multi-Input 1D-CNN Sequence Encoder and Autoencoder models."""
    m_cfg = settings.model
    seq_len = seq_len if seq_len is not None else m_cfg.seq_len
    vocab_size = vocab_size if vocab_size is not None else m_cfg.vocab_size
    embedding_dim = (
        embedding_dim if embedding_dim is not None else m_cfg.embedding_dim
    )
    latent_dim = latent_dim if latent_dim is not None else m_cfg.latent_dim
    enc_filters = (
        list(encoder_filters)
        if encoder_filters is not None
        else list(m_cfg.encoder_filters)
    )
    dec_filters = (
        decoder_filters if decoder_filters is not None else m_cfg.decoder_filters
    )
    k_size = kernel_size if kernel_size is not None else m_cfg.kernel_size
    lr = learning_rate if learning_rate is not None else m_cfg.learning_rate
    w_mcc = (
        mcc_loss_weight if mcc_loss_weight is not None else m_cfg.mcc_loss_weight
    )
    w_num = (
        num_loss_weight if num_loss_weight is not None else m_cfg.num_loss_weight
    )

    mcc_in = layers.Input(
        shape=(seq_len,), dtype=tf.int32, name="mcc_input"
    )
    amount_in = layers.Input(
        shape=(seq_len, 1), dtype=tf.float32, name="amount_input"
    )
    time_in = layers.Input(
        shape=(seq_len, 1), dtype=tf.float32, name="time_input"
    )

    # Project discrete category tokens into R^{T x d_emb} and fuse with continuous scalar streams into R^{T x (d_emb + 2)}.
    mcc_emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="category_embedding",
    )(mcc_in)

    x = layers.Concatenate(
        axis=-1,
        name="feature_fusion",
    )([mcc_emb, amount_in, time_in])

    # Extract local temporal motifs via length-preserving 1D convolutions (padding='same') and collapse across time into R^{d_latent}.
    for filters in enc_filters:
        x = layers.Conv1D(
            filters=filters,
            kernel_size=k_size,
            activation="relu",
            padding="same",
        )(x)
        x = layers.BatchNormalization()(x)

    x = layers.GlobalMaxPooling1D()(x)
    latent = layers.Dense(
        latent_dim, activation="relu", name="latent_embedding"
    )(x)

    encoder_model = Model(
        inputs=[mcc_in, amount_in, time_in],
        outputs=latent,
        name="SequenceEncoder",
    )

    # Unfold bottleneck representation back to temporal grid (T, C_last) for multi-task sequence reconstruction.
    last_enc_filters = enc_filters[-1] if enc_filters else 64
    d = layers.Dense(seq_len * last_enc_filters, activation="relu")(latent)
    d = layers.Reshape((seq_len, last_enc_filters))(d)

    d = layers.Conv1D(
        filters=dec_filters,
        kernel_size=k_size,
        padding="same",
        activation="relu",
    )(d)
    d = layers.BatchNormalization()(d)

    # Multi-task output heads: categorical token distribution over V classes and 2D continuous trajectory [amount, time_delta].
    mcc_out = layers.Dense(
        vocab_size, activation="softmax", name="mcc_reconstruction"
    )(d)

    num_out = layers.Dense(
        2, activation="linear", name="num_reconstruction"
    )(d)

    autoencoder_model = Model(
        inputs=[mcc_in, amount_in, time_in],
        outputs=[mcc_out, num_out],
        name="1D_CNN_Autoencoder",
    )

    # Joint composite objective: L = w_mcc * CE(y_mcc, \hat{y}_mcc) + w_num * MSE(y_num, \hat{y}_num).
    autoencoder_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "mcc_reconstruction": "sparse_categorical_crossentropy",
            "num_reconstruction": "mse",
        },
        loss_weights={
            "mcc_reconstruction": w_mcc,
            "num_reconstruction": w_num,
        },
        metrics={
            "mcc_reconstruction": ["sparse_categorical_accuracy"],
            "num_reconstruction": ["mae"],
        },
    )

    return encoder_model, autoencoder_model


if __name__ == "__main__":
    encoder, autoencoder = build_1d_cnn_autoencoder()
    encoder.summary()
    autoencoder.summary()