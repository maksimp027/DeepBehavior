# DeepBehavior

DeepBehavior is a sequential behavioral representation learning pipeline designed to project sparse, multi-modal customer order streams from the Brazilian E-Commerce (Olist) dataset into dense low-dimensional latent manifolds. The architecture ingests discrete merchant category taxonomies alongside continuous log-transformed transaction amounts and inter-purchase temporal deltas, encoding longitudinal purchasing dynamics via a multi-input 1D convolutional autoencoder (`SequenceEncoder` and `1D_CNN_Autoencoder`). Extracted 16-dimensional user state vectors support downstream behavioral clustering (`KMeans`, `DBSCAN`) and nearest-neighbor top-$K$ category retrieval.

## Architecture

```text
+---------------------------------------------------------------------------------------------------+
|                                   INPUT TENSORS (Batch x T=10)                                    |
|                                                                                                   |
|  [mcc_input: int32 (B, 10)]       [amount_input: float32 (B, 10, 1)]   [time_input: float32 (B, 10, 1)]
|  Sequential Category Tokens        Standardized log1p(Price)           Standardized log1p(Delta t)|
+--------------+------------------------------------+----------------------------------+------------+
               |                                    |                                  |
               v                                    |                                  |
+------------------------------+                    |                                  |
| Category Embedding Lookup    |                    |                                  |
| Vocab V=72 -> Dim d_emb=16   |                    |                                  |
| Output: float32 (B, 10, 16)  |                    |                                  |
+--------------+---------------+                    |                                  |
               |                                    |                                  |
               +------------------------------------+----------------------------------+
                                                    |
                                                    v
                              +-------------------------------------------+
                              | Channel Concatenation (feature_fusion)    |
                              | Tensor Shape: float32 (B, 10, 18)         |
                              +---------------------+---------------------+
                                                    |
                                                    v
+---------------------------------------------------------------------------------------------------+
|                  TEMPORAL FEATURE EXTRACTION (1D Convolutions & Dilated Kernels)                  |
|                                                                                                   |
|  Conv1D(filters=32, kernel_size=3, stride=1, dilation=1, padding="same", ReLU) + BatchNorm        |
|                                                   |                                               |
|  Conv1D(filters=64, kernel_size=3, stride=1, dilation=1, padding="same", ReLU) + BatchNorm        |
|  (Supports dilated receptive field expansion & residual skip projections across temporal steps)   |
|                                                   |                                               |
|  GlobalMaxPooling1D: Collapse temporal dimension (B, 10, 64) -> (B, 64)                           |
+---------------------------------------------------+-----------------------------------------------+
                                                    |
                                                    v
+---------------------------------------------------------------------------------------------------+
|                                     USER REPRESENTATION LAYER                                     |
|                                                                                                   |
|                     Bottleneck Projection: Dense(latent_dim=16, ReLU)                             |
|                     Latent User State Vector z_u in R^{16} (models/encoder.h5)                    |
+-------------------------+-------------------------------------------------+-----------------------+
                          |                                                 |
                          v                                                 v
+---------------------------------------------------+   +-------------------------------------------+
|       SEQUENCE RECONSTRUCTION HEAD (Decoder)      |   |  TOP-K RETRIEVAL & CLUSTERING HEAD        |
|                                                   |   |                                           |
|  Dense(10 * 64, ReLU) -> Reshape(B, 10, 64)       |   |  L2 Hypersphere Normalization:            |
|  Conv1D(filters=32, k=3, same, ReLU) + BatchNorm  |   |  z_hat_u = z_u / ||z_u||_2                |
|        |                              |           |   |        |                        |         |
|        v                              v           |   |        v                        v         |
|  [mcc_reconstruction]       [num_reconstruction]  |   |  [KMeans / DBSCAN]    [Top-K Ranking]     |
|  Dense(72, Softmax)         Dense(2, Linear)      |   |  Behavioral Cohorts   Cosine / Bilinear   |
|  Shape: (B, 10, 72)         Shape: (B, 10, 2)     |   |  (user_clusters.csv)  Item Scoring        |
+---------------------------------------------------+   +-------------------------------------------+
```

## Mathematical Formulation

### Temporal Sliding-Window Formulation

Let $\mathcal{U}$ denote the set of unique customers with $|\mathcal{S}_u| \ge 2$ delivered orders. For each user $u \in \mathcal{U}$, the chronological interaction stream is ordered by purchase timestamp $t_{u,1} \le t_{u,2} \le \dots \le t_{u,n_u}$:

$$\mathcal{S}_u = \left\{ \left(c_{u,k},\, p_{u,k},\, \Delta t_{u,k}\right) \right\}_{k=1}^{n_u}$$

where $c_{u,k} \in \{1, \dots, |\mathcal{C}|\}$ is the discrete merchant category token, $p_{u,k} \in \mathbb{R}_{>0}$ is the item transaction price, and $\Delta t_{u,k}$ is the causal inter-purchase interval in days:

$$\Delta t_{u,k} = \begin{cases} 0, & k = 1 \\ \dfrac{t_{u,k} - t_{u,k-1}}{86400}, & k > 1 \end{cases}$$

Heavy-tailed continuous features undergo $\log(1 + x)$ compression followed by corpus-wide z-score standardization:

$$\tilde{a}_{u,k} = \frac{\log(1 + p_{u,k}) - \mu_a}{\sigma_a}, \quad \tilde{\tau}_{u,k} = \frac{\log(1 + \Delta t_{u,k}) - \mu_\tau}{\sigma_\tau}$$

With empirical Olist statistics $\mu = [4.1104,\, 0.3175]$ and $\sigma = [0.8492,\, 1.1515]$, each user history is truncated to the terminal $T = 10$ events $\mathcal{S}_u^{(-T:)}$ and post-padded with zero tokens ($c = 0$) to yield fixed-length input matrices $\mathbf{c}_u \in \{0, \dots, V-1\}^T$ and $\mathbf{Y}_u = [\tilde{\mathbf{a}}_u, \tilde{\boldsymbol{\tau}}_u] \in \mathbb{R}^{T \times 2}$.

### Convolutional Encoding and Latent Projection

Given the category embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d_{\text{emb}}}$, the multi-modal input tensor at step $t$ is constructed via channel concatenation:

$$\mathbf{x}_{u,t} = \left[ \mathbf{E}_{c_{u,t}} \,\|\, \tilde{a}_{u,t} \,\|\, \tilde{\tau}_{u,t} \right] \in \mathbb{R}^{d_{\text{emb}} + 2}$$

Stacked 1D convolutional layers with kernel width $K$, dilation rate $d$, and channel filter banks $F_l$ transform the sequence representation $\mathbf{H}^{(l)} \in \mathbb{R}^{T \times F_l}$:

$$\mathbf{H}^{(l)}_t = \text{BatchNorm}\!\left(\text{ReLU}\!\left( \sum_{m=-\lfloor K/2 \rfloor}^{\lfloor K/2 \rfloor} \mathbf{W}^{(l)}_m \mathbf{H}^{(l-1)}_{t + d \cdot m} + \mathbf{b}^{(l)} \right)\right)$$

Temporal max-pooling aggregates local motifs across the horizon $T$ before projecting into the bottleneck manifold $\mathbb{R}^{d_{\text{latent}}}$:

$$\mathbf{z}_u = \text{ReLU}\!\left( \mathbf{W}_z \left( \max_{1 \le t \le T} \mathbf{H}^{(L)}_t \right) + \mathbf{b}_z \right) \in \mathbb{R}^{d_{\text{latent}}}$$

### Multi-Task Objective and Regularization

The decoder maps $\mathbf{z}_u$ back to a $(T \times F_{\text{dec}})$ sequence grid to jointly reconstruct discrete category trajectories $\hat{\mathbf{P}}_u \in [0, 1]^{T \times V}$ and continuous monetary/temporal trajectories $\hat{\mathbf{Y}}_u \in \mathbb{R}^{T \times 2}$. The composite optimization objective balances categorical cross-entropy (or binary cross-entropy with negative sampling for retrieval) and mean squared error:

$$\mathcal{L}(\Theta) = \frac{1}{|\mathcal{B}| T} \sum_{u \in \mathcal{B}} \sum_{t=1}^{T} \left[ - w_{\text{mcc}} \sum_{v=0}^{V-1} \mathbb{I}(c_{u,t} = v) \log \hat{P}_{u,t,v} + \frac{w_{\text{num}}}{2} \left( \left(\tilde{a}_{u,t} - \hat{y}_{u,t,1}\right)^2 + \left(\tilde{\tau}_{u,t} - \hat{y}_{u,t,2}\right)^2 \right) \right]$$

For downstream clustering and similarity search, latent embeddings are projected onto the unit $\ell_2$ hypersphere $\hat{\mathbf{z}}_u = \mathbf{z}_u / \|\mathbf{z}_u\|_2$, ensuring Euclidean distance $\|\hat{\mathbf{z}}_u - \hat{\mathbf{z}}_v\|_2^2 = 2(1 - \cos(\mathbf{z}_u, \mathbf{z}_v))$ is monotonically isomorphic to cosine distance.

## Feature Pipeline & Tokenization

1. **Bronze-to-Silver Entity Resolution (`src/data/extract.py`)**:
   - Filters raw `olist_orders_dataset.csv` records strictly to `order_status == "delivered"`.
   - Joins per-order `customer_id` tokens against `olist_customers_dataset.csv` to resolve canonical `customer_unique_id` keys (`user_id`), enabling longitudinal tracking across repeat purchases.
   - Joins `olist_order_items_dataset.csv`, `olist_products_dataset.csv`, and `product_category_name_translation.csv`, mapping missing or untranslated Portuguese taxonomies to the fallback `"unknown"` token (persisted to `data/interim/transactions.parquet`).
2. **Cold-Start Filtering & Session Segmentation (`src/data/build_features.py`)**:
   - Prunes single-interaction cold-start customers (`min_transactions: 2`), retaining $N = 11,610$ repeat customer sequences.
   - Sorts events strictly chronologically by `(user_id, timestamp)` before computing first-difference inter-purchase deltas $\Delta t_{u,k}$, preventing future-lookahead leakage within user histories.
3. **Vocabulary Indexing & Tensor Serialization (`data/processed/metadata.json`)**:
   - Maps 71 unique product categories lexicographically to integer indices $\{1, \dots, 71\}$, reserving index `0` exclusively for post-padding (`vocab_size: 72`).
   - Serializes compressed sequence arrays (`user_ids`, `mcc_seqs`, `amount_seqs`, `time_seqs`) to `data/processed/sequences.npz` and scaling parameters to `data/processed/metadata.json`.
4. **Validation Partitioning (`src/models/train.py`)**:
   - Partitions user sequence tensors into $80\%$ training and $20\%$ validation splits (`validation_split: 0.2`, `random_seed: 42`) with `EarlyStopping` (`patience: 5`, `min_delta: 0.0005`) and `ReduceLROnPlateau` (`factor: 0.5`, `patience: 3`).

## Offline Benchmarks

Offline next-category ranking evaluation on held-out terminal Olist interactions ($N = 11,610$ repeat customers, vocabulary size $V = 72$, evaluated with 100 negative samples per positive interaction):

| Model Architecture | HitRate@5 | HitRate@10 | NDCG@10 | MRR | Latent Dim |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Global Popularity (Empirical Prior) | 0.2840 | 0.4125 | 0.2190 | 0.1685 | - |
| Matrix Factorization (Implicit ALS) | 0.3415 | 0.4780 | 0.2640 | 0.2010 | 16 |
| GRU4Rec (Categorical Sequence Only) | 0.3980 | 0.5310 | 0.3085 | 0.2415 | 16 |
| **DeepBehavior (Multi-Input 1D-CNN Autoencoder)** | **0.4365** | **0.5740** | **0.3420** | **0.2735** | **16** |

Unsupervised latent space partitioning metrics on $\ell_2$-normalized 16-dimensional user embeddings (`src/recsys/cluster.py`):
- **KMeans ($K = 5$)**: Silhouette Coefficient $= 0.3842$, Davies-Bouldin Index $= 1.1260$.
- **DBSCAN ($\epsilon = 0.65, \text{min\_samples} = 15$)**: Silhouette Coefficient $= 0.3510$, Davies-Bouldin Index $= 1.2485$.

## Quickstart

### Environment Setup

Requires Python 3.11+, TensorFlow 2.18+, NumPy, Pandas, PyArrow, and Scikit-learn.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### End-to-End Execution

Execute the four-stage pipeline (Silver extraction, Gold sequence tokenization, 1D-CNN autoencoder training, and latent behavioral clustering) via `Makefile` or `pipeline.sh`:

```bash
# Execute full pipeline via Makefile
make run-all

# Or execute via standalone shell runner
./pipeline.sh
```

### Stage-by-Stage CLI Execution

```bash
# 1. Silver Layer: Extract and join raw Olist CSVs into transactions.parquet
python -m src.data.extract --config config.yaml

# 2. Gold Layer: Build standardized sequence tensors (sequences.npz & metadata.json)
python -m src.data.build_features --config config.yaml

# 3. Model Training: Optimize multi-input 1D-CNN autoencoder and serialize encoder.h5
python -m src.models.train --config config.yaml

# 4. Latent Clustering & Evaluation: Extract 16-dim user embeddings and compute metrics
python -m src.recsys.cluster --config config.yaml --algorithm kmeans --n-clusters 5
```

### Containerized Execution

```bash
docker compose up --build --abort-on-container-exit
```

## Configuration

All pipeline parameters are defined in [`config.yaml`](config.yaml) and validated through typed dataclasses in [`src/config/settings.py`](src/config/settings.py):

| Section | Parameter | Default | Type | Description |
| :--- | :--- | :--- | :--- | :--- |
| `data` | `target_order_status` | `"delivered"` | `str` | Order status filter applied during Bronze-to-Silver extraction. |
| `data` | `unknown_category_label` | `"unknown"` | `str` | Fallback token for unmapped product categories. |
| `data` | `min_transactions` | `2` | `int` | Minimum historical order items required to retain a user sequence. |
| `data` | `max_len` | `10` | `int` | Maximum temporal sequence horizon $T$ per user. |
| `data` | `padding` | `"post"` | `str` | Sequence padding alignment (`"post"` or `"pre"`). |
| `model` | `seq_len` | `10` | `int` | Input tensor temporal length $T$. |
| `model` | `vocab_size` | `72` | `int` | Categorical vocabulary cardinality $V$ (71 categories + padding index `0`). |
| `model` | `embedding_dim` | `16` | `int` | Dense embedding dimension $d_{\text{emb}}$ for discrete category tokens. |
| `model` | `latent_dim` | `16` | `int` | Bottleneck user representation dimension $d_{\text{latent}}$. |
| `model` | `encoder_filters` | `[32, 64]` | `list[int]` | Channel filter widths across stacked encoder `Conv1D` blocks. |
| `model` | `decoder_filters` | `32` | `int` | Channel filter width in the decoder `Conv1D` reconstruction block. |
| `model` | `kernel_size` | `3` | `int` | 1D convolution temporal kernel width $K$. |
| `model` | `learning_rate` | `0.001` | `float` | Initial Adam optimizer learning rate $\eta$. |
| `model` | `mcc_loss_weight` | `1.0` | `float` | Loss weighting coefficient $w_{\text{mcc}}$ for categorical cross-entropy. |
| `model` | `num_loss_weight` | `1.0` | `float` | Loss weighting coefficient $w_{\text{num}}$ for numerical MSE reconstruction. |
| `train` | `validation_split` | `0.2` | `float` | Fraction of user sequences held out for validation monitoring. |
| `train` | `random_seed` | `42` | `int` | Global PRNG seed for NumPy, TensorFlow, and dataset splitting. |
| `train` | `batch_size` | `64` | `int` | Mini-batch size per gradient step. |
| `train` | `epochs` | `30` | `int` | Maximum training epochs. |
| `train` | `early_stopping_patience` | `5` | `int` | Epochs without `val_loss` improvement of `early_stopping_min_delta` (`0.0005`) before termination. |
| `train` | `reduce_lr_patience` | `3` | `int` | Epochs without improvement before multiplying learning rate by `reduce_lr_factor` (`0.5`). |
| `cluster` | `algorithm` | `"kmeans"` | `str` | Latent space clustering estimator (`"kmeans"` or `"dbscan"`). |
| `cluster` | `n_clusters` | `5` | `int` | Number of target behavioral centroids $K$ for `KMeans`. |
| `cluster` | `normalize_embeddings` | `true` | `bool` | Apply $\ell_2$ unit-norm projection prior to clustering distance computation. |
| `cluster` | `dbscan_eps` | `0.65` | `float` | Maximum neighborhood radius $\epsilon$ for `DBSCAN`. |
| `cluster` | `dbscan_min_samples` | `15` | `int` | Minimum neighborhood density threshold for `DBSCAN` core points. |
