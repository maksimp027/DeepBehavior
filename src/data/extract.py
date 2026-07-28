import argparse
import logging
from pathlib import Path
import pandas as pd

from src.config.settings import load_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("deepbehavior.data.extract")


def extract_data(
    raw_data_dir: Path | None = None,
    output_file: Path | None = None,
    target_order_status: str | None = None,
    unknown_category_label: str | None = None,
    config_path: Path | str | None = None,
) -> pd.DataFrame:
    """Extract and join Bronze Olist CSVs into Silver transactions.parquet."""
    cfg = load_settings(config_path)
    raw_dir = Path(raw_data_dir) if raw_data_dir is not None else cfg.paths.raw_dir
    out_path = Path(output_file) if output_file is not None else cfg.paths.interim_parquet
    order_status = (
        target_order_status
        if target_order_status is not None
        else cfg.data.target_order_status
    )
    unknown_label = (
        unknown_category_label
        if unknown_category_label is not None
        else cfg.data.unknown_category_label
    )

    logger.info("Reading raw Olist datasets from %s", raw_dir)
    orders = pd.read_csv(
        raw_dir / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    )
    customers = pd.read_csv(
        raw_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_unique_id"],
    )
    items = pd.read_csv(
        raw_dir / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id", "price"],
    )
    products = pd.read_csv(
        raw_dir / "olist_products_dataset.csv",
        usecols=["product_id", "product_category_name"],
    )
    translations = pd.read_csv(
        raw_dir / "product_category_name_translation.csv",
    )

    orders = orders[orders["order_status"] == order_status].copy()

    # Resolve order-scoped customer_id to persistent customer_unique_id to link longitudinal sessions.
    df = orders.merge(customers, on="customer_id", how="inner")
    df = df.merge(items, on="order_id", how="inner")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(translations, on="product_category_name", how="left")

    # Map unmapped Portuguese product taxonomies to the canonical out-of-vocabulary fallback token.
    df["category"] = df["product_category_name_english"].fillna(unknown_label)
    df["timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    clean_transactions = df[
        ["customer_unique_id", "timestamp", "category", "price"]
    ].rename(
        columns={
            "customer_unique_id": "user_id",
            "price": "amount",
            "category": "mcc_category",
        }
    )

    clean_transactions = clean_transactions.sort_values(
        by=["user_id", "timestamp"]
    ).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_transactions.to_parquet(out_path, engine="pyarrow", index=False)
    logger.info(
        "Saved %d silver transactions (%d unique users) to %s",
        len(clean_transactions),
        clean_transactions["user_id"].nunique(),
        out_path,
    )

    return clean_transactions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Olist CSVs into Silver transactions.parquet"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()
    extract_data(config_path=args.config)


if __name__ == "__main__":
    main()
