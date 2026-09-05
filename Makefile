PYTHON ?= $(shell if [ -x ./venv/bin/python ]; then echo ./venv/bin/python; else echo python3; fi)
CONFIG ?= config.yaml

.PHONY: help install extract features etl train cluster run-all docker-build docker-run clean

help:
	@echo "DeepBehavior ML Pipeline Commands:"
	@echo "  make run-all       Run full pipeline locally (ETL -> Train -> Cluster)"
	@echo "  make etl           Run Silver & Gold data pipeline stages"
	@echo "  make extract       Run Silver layer extraction (transactions.parquet)"
	@echo "  make features      Run Gold layer sequence building (sequences.npz)"
	@echo "  make train         Train 1D-CNN Autoencoder and save models/encoder.h5"
	@echo "  make cluster       Extract 16-dim user embeddings and run clustering"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run full pipeline in Docker Compose"
	@echo "  make clean         Remove generated interim/processed artifacts and models"

install:
	$(PYTHON) -m pip install -r requirements.txt

extract:
	$(PYTHON) -m src.data.extract --config $(CONFIG)

features:
	$(PYTHON) -m src.data.build_features --config $(CONFIG)

etl: extract features

train:
	$(PYTHON) -m src.models.train --config $(CONFIG)

cluster:
	$(PYTHON) -m src.recsys.cluster --config $(CONFIG)

run-all: etl train cluster

docker-build:
	docker compose build

docker-run:
	docker compose up --build --abort-on-container-exit

clean:
	rm -f data/interim/transactions.parquet
	rm -f data/processed/sequences.npz data/processed/user_clusters.parquet data/processed/user_clusters.csv data/processed/user_embeddings.npz
	rm -f models/encoder.h5 models/autoencoder.h5 models/training_history.json models/cluster_summary.json
