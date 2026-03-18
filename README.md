# Customer Churn Prediction Pipeline

An end-to-end Data Science project that trains a churn model on data stored in PostgreSQL, persists predictions back to the database, and serves them via a FastAPI endpoint.

## Overview

The dataset is the IBM Telco Customer Churn dataset (~7 000 customers). A PostgreSQL instance running in Docker acts as the simulated production database. The pipeline covers every stage from raw data ingestion to model serving.

```
CSV → Postgres (raw_churn) → Train → Artifacts + MLflow → Batch Score → Postgres (ml.predictions) → FastAPI
```

## Stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL 16 (Docker) |
| Training | scikit-learn, pandas, joblib |
| Experiment tracking | MLflow (SQLite backend) |
| Serving | FastAPI |
| Orchestration | Docker Compose |
| Linting / formatting | Ruff |
| Testing | pytest |

## Project Structure

```
.
├── data/raw/                  # Raw CSV (Telco churn dataset)
├── sql/
│   ├── 01_schema.sql          # raw_churn table definition
│   └── 02_ml_schema.sql       # ml.model_runs + ml.predictions tables
├── src/
│   ├── config.py              # Central config (paths, DB URL, hyperparams)
│   ├── ingest.py              # Loads CSV → raw_churn table
│   ├── train_baseline_v1.py   # Trains, calibrates, and registers the model
│   ├── score_batch.py         # Scores all customers → ml.predictions
│   ├── smoke_test_artifact.py # Sanity-checks a saved model artifact
│   ├── setup_db.py            # Helper to run schema migrations manually
│   └── api/
│       └── main.py            # FastAPI inference endpoints
├── artifacts/                 # Saved model artifacts (joblib + metrics.json)
├── notebooks/                 # EDA notebooks
├── tests/                     # pytest test suite
├── docker-compose.yml         # DB + trainer services
├── Dockerfile.train           # Image for the training job
├── makefile                   # Common task shortcuts
└── pyproject.toml             # Ruff + pytest configuration
```

## Quickstart

### Prerequisites

- Docker + Docker Compose
- Conda (environment named `churn`)

### 1. Start the database

```bash
make up
```

This starts a PostgreSQL 16 container on port `5432`. The SQL files under `sql/` are auto-executed on first run to create the schema.

### 2. Ingest the raw data

```bash
make ingest
```

Reads `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`, cleans column names, casts types, and upserts ~7 000 rows into `raw_churn`.

### 3. Train the model

**Locally (conda):**

```bash
make train-baseline
```

**Inside Docker (reproducible environment):**

```bash
make train-baseline-docker
```

This builds the trainer image and runs it as a one-off container. Artifacts are written back to the host via a bind mount.

### 4. Score customers in batch

```bash
make score-batch
```

Runs inference on all customers and writes churn probability + label into `ml.predictions`.

### 5. Start the API

```bash
uvicorn src.api.main:app --reload
```

Set `DATABASE_URL` if Postgres is not on the default `localhost:5432`.

## Model

The baseline model (`train_baseline_v1.py`) uses:

- **Logistic Regression** with `class_weight="balanced"` (handles class imbalance)
- **Probability calibration** via Platt scaling (sigmoid) on a held-out validation set
- **Cost-sensitive threshold tuning** — sweeps thresholds and picks the one that minimises `C_FN × FN + C_FP × FP`
- **Train / val / test split:** 60 / 20 / 20

Every run logs parameters and metrics to **MLflow** and writes a record to `ml.model_runs` (including the git commit SHA and Docker image tag for reproducibility).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/runs/latest` | Metadata for the most recent training run |
| `POST` | `/predict_by_id` | Churn probability for a single `customer_id` |
| `GET` | `/top_at_risk?n=20` | Top-N customers by churn probability (max 500) |

**Example:**

```bash
curl -X POST http://localhost:8000/predict_by_id \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "7590-VHVEG"}'
```

## Makefile Reference

| Target | Description |
|---|---|
| `make up` | Start the database container |
| `make down` | Stop all containers |
| `make ps` | Show container status |
| `make ingest` | Load CSV into Postgres |
| `make train-baseline` | Train locally via conda |
| `make build-trainer` | Build the trainer Docker image |
| `make train-baseline-docker` | Train inside Docker |
| `make score-batch` | Run batch scoring |
| `make inspect-preds` | `psql` query of the latest predictions |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://app:app@localhost:5432/app` | SQLAlchemy connection string |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend store |
| `MLFLOW_EXPERIMENT_NAME` | `churn-baseline` | MLflow experiment name |

## Database Schema

**`public.raw_churn`** — raw feature table, one row per customer.

**`ml.model_runs`** — one row per training run, storing artifact path, threshold, metrics (JSONB), git commit, and Docker image.

**`ml.predictions`** — one row per customer per run, storing `proba_churn`, `pred_label`, and `threshold`.
