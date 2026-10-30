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
| Deployment | Cloud Run (europe-west1) |
| CI/CD | GitHub Actions + Workload Identity Federation |
| Secrets | Google Secret Manager |
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
├── .github/workflows/ci.yml   # Tests on every push/PR, deploy on main
├── .gcp/
│   └── ar-cleanup-policy.json # Artifact Registry retention (keep 3 versions)
├── docker-compose.yml         # DB + trainer services
├── Dockerfile                 # Serving image for Cloud Run (multi-stage)
├── Dockerfile.train           # Image for the training job
├── requirements-api.txt       # Pinned serving-only deps (no pandas/sklearn)
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
export API_KEY=dev-key
uvicorn src.api.main:app --reload
```

Set `DATABASE_URL` if Postgres is not on the default `localhost:5432`. `API_KEY` is
required for every route except `/health` — without it those routes return `503`.

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

Every route except `/health` requires an `X-API-Key` header matching the `API_KEY`
environment variable. If `API_KEY` is unset the guarded routes return `503` rather
than falling open — export it before starting uvicorn locally.

**Example:**

```bash
export API_KEY=dev-key
uvicorn src.api.main:app --reload

curl -X POST http://localhost:8000/predict_by_id \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"customer_id": "7590-VHVEG"}'
```

## Deployment (Google Cloud Run)

The API runs on Cloud Run in `europe-west1`, deployed automatically on every push to
`main` that passes the test suite. The whole setup is designed to stay inside the
free tiers — see [Cost notes](#cost-notes) for the one exception.

```
Push auf main
   │
   ▼
GitHub Actions  ──WIF (OIDC, no SA keys)──►  github-deployer@ SA
   │  test job (Postgres service, full pipeline)  ──must be green──┐
   │  docker build -t …:$GITHUB_SHA  →  docker push                │
   │  gcloud run deploy --image                                    ▼
   ▼
Cloud Run  churn-api   europe-west1 · min=0 · max=1 · 1 vCPU · 512 MiB
   │  identity: churn-api-run@ SA   (secretAccessor on 2 secrets, nothing else)
   │  env DATABASE_URL ◄── Secret Manager
   │  env API_KEY      ◄── Secret Manager
   ▼
Neon Postgres (free plan, EU)   ml.model_runs · ml.predictions
```

Two service accounts, split by job: the **deployer** may push images and deploy but
cannot read secrets; the **runtime** may read its two secrets but cannot deploy.

### Resources

| Resource | Name | Notes |
|---|---|---|
| Cloud Run service | `churn-api` | `europe-west1`, scales to zero |
| Artifact Registry | `apps` | Docker, `europe-west1`, cleanup policy keeps 3 versions |
| Secret Manager | `DATABASE_URL`, `API_KEY` | injected as env vars at deploy time |
| Runtime SA | `churn-api-run@` | `roles/secretmanager.secretAccessor` per secret |
| Deployer SA | `github-deployer@` | `run.admin`, `artifactregistry.writer`, `iam.serviceAccountUser` |
| WIF pool / provider | `github` / `github` | locked to this repo via attribute condition |
| Database | Neon (free plan) | external; Cloud SQL has no always-free tier |

### One-time setup

Prerequisites: `gcloud` (`brew install --cask google-cloud-sdk`), a billing account,
and a Neon project with the schema loaded.

**1. Load the database.** Point the existing pipeline scripts at Neon — nothing in
them is Postgres-vendor-specific:

```bash
export DATABASE_URL='postgresql+psycopg2://USER:PASS@HOST/DB?sslmode=require'
python src/setup_db.py        # sql/01_schema.sql + sql/02_ml_schema.sql
python src/ingest.py          # ~7000 rows → raw_churn
python src/train_baseline_v1.py
python src/score_batch.py     # → ml.predictions
```

Keep this in the shell only — writing it into `.env` would repoint local
development at the cloud database.

**2. Project, budget, APIs.** Create the budget alert *before* anything else;
budgets themselves are free.

```bash
gcloud projects create "$PROJECT_ID"
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT"
gcloud config set project "$PROJECT_ID"
# then: Billing → Budgets & alerts → create a 1 € budget
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com iamcredentials.googleapis.com sts.googleapis.com
```

**3. Registry with a cleanup policy.** The free tier is only 0.5 GB, and each
image version counts, so the policy is not optional housekeeping:

```bash
gcloud artifacts repositories create apps \
  --repository-format=docker --location=europe-west1
gcloud artifacts repositories set-cleanup-policies apps \
  --location=europe-west1 --policy=.gcp/ar-cleanup-policy.json --no-dry-run
```

**4. Secrets and the runtime identity.** `--data-file=-` keeps the values out of
shell history; the IAM binding is per secret rather than project-wide:

```bash
printf '%s' "$DATABASE_URL" | gcloud secrets create DATABASE_URL --data-file=-
openssl rand -base64 32 | tr -d '\n' | gcloud secrets create API_KEY --data-file=-

gcloud iam service-accounts create churn-api-run
for S in DATABASE_URL API_KEY; do
  gcloud secrets add-iam-policy-binding "$S" \
    --member="serviceAccount:churn-api-run@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor
done
```

**5. Deployer identity and Workload Identity Federation.**

```bash
gcloud iam service-accounts create github-deployer
DEPLOYER="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
for R in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${DEPLOYER}" --role="$R"
done
# may set the runtime identity on the service, but not read its secrets
gcloud iam service-accounts add-iam-policy-binding \
  "churn-api-run@${PROJECT_ID}.iam.gserviceaccount.com" \
  --member="serviceAccount:${DEPLOYER}" --role=roles/iam.serviceAccountUser

gcloud iam workload-identity-pools create github --location=global
gcloud iam workload-identity-pools providers create-oidc github \
  --location=global --workload-identity-pool=github \
  --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping='google.subject=assertion.sub,attribute.repository=assertion.repository' \
  --attribute-condition="assertion.repository == 'uvxpv/ds-project-pipeline'"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/uvxpv/ds-project-pipeline"
```

> **The `--attribute-condition` is load-bearing.** Without it, *any* GitHub repository
> in the world can mint a token for this service account. The pool and provider names
> are readable in `.github/workflows/ci.yml`, so on a public repo this is the only
> thing standing between the workflow and anyone else's.

**6. Tell the workflow where to deploy.** Project ID and number are not secrets —
WIF is bound by the attribute condition, not by their obscurity:

```bash
gh variable set GCP_PROJECT_ID     --body "$PROJECT_ID"
gh variable set GCP_PROJECT_NUMBER --body "$PROJECT_NUMBER"
```

### Deploy flags, and why

```bash
gcloud run deploy churn-api --image="$IMAGE" --region=europe-west1 \
  --service-account="churn-api-run@${PROJECT_ID}.iam.gserviceaccount.com" \
  --allow-unauthenticated --min-instances=0 --max-instances=1 \
  --cpu=1 --memory=512Mi --concurrency=80 --timeout=30 \
  --set-secrets=DATABASE_URL=DATABASE_URL:latest,API_KEY=API_KEY:latest
```

- `--min-instances=0` — anything above zero switches to instance-based billing,
  which has no free tier and bills continuously.
- `--max-instances=1` — the hard ceiling on what a traffic spike can cost.
- `--cpu=1 --memory=512Mi` — this *is* the smallest sensible configuration. Cloud Run
  allows down to 0.08 vCPU, but below 1 vCPU it forces `concurrency=1` and the gen1
  execution environment, and since request-based billing charges per vCPU-*second*,
  less CPU just means longer requests. No saving, worse latency.
- `--allow-unauthenticated` — the endpoint is public, but the data routes require
  `X-API-Key`; only `/health` is open, so Cloud Run's startup probe and the CI smoke
  test work without a key.

### Local verification before deploying

The image is amd64; on Apple Silicon the `--platform` flag is required or Cloud Run
rejects the image:

```bash
docker build --platform linux/amd64 -t churn-api:local .
docker run --rm -p 8080:8080 \
  -e DATABASE_URL="$DATABASE_URL" -e API_KEY=dev-key -e PORT=8080 churn-api:local

curl -fsS localhost:8080/health                               # 200 {"status":"ok"}
curl -i   localhost:8080/runs/latest                          # 401
curl -fsS localhost:8080/runs/latest -H 'X-API-Key: dev-key'  # 200
```

`requirements-api.txt` is fully pinned; regenerate it with
`docker run --rm churn-api:local pip freeze`.

### Cost notes

| Item | Free allowance | Guard in this setup |
|---|---|---|
| Cloud Run compute | 2 M requests, 180 k vCPU-s, 360 k GiB-s / month | `min=0`, `max=1`, request-based billing |
| **Egress** | 1 GiB / month — **North America only** | see below |
| Artifact Registry | 0.5 GB storage | ~206 MB image, cleanup policy keeps 3 |
| Secret Manager | 6 active versions, 10 k accesses / month | 2 secrets; destroy old versions on rotation |
| Cloud Logging | 50 GiB / month | default retention |
| GitHub Actions | unlimited on public repos | — |
| Neon | 1 GB storage, 100 CU-h / month, no overage billing | scales to zero after 5 min |

**The one thing that is not strictly free:** the Cloud Run egress free tier covers
North America only, so outbound traffic from `europe-west1` is billed at roughly
$0.12/GB. With KB-sized JSON responses and `max-instances=1` this lands in fractions
of a cent, but it is not exactly zero. Cloud SQL is avoided entirely for the same
reason — it has no always-free tier and its smallest instance runs about 9 €/month.

Neon suspends its compute after ~5 minutes idle, which kills pooled connections.
`src/api/main.py` therefore creates the engine with `pool_pre_ping=True`; without it
the first request after an idle period fails.

### Teardown

```bash
gcloud projects delete "$PROJECT_ID"   # Cloud Run, registry, secrets, SAs, WIF pool
```

Delete the Neon project separately in its console.

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
| `API_KEY` | _(empty)_ | Shared secret for the guarded endpoints; empty = fail closed (`503`) |
| `PORT` | `8080` | Injected by Cloud Run; uvicorn binds to it |

## Database Schema

**`public.raw_churn`** — raw feature table, one row per customer.

**`ml.model_runs`** — one row per training run, storing artifact path, threshold, metrics (JSONB), git commit, and Docker image.

**`ml.predictions`** — one row per customer per run, storing `proba_churn`, `pred_label`, and `threshold`.
