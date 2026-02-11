SHELL := /bin/bash
CONDA_ENV := churn

.PHONY: up down ps ingest train-baseline build-trainer train-baseline-docker score-batch

# Start ONLY the DB (trainer is meant to run as a one-off job)
up:
	docker compose up -d db

down:
	docker compose down

ps:
	docker compose ps

ingest:
	conda run -n $(CONDA_ENV) python src/ingest.py

train-baseline:
	conda run -n $(CONDA_ENV) python src/train_baseline_v1.py

build-trainer:
	docker compose build trainer

train-baseline-docker:
	docker compose run --rm --build trainer

score-batch:
	conda run -n $(CONDA_ENV) python src/score_batch.py

inspect-preds:
	docker exec -it $$(docker compose ps -q db) psql -U app -d app -c "SELECT * FROM ml.predictions ORDER BY scored_at DESC LIMIT 10;"