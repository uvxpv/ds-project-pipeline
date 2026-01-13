SHELL := /bin/bash
CONDA_ENV := churn

up:
	docker compose up -d

down:
	docker compose down

ps:
	docker compose ps

ingest:
	conda run -n $(CONDA_ENV) python src/ingest.py
