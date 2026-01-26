CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS ml.model_runs (
  run_id           TEXT PRIMARY KEY,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  model_name       TEXT NOT NULL,
  artifact_path    TEXT NOT NULL,
  data_query       TEXT NOT NULL,
  random_state     INT,
  threshold        DOUBLE PRECISION,
  metrics          JSONB NOT NULL,
  git_commit       TEXT,
  docker_image     TEXT
);

CREATE TABLE IF NOT EXISTS ml.predictions (
  run_id        TEXT NOT NULL REFERENCES ml.model_runs(run_id) ON DELETE CASCADE, -- ensure deletion of model run also deletes predictions
  customer_id   TEXT NOT NULL,
  scored_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  proba_churn   DOUBLE PRECISION NOT NULL,
  pred_label    INT NOT NULL,
  threshold     DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (run_id, customer_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_customer ON ml.predictions(customer_id);
