"""
Batch scoring script: loads a trained model, scores customers from Postgres,
writes results to ml.predictions with upsert.
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_artifact_path(artifact_path: str) -> Path:
    """Resolve artifact path (absolute or relative to project root)."""
    p = Path(artifact_path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    # Try relative to project root
    root_path = PROJECT_ROOT / artifact_path
    if root_path.exists():
        return root_path
    raise FileNotFoundError(f"Model artifact not found: {artifact_path}")


def fetch_run(engine, run_id: str | None) -> dict:
    """Fetch model run from ml.model_runs (latest or by run_id)."""
    if run_id:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT run_id, artifact_path, data_query, threshold
                    FROM ml.model_runs
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            ).fetchone()
        if row is None:
            raise ValueError(f"Run not found: {run_id}")
    else:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT run_id, artifact_path, data_query, threshold
                    FROM ml.model_runs
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ),
            ).fetchone()
        if row is None:
            raise ValueError("No model runs found in ml.model_runs")

    return {
        "run_id": row[0],
        "artifact_path": row[1],
        "data_query": row[2],
        "threshold": float(row[3]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch score customers with a trained model"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID from ml.model_runs (default: latest run)",
    )
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    engine = create_engine(db_url)

    # 1. Fetch run
    print("Fetching model run...")
    run = fetch_run(engine, args.run_id)
    run_id = run["run_id"]
    threshold = run["threshold"]
    print(f"  run_id={run_id}, threshold={threshold}")

    # 2. Load model
    artifact_path = resolve_artifact_path(run["artifact_path"])
    print(f"Loading model from {artifact_path}...")
    model = joblib.load(artifact_path)

    # 3. Pull feature rows (same data_query as training)
    data_query = run["data_query"]
    print(f"Pulling data: {data_query[:60]}...")
    df = pd.read_sql(data_query, engine)
    n_customers = len(df)

    # Align with training: drop target and id cols for features
    target = "churn"
    id_cols = ["customer_id", "customerID"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in data")
    X = df.drop(columns=[target])
    for id_col in id_cols:
        if id_col in X.columns:
            X = X.drop(columns=[id_col])
    customer_id_col = "customer_id" if "customer_id" in df.columns else "customerID"
    customer_ids = df[customer_id_col].astype(str).tolist()

    # 4. Score
    proba_churn = model.predict_proba(X)[:, 1]
    pred_label = (proba_churn >= threshold).astype(int)

    n_scored = len(proba_churn)

    # 5. Sanity checks
    print("\n--- Sanity checks ---")
    print(f"  n_scored: {n_scored}")
    print(f"  n_customers: {n_customers}")
    if n_scored != n_customers:
        print(f"  WARNING: n_scored ({n_scored}) != n_customers ({n_customers})")
    else:
        print("  OK: n_scored == n_customers")

    print(f"  proba_churn min: {proba_churn.min():.4f}")
    print(f"  proba_churn max: {proba_churn.max():.4f}")
    print(f"  proba_churn mean: {proba_churn.mean():.4f}")
    n_positive = int(pred_label.sum())
    print(f"  n_predicted_positive (at threshold={threshold}): {n_positive}")

    # 6. Upsert to ml.predictions
    print("\nWriting to ml.predictions...")
    upsert_sql = text(
        """
        INSERT INTO ml.predictions (run_id, customer_id, proba_churn, pred_label, threshold)
        VALUES (:run_id, :customer_id, :proba_churn, :pred_label, :threshold)
        ON CONFLICT (run_id, customer_id)
        DO UPDATE SET
            proba_churn = EXCLUDED.proba_churn,
            pred_label = EXCLUDED.pred_label,
            threshold = EXCLUDED.threshold,
            scored_at = NOW()
        """
    )
    rows = [
        {
            "run_id": run_id,
            "customer_id": str(cid),
            "proba_churn": float(p),
            "pred_label": int(lbl),
            "threshold": threshold,
        }
        for cid, p, lbl in zip(customer_ids, proba_churn, pred_label, strict=True)
    ]
    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)

    print(f"Done. Upserted {n_scored} predictions for run_id={run_id}")


if __name__ == "__main__":
    main()
