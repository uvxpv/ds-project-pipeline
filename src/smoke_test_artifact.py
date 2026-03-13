import json
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.config import ARTIFACTS_DIR, DB_URL

load_dotenv()


def latest_run_dir(prefix="baseline_v1_") -> Path:
    runs = sorted([p for p in ARTIFACTS_DIR.glob(f"{prefix}*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(
            f"No runs found under {ARTIFACTS_DIR}/ with prefix {prefix}"
        )
    return runs[-1]


def main():
    run_dir = latest_run_dir()
    model_path = run_dir / "model.joblib"
    metrics_path = run_dir / "metrics.json"

    model = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text())
    threshold = metrics["val"]["threshold"]

    engine = create_engine(DB_URL)

    df = pd.read_sql("SELECT * FROM raw_churn LIMIT 50;", engine)
    X = df.drop(columns=["churn"])

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    print("Run dir:", run_dir)
    print("Threshold:", threshold)
    print("First 10 probabilities:", proba[:10].round(4).tolist())
    print("First 10 predictions:", pred[:10].tolist())


if __name__ == "__main__":
    main()
