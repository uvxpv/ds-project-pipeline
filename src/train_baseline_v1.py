from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import joblib
from dotenv import load_dotenv
import subprocess, json, os

load_dotenv()

# cost assumption (tune later)
C_FN = 1  # missed churn
C_FP = 1  # false alarm

RANDOM_STATE = 42

ARTIFACTS_DIR = Path("artifacts")


def load_data() -> tuple[pd.DataFrame, str, object]:
    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url)
    return pd.read_sql("SELECT * FROM raw_churn;", engine), db_url, engine


def build_base_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def pick_threshold_by_cost(
    y_true: np.ndarray, proba: np.ndarray, c_fn: float, c_fp: float
) -> tuple[float, float]:
    # Search thresholds; you can tighten this later.
    ts = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = None, float("inf")
    for t in ts:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        cost = c_fn * fn + c_fp * fp
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
    return best_t, float(best_cost)


def metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray, t: float) -> dict:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(t),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(y_true, pred, output_dict=True),
    }


def main():
    df, db_url, engine = load_data()
    target = "churn"

    X = df.drop(columns=[target])
    for id_col in ["customer_id", "customerID"]:
        if id_col in X.columns:
            X = X.drop(columns=[id_col])
    y = df[target].astype(int).values

    # Split into train/val/test: 60/20/20
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_tmp
    )

    base = build_base_pipeline(X_train)

    # Fit base model
    base.fit(X_train, y_train)

    # Calibrate on validation set (sigmoid)
    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)

    # Pick threshold on validation set by cost
    proba_val = cal.predict_proba(X_val)[:, 1]
    best_t, best_cost = pick_threshold_by_cost(y_val, proba_val, C_FN, C_FP)

    # Evaluate on test set (one time)
    proba_test = cal.predict_proba(X_test)[:, 1]
    out = {
        "model": "LogReg(class_weight=balanced) + sigmoid calibration",
        "random_state": RANDOM_STATE,
        "data": {
            "n_total": int(len(df)),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
        },
        "costs": {"C_FN": C_FN, "C_FP": C_FP, "best_val_cost": best_cost},
        "val": {
            "roc_auc": float(roc_auc_score(y_val, proba_val)),
            "pr_auc": float(average_precision_score(y_val, proba_val)),
            **metrics_at_threshold(y_val, proba_val, best_t),
        },
        "test": {
            "roc_auc": float(roc_auc_score(y_test, proba_test)),
            "pr_auc": float(average_precision_score(y_test, proba_test)),
            **metrics_at_threshold(y_test, proba_test, best_t),
        },
    }

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_DIR / f"baseline_v1_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save calibrated model (includes preprocessing + base model)
    joblib.dump(cal, run_dir / "model.joblib")

    # Save metrics + chosen threshold
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    git_commit = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        pass

    docker_image = os.getenv("DOCKER_IMAGE")  # optional, can be None
    data_query = "SELECT * FROM raw_churn;"  # keep it simple for now

    with engine.begin() as conn:
        conn.execute(
            text(
                """
            INSERT INTO ml.model_runs
            (run_id, model_name, artifact_path, data_query, random_state, threshold, metrics, git_commit, docker_image)
            VALUES
            (:run_id, :model_name, :artifact_path, :data_query, :random_state, :threshold,
            CAST(:metrics AS jsonb), :git_commit, :docker_image)
            """
            ),
            {
                "run_id": run_id,
                "model_name": "logreg_balanced + sigmoid_calibration",
                "artifact_path": str((run_dir / "model.joblib").as_posix()),
                "data_query": data_query,
                "random_state": RANDOM_STATE,
                "threshold": float(best_t),
                "metrics": json.dumps(out),
                "git_commit": git_commit,
                "docker_image": docker_image,
            },
        )

    print("Saved run to:", run_dir)
    print("Chosen threshold (val):", best_t)
    print("TEST ROC-AUC:", out["test"]["roc_auc"])
    print("TEST PR-AUC :", out["test"]["pr_auc"])
    print("TEST confusion:", out["test"]["confusion_matrix"])


if __name__ == "__main__":
    main()
