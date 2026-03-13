import json
import os
import subprocess
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine, text

from src.config import (
    ARTIFACTS_DIR,
    ARTIFACT_PREFIX,
    C_FN,
    C_FP,
    ID_COLS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    VAL_TEST_SPLIT,
)

load_dotenv()


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

    X = df.drop(columns=[TARGET_COL])
    for id_col in ID_COLS:
        if id_col in X.columns:
            X = X.drop(columns=[id_col])
    y = df[TARGET_COL].astype(int).values

    # Split into train/val/test: 60/20/20
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=VAL_TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_tmp,
    )

    base = build_base_pipeline(X_train)

    # Fit base model
    base.fit(X_train, y_train)

    # Calibrate on validation set (sigmoid)
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
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
    run_dir = ARTIFACTS_DIR / f"{ARTIFACT_PREFIX}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save calibrated model (includes preprocessing + base model)
    joblib.dump(cal, run_dir / "model.joblib")

    # Save metrics + chosen threshold
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    git_commit = None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode == 0:
        git_commit = result.stdout.strip()

    docker_image = os.getenv("DOCKER_IMAGE")
    data_query = "SELECT * FROM raw_churn;"

    # ── MLflow tracking ───────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{ARTIFACT_PREFIX}_{run_id}") as mlflow_run:
        mlflow.log_params(
            {
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "val_test_split": VAL_TEST_SPLIT,
                "C_FN": C_FN,
                "C_FP": C_FP,
                "threshold": float(best_t),
                "model_name": MODEL_NAME,
            }
        )
        mlflow.log_metrics(
            {
                "val_roc_auc": out["val"]["roc_auc"],
                "val_pr_auc": out["val"]["pr_auc"],
                "test_roc_auc": out["test"]["roc_auc"],
                "test_pr_auc": out["test"]["pr_auc"],
                "best_val_cost": best_cost,
            }
        )
        mlflow.log_artifact(str(run_dir / "metrics.json"))
        mlflow.sklearn.log_model(
            cal,
            artifact_path="model",
            registered_model_name="churn-baseline",
        )

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
                "model_name": MODEL_NAME,
                "artifact_path": str((run_dir / "model.joblib").as_posix()),
                "data_query": data_query,
                "random_state": RANDOM_STATE,
                "threshold": float(best_t),
                "metrics": json.dumps(out),
                "git_commit": git_commit,
                "docker_image": docker_image,
            },
        )


if __name__ == "__main__":
    main()
