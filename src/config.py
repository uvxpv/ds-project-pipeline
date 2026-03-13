"""
Central configuration for the ds-project-pipeline.

All paths, database URLs, model hyper-parameters and constants live here
so that every module imports from one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ── Database ──────────────────────────────────────────────────────────
DB_URL_DEFAULT = "postgresql+psycopg2://app:app@localhost:5432/app"
DB_URL = os.getenv("DATABASE_URL", DB_URL_DEFAULT)

# ── Model / training ─────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.4
VAL_TEST_SPLIT = 0.5

# Cost assumptions for threshold tuning
C_FN = 1  # cost of missed churn
C_FP = 1  # cost of false alarm

MODEL_NAME = "logreg_balanced + sigmoid_calibration"
ARTIFACT_PREFIX = "baseline_v1"

# ── MLflow ────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-baseline")

# ── API ───────────────────────────────────────────────────────────────
TOP_AT_RISK_MAX = 500

# ── Feature / target columns ─────────────────────────────────────────
TARGET_COL = "churn"
ID_COLS = ["customer_id", "customerID"]

YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

EXPECTED_RAW_COLS = [
    "customer_id",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "churn",
]

RENAME_MAP = {
    "customerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
    "gender": "gender",
    "tenure": "tenure",
}
