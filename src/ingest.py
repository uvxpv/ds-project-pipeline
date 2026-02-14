import os

import pandas as pd
from sqlalchemy import create_engine, text

CSV_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://app:app@localhost:5432/app")
# TODO: get this from the environment later

YES_NO_TO_BOOL = {
    "Yes": True,
    "No": False,
}

YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]


def yn_to_bool(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return YES_NO_TO_BOOL.get(x)


def main():
    df = pd.read_csv(CSV_PATH)

    # Convert pure Yes/No columns to bool
    for c in YES_NO_COLS:
        df[c] = df[c].apply(yn_to_bool)

    # Numeric conversions
    df["SeniorCitizen"] = (
        pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    )
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # TotalCharges sometimes blank -> 0
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace({"": 0}), errors="coerce"
    )

    # Rename to match SQL schema
    df = df.rename(
        columns={
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
    )

    expected_cols = [
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
    df = df[expected_cols]

    engine = create_engine(DB_URL)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE raw_churn;"))

    df.to_sql(
        "raw_churn",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    print(f"Loaded {len(df)} rows into raw_churn")


if __name__ == "__main__":
    main()
