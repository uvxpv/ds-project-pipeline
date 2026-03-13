import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DB_URL, EXPECTED_RAW_COLS, RAW_CSV_PATH, RENAME_MAP, YES_NO_COLS

YES_NO_TO_BOOL = {
    "Yes": True,
    "No": False,
}


def yn_to_bool(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    return YES_NO_TO_BOOL.get(x)


def main():
    df = pd.read_csv(RAW_CSV_PATH)

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

    df = df.rename(columns=RENAME_MAP)

    df = df[EXPECTED_RAW_COLS]

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
