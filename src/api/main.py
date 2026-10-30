from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from src.config import DB_URL, TOP_AT_RISK_MAX

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # pool_pre_ping: Neon suspends its compute after ~5 min of inactivity, which
    # silently kills pooled connections. Without the pre-ping the first request
    # after an idle period fails with a stale-connection error.
    engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=300)
    yield


app = FastAPI(lifespan=lifespan)


class PredictByIdRequest(BaseModel):
    customer_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/runs/latest")
def latest_run():
    with engine.begin() as conn:
        row = (
            conn.execute(
                text(
                    """
                    SELECT run_id, model_name, created_at, threshold
                    FROM ml.model_runs
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
            )
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(status_code=404, detail="No runs found")
    return dict(row)


@app.post("/predict_by_id")
def predict_by_id(req: PredictByIdRequest):
    # return the latest batch prediction for this customer
    with engine.begin() as conn:
        row = (
            conn.execute(
                text(
                    """
                    SELECT p.customer_id, p.run_id, p.scored_at, p.proba_churn, p.pred_label, p.threshold
                    FROM ml.predictions p
                    JOIN (
                    SELECT run_id
                    FROM ml.model_runs
                    ORDER BY created_at DESC
                    LIMIT 1
                    ) r USING (run_id)
                    WHERE p.customer_id = :cid
                    """
                ),
                {"cid": req.customer_id},
            )
            .mappings()
            .first()
        )

    if row is None:
        raise HTTPException(
            status_code=404, detail="No prediction found for customer_id (or no runs)"
        )
    return dict(row)


@app.get("/top_at_risk")
def top_at_risk(n: int = 20):
    n = max(1, min(n, TOP_AT_RISK_MAX))
    with engine.begin() as conn:
        rows = (
            conn.execute(
                text(
                    f"""
                    SELECT p.customer_id, p.proba_churn, p.pred_label
                    FROM ml.predictions p
                    JOIN (
                    SELECT run_id
                    FROM ml.model_runs
                    ORDER BY created_at DESC
                    LIMIT 1
                    ) r USING (run_id)
                    ORDER BY p.proba_churn DESC
                    LIMIT {n}
                    """
                )
            )
            .mappings()
            .all()
        )
    return {"n": len(rows), "rows": [dict(r) for r in rows]}
