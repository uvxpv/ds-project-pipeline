from contextlib import asynccontextmanager
from secrets import compare_digest

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from src import config
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


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard the data endpoints with a shared secret.

    Read through the module (``config.API_KEY``, not a from-import) so tests can
    monkeypatch it. ``/health`` stays unguarded so Cloud Run's startup probe and
    the CI smoke test work without a key.
    """
    expected = config.API_KEY
    if not expected:
        raise HTTPException(status_code=503, detail="API key not configured")
    # Compare as bytes: compare_digest rejects non-ASCII str outright, which would
    # turn a header like "ü" into a 500 instead of a 401.
    if x_api_key is None or not compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class PredictByIdRequest(BaseModel):
    customer_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/runs/latest", dependencies=[Depends(require_api_key)])
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


@app.post("/predict_by_id", dependencies=[Depends(require_api_key)])
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


@app.get("/top_at_risk", dependencies=[Depends(require_api_key)])
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
