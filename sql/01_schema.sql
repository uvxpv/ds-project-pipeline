-- sql/01_schema.sql

CREATE TABLE IF NOT EXISTS raw_churn (
  customer_id         TEXT PRIMARY KEY,

  gender              TEXT,
  senior_citizen      SMALLINT,          -- 0/1 in CSV
  partner             BOOLEAN,           -- Yes/No -> bool on ingest
  dependents          BOOLEAN,           -- Yes/No -> bool on ingest
  tenure              INTEGER,

  phone_service       BOOLEAN,           -- Yes/No -> bool on ingest
  multiple_lines      TEXT,              -- Yes/No/No phone service

  internet_service    TEXT,              -- DSL/Fiber optic/No
  online_security     TEXT,              -- Yes/No/No internet service
  online_backup       TEXT,
  device_protection   TEXT,
  tech_support        TEXT,
  streaming_tv        TEXT,
  streaming_movies    TEXT,

  contract            TEXT,              -- Month-to-month/One year/Two year
  paperless_billing   BOOLEAN,           -- Yes/No -> bool on ingest
  payment_method      TEXT,

  monthly_charges     NUMERIC(10,2),
  total_charges       NUMERIC(12,2),     -- may be NULL (CSV has blanks)
  churn               BOOLEAN            -- Yes/No -> bool on ingest
);

-- Helpful indexes (not required, but nice)
CREATE INDEX IF NOT EXISTS idx_raw_churn_churn ON raw_churn(churn);
CREATE INDEX IF NOT EXISTS idx_raw_churn_contract ON raw_churn(contract);
CREATE INDEX IF NOT EXISTS idx_raw_churn_payment_method ON raw_churn(payment_method);
