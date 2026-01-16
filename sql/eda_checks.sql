-- row count
SELECT COUNT(*) FROM raw_churn;

-- churn rate
SELECT churn, COUNT(*) FROM raw_churn GROUP BY churn;

-- missing TotalCharges
SELECT COUNT(*) AS missing_total_charges
FROM raw_churn
WHERE total_charges IS NULL;

-- tenure stats by churn
SELECT churn,
       AVG(tenure) AS avg_tenure,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tenure) AS median_tenure
FROM raw_churn
GROUP BY churn;
