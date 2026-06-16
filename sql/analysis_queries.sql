-- 1. Data quality categories
SELECT category, COUNT(*) AS rows FROM processed_transactions GROUP BY category ORDER BY rows DESC;
-- 2. Monthly net amount, including returns
SELECT DATE_TRUNC('month', invoice_date) AS month, SUM(amount_gbp) AS net_gbp
FROM processed_transactions WHERE category IN ('sale','return_or_cancellation') GROUP BY 1 ORDER BY 1;
-- 3. Countries with most model flags
SELECT country, COUNT(*) AS flagged_invoices FROM anomaly_predictions
WHERE is_anomaly GROUP BY country ORDER BY flagged_invoices DESC LIMIT 15;
-- 4. Highest scoring sale invoices
SELECT invoice_no, invoice_date, country, total_gbp, line_count, anomaly_score
FROM anomaly_predictions ORDER BY anomaly_score DESC LIMIT 20;
-- 5. Model and simple rule comparison
SELECT is_anomaly, rule_flag, COUNT(*) AS invoices FROM anomaly_predictions
GROUP BY is_anomaly, rule_flag ORDER BY is_anomaly DESC, rule_flag DESC;
-- 6. Verification: this pair must match run_summary.json and parquet row counts.
SELECT (SELECT COUNT(*) FROM processed_transactions) AS audit_rows,
       (SELECT COUNT(*) FROM sale_invoices) AS sale_invoices,
       (SELECT COUNT(*) FROM anomaly_predictions) AS scored_invoices;
