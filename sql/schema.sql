-- PostgreSQL reference schema. The running loader creates equivalent tables with SQLAlchemy.
CREATE TABLE IF NOT EXISTS processed_transactions (
 source_row_id BIGINT PRIMARY KEY, invoice_no VARCHAR(20), stock_code VARCHAR(30),
 invoice_date TIMESTAMP, country VARCHAR(100), customer_id BIGINT,
 quantity DOUBLE PRECISION, unit_price DOUBLE PRECISION, amount_gbp DOUBLE PRECISION,
 category VARCHAR(40), review_flag BOOLEAN, invalid_reason TEXT
);
CREATE INDEX IF NOT EXISTS ix_transactions_invoice ON processed_transactions(invoice_no);
CREATE INDEX IF NOT EXISTS ix_transactions_category ON processed_transactions(category);
CREATE TABLE IF NOT EXISTS sale_invoices (
 invoice_no VARCHAR(20) PRIMARY KEY, invoice_date TIMESTAMP, country VARCHAR(100),
 total_gbp DOUBLE PRECISION, line_count INTEGER
);
CREATE TABLE IF NOT EXISTS anomaly_predictions (
 invoice_no VARCHAR(20) PRIMARY KEY REFERENCES sale_invoices(invoice_no), invoice_date TIMESTAMP, country VARCHAR(100),
 total_gbp DOUBLE PRECISION, line_count INTEGER, anomaly_score DOUBLE PRECISION,
 is_anomaly BOOLEAN, subset VARCHAR(15), rule_flag BOOLEAN
);
CREATE TABLE IF NOT EXISTS data_quality_results (
 metric VARCHAR(100) PRIMARY KEY, value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_metadata (
 model_id VARCHAR(40) PRIMARY KEY, details_json TEXT NOT NULL
);
