"""SQLAlchemy schema and bulk loading; DATABASE_URL may target PostgreSQL."""
import json
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import (MetaData, Table, Column, BigInteger, Integer, Float, String, Text,
                        Boolean, DateTime, ForeignKey, create_engine, select, func)

metadata = MetaData()
transactions = Table("processed_transactions", metadata,
    Column("source_row_id", BigInteger, primary_key=True), Column("invoice_no", String(20), index=True),
    Column("stock_code", String(30)), Column("invoice_date", DateTime), Column("country", String(100)),
    Column("customer_id", BigInteger, nullable=True), Column("quantity", Float), Column("unit_price", Float),
    Column("amount_gbp", Float), Column("category", String(40), index=True),
    Column("review_flag", Boolean), Column("invalid_reason", Text))
sale_invoices = Table("sale_invoices", metadata,
    Column("invoice_no", String(20), primary_key=True), Column("invoice_date", DateTime),
    Column("country", String(100)), Column("total_gbp", Float), Column("line_count", Integer))
predictions = Table("anomaly_predictions", metadata,
    Column("invoice_no", String(20), ForeignKey("sale_invoices.invoice_no"), primary_key=True), Column("invoice_date", DateTime),
    Column("country", String(100)), Column("total_gbp", Float), Column("line_count", Integer),
    Column("anomaly_score", Float), Column("is_anomaly", Boolean), Column("subset", String(15)),
    Column("rule_flag", Boolean))
quality = Table("data_quality_results", metadata,
    Column("metric", String(100), primary_key=True), Column("value_json", Text, nullable=False))
model_metadata = Table("model_metadata", metadata,
    Column("model_id", String(40), primary_key=True), Column("details_json", Text, nullable=False))


def load_database(cleaned: pd.DataFrame, scored: pd.DataFrame, audit: dict, metrics: dict,
                  root: Path, url: str | None = None) -> dict:
    url = url or os.environ.get("DATABASE_URL") or f"sqlite:///{root / 'data' / 'processed' / 'transactions.sqlite'}"
    engine = create_engine(url)
    tx = cleaned.rename(columns={"InvoiceNo": "invoice_no", "StockCode": "stock_code", "InvoiceDate": "invoice_date",
                                 "Country": "country", "CustomerID": "customer_id", "Quantity": "quantity", "UnitPrice": "unit_price"})
    tx = tx[[c.name for c in transactions.columns]].copy()
    tx["customer_id"] = tx["customer_id"].astype(object).where(tx["customer_id"].notna(), None)
    pred = scored.rename(columns={"InvoiceNo": "invoice_no", "invoice_date": "invoice_date"})
    pred = pred[[c.name for c in predictions.columns]].copy()
    invoice_rows = pred[[c.name for c in sale_invoices.columns]].copy()
    try:
        with engine.begin() as connection:
            metadata.create_all(connection)
            connection.execute(predictions.delete())
            connection.execute(sale_invoices.delete())
            connection.execute(transactions.delete())
            connection.execute(quality.delete())
            connection.execute(model_metadata.delete())
            # Passing this same Connection makes pandas join the open transaction.
            tx.to_sql("processed_transactions", connection, if_exists="append", index=False, chunksize=5000)
            invoice_rows.to_sql("sale_invoices", connection, if_exists="append", index=False, chunksize=5000)
            pred.to_sql("anomaly_predictions", connection, if_exists="append", index=False, chunksize=5000)
            connection.execute(quality.insert(), [{"metric": k, "value_json": json.dumps(v)} for k, v in audit.items()])
            connection.execute(model_metadata.insert(), {"model_id": "isolation_forest_v1", "details_json": json.dumps(metrics)})
            tx_count = connection.scalar(select(func.count()).select_from(transactions))
            invoice_count = connection.scalar(select(func.count()).select_from(sale_invoices))
            pred_count = connection.scalar(select(func.count()).select_from(predictions))
            if tx_count != len(cleaned) or invoice_count != len(scored) or pred_count != len(scored):
                raise RuntimeError("Stored row counts differ from processed datasets")
        return {"database_url_kind": engine.dialect.name, "stored_transactions": int(tx_count),
                "stored_sale_invoices": int(invoice_count), "stored_predictions": int(pred_count)}
    finally:
        engine.dispose()
