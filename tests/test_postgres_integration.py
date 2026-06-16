"""Optional isolated PostgreSQL integration check. Requires TEST_POSTGRES_URL."""
import os
from uuid import uuid4
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from src.data_cleaning import clean_transactions
from src.feature_engineering import invoice_features
from src.train_model import train_and_score
from src.database import load_database


def test_postgres_reload_relationships_and_rollback(tmp_path, monkeypatch):
    configured_url = os.getenv("TEST_POSTGRES_URL")
    if not configured_url:
        pytest.skip("TEST_POSTGRES_URL is not configured; isolated PostgreSQL integration was not run")
    parsed = make_url(configured_url)
    if parsed.get_backend_name() != "postgresql":
        pytest.fail("TEST_POSTGRES_URL must use the postgresql SQLAlchemy dialect")
    schema = "financial_anomaly_test_" + uuid4().hex[:12]
    administrative_engine = create_engine(parsed, connect_args={"connect_timeout": 5})
    isolated_url = parsed.update_query_dict({"options": f"-csearch_path={schema}"})
    isolated_connection_string = isolated_url.render_as_string(hide_password=False)
    raw = pd.DataFrame([
        ["500001", "A", "x", 2, "12/1/2010 8:00", 3.0, 12345, "UK"],
        ["500002", "A", "x", 3, "12/2/2010 8:00", 3.0, 12345, "UK"],
        ["500003", "A", "x", 4, "12/3/2010 8:00", 3.0, 12345, "UK"],
    ], columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    cleaned, audit = clean_transactions(raw)
    scored, metrics = train_and_score(invoice_features(cleaned), tmp_path / "pg_model.joblib")
    try:
        with administrative_engine.begin() as connection:
            connection.execute(text(f"CREATE SCHEMA {schema}"))
        first = load_database(cleaned, scored, audit, metrics, tmp_path, isolated_connection_string)
        assert first["stored_transactions"] == 3 and first["stored_predictions"] == 3
        repeated = cleaned.copy()
        repeated.loc[0, "amount_gbp"] = 10.0
        second = load_database(repeated, scored, audit, metrics, tmp_path, isolated_connection_string)
        assert second == first
        with administrative_engine.connect() as connection:
            relationships = connection.scalar(text("""SELECT COUNT(*) FROM information_schema.table_constraints
                WHERE table_schema=:schema AND table_name='anomaly_predictions' AND constraint_type='FOREIGN KEY'"""), {"schema": schema})
            old_amount = connection.scalar(text(f"SELECT amount_gbp FROM {schema}.processed_transactions WHERE source_row_id=0"))
            old_counts = [connection.scalar(text(f"SELECT COUNT(*) FROM {schema}.{table}")) for table in
                ("processed_transactions", "sale_invoices", "anomaly_predictions", "data_quality_results", "model_metadata")]
        assert relationships >= 1 and old_amount == 10.0
        changed = cleaned.copy()
        changed.loc[0, "amount_gbp"] = 888.0
        original_to_sql = pd.DataFrame.to_sql

        def fail_on_invoice_insert(self, name, con, *args, **kwargs):
            if name == "sale_invoices":
                raise RuntimeError("injected PostgreSQL insert failure")
            return original_to_sql(self, name, con, *args, **kwargs)

        with monkeypatch.context() as patcher:
            patcher.setattr(pd.DataFrame, "to_sql", fail_on_invoice_insert)
            with pytest.raises(RuntimeError, match="injected"):
                load_database(changed, scored, audit, metrics, tmp_path, isolated_connection_string)
        with administrative_engine.connect() as connection:
            after_amount = connection.scalar(text(f"SELECT amount_gbp FROM {schema}.processed_transactions WHERE source_row_id=0"))
            after_counts = [connection.scalar(text(f"SELECT COUNT(*) FROM {schema}.{table}")) for table in
                ("processed_transactions", "sale_invoices", "anomaly_predictions", "data_quality_results", "model_metadata")]
        assert after_amount == old_amount and after_counts == old_counts
    finally:
        with administrative_engine.begin() as connection:
            connection.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        administrative_engine.dispose()
