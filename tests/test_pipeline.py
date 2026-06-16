from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest
from src.data_loader import load_transactions
from src.data_cleaning import clean_transactions
from src.feature_engineering import invoice_features, FEATURES
from src.train_model import train_and_score
from src.predict import predict_invoices
from src.database import load_database


def sample() -> pd.DataFrame:
    return pd.DataFrame([
        ["100001", "A1", "Widget", 2, "12/1/2010 8:26", 3.0, None, "UK"],
        ["100001", "A1", "Widget", 2, "12/1/2010 8:26", 3.0, None, "UK"],
        ["C100002", "A1", "Widget", -1, "12/1/2010 9:26", 3.0, 12345, "UK"],
        ["bad", "A1", "Widget", 1, "12/1/2010 10:26", 3.0, None, "UK"],
        ["100003", "A1", "Widget", 4, "12/2/2010 8:26", 0.0, None, "France"],
        ["100004", "A1", "Widget", 0, "12/2/2010 9:26", 3.0, None, "UK"],
    ], columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])


def test_load_csv_and_required_fields(tmp_path):
    path = tmp_path / "source.csv"
    sample().to_csv(path, index=False)
    assert load_transactions(path).shape == (6, 8)
    with pytest.raises(ValueError):
        load_transactions(_incomplete(tmp_path))


def _incomplete(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"Quantity": [1]}).to_csv(path, index=False)
    return path


def test_clean_categories_and_amounts():
    clean, audit = clean_transactions(sample())
    assert audit["exact_duplicates"] == 1
    assert audit["invalid_rows_excluding_duplicates"] == 2
    assert audit["retained_rows"] == 3
    assert audit["retained_missing_by_column"]["CustomerID"] == 2
    assert clean.loc[0, "amount_gbp"] == 6
    assert clean.loc[2, "amount_gbp"] == -3
    assert clean.loc[2, "category"] == "return_or_cancellation"
    assert clean.loc[0, "category"] == "sale" and pd.isna(clean.loc[0, "CustomerID"])
    assert clean.loc[4, "category"] == "sale"  # zero price retained


def test_features_training_prediction_and_database(tmp_path):
    rows = []
    for i in range(30):
        rows.append([f"{100000+i}", "A1", "Widget", i+1, f"12/{1+i//3}/2010 8:26", 2.0, None, "UK"])
    raw = pd.DataFrame(rows, columns=sample().columns)
    clean, audit = clean_transactions(raw)
    features = invoice_features(clean)
    assert len(features) == 30 and set(FEATURES).issubset(features.columns)
    model_path = tmp_path / "model.joblib"
    scored, metrics = train_and_score(features, model_path)
    assert model_path.exists() and metrics["holdout_invoices"] == 6
    restored = predict_invoices(features, model_path).set_index("InvoiceNo")
    original = scored.set_index("InvoiceNo")
    np.testing.assert_allclose(restored.loc[original.index, "anomaly_score"], original["anomaly_score"])
    assert restored.loc[original.index, "is_anomaly"].tolist() == original["is_anomaly"].tolist()
    result = load_database(clean, scored, audit, metrics, tmp_path,
                           f"sqlite:///{tmp_path / 'test.sqlite'}")
    assert result["stored_transactions"] == 30 and result["stored_sale_invoices"] == result["stored_predictions"] == 30


def test_empty_input_has_clear_training_failure(tmp_path):
    cleaned, audit = clean_transactions(sample().iloc[:0])
    assert cleaned.empty and audit["initial_rows"] == 0
    features = invoice_features(cleaned)
    assert features.empty
    with pytest.raises(ValueError, match="At least two"):
        train_and_score(features, tmp_path / "empty.joblib")


def test_invalid_numeric_values_and_customer_id_policy():
    rows = []
    for i in range(8):
        rows.append([f"{200000+i}", "A1", "Widget", 1, "12/1/2010 8:26", 2.0, None, "UK"])
    raw = pd.DataFrame(rows, columns=sample().columns)
    raw["Quantity"] = raw["Quantity"].astype(float)
    raw.loc[0, "CustomerID"] = 12345.5
    raw.loc[1, "CustomerID"] = float("inf")
    raw.loc[2, "CustomerID"] = "unparseable"
    raw.loc[3, "CustomerID"] = None
    raw.loc[4, "Quantity"] = float("inf")
    raw.loc[5, "Quantity"] = float("-inf")
    raw.loc[6, "UnitPrice"] = float("inf")
    raw.loc[7, "Quantity"] = 1e308
    raw.loc[7, "UnitPrice"] = 1e308
    cleaned, audit = clean_transactions(raw)
    assert cleaned.loc[:2, "customer_id_status"].tolist() == ["invalid"] * 3
    assert cleaned.loc[3, "customer_id_status"] == "missing"
    assert cleaned.loc[:2, "category"].tolist() == ["sale"] * 3
    assert cleaned.loc[:2, "invalid_reason"].str.contains("customer_id_invalid").all()
    assert "customer_id_invalid" not in cleaned.loc[3, "invalid_reason"]
    assert cleaned.loc[4:7, "category"].eq("invalid").all()
    assert "amount_nonfinite" in cleaned.loc[7, "invalid_reason"]
    assert audit["customer_id_invalid_rows"] == 3
    assert audit["customer_id_missing_rows"] == 5
    assert audit["nonfinite_quantity_rows"] == 2
    assert audit["nonfinite_unit_price_rows"] == 1
    assert audit["amount_overflow_rows"] == 1
    assert len(invoice_features(cleaned)) == 4
    assert not invoice_features(cleaned)["eligible_for_model"].iloc[:3].any()


def test_invoice_conflicts_and_permitted_time_variation():
    raw = pd.DataFrame([
        ["300001", "A", "x", 1, "12/1/2010 8:00", 2.0, 12345, "UK"],
        ["300001", "B", "y", 1, "12/1/2010 9:00", 2.0, 12346, "France"],
        ["300002", "A", "x", 1, "12/1/2010 8:00", 2.0, 12345, "UK"],
        ["300002", "B", "y", 1, "12/1/2010 10:00", 2.0, 12345, "UK"],
        ["300003", "A", "x", 1, "12/1/2010 8:00", 2.0, 12345, "UK"],
        ["300003", "B", "y", 1, "12/2/2010 10:00", 2.0, 12345, "UK"],
    ], columns=sample().columns)
    cleaned, _ = clean_transactions(raw)
    invoices = invoice_features(cleaned).set_index("InvoiceNo")
    assert not invoices.loc["300001", "eligible_for_model"]
    assert "multiple_customer_ids" in invoices.loc["300001", "invoice_conflict_reason"]
    assert "multiple_countries" in invoices.loc["300001", "invoice_conflict_reason"]
    assert pd.isna(invoices.loc["300001", "customer_id"])
    assert pd.isna(invoices.loc["300001", "country"])
    assert invoices.loc["300002", "eligible_for_model"]
    assert invoices.loc["300002", "time_variation"]
    assert invoices.loc["300002", "invoice_date"] == pd.Timestamp("2010-12-01 10:00")
    assert not invoices.loc["300003", "eligible_for_model"]
    assert "time_span_over_24h" in invoices.loc["300003", "invoice_conflict_reason"]


def test_chronological_split_keeps_tied_timestamps_together(tmp_path):
    rows = []
    for i in range(10):
        stamp = "12/1/2010 8:00" if i < 7 else "12/2/2010 8:00" if i < 9 else "12/3/2010 8:00"
        rows.append([f"{400000+i}", "A", "x", i+1, stamp, 2.0, None, "UK"])
    cleaned, _ = clean_transactions(pd.DataFrame(rows, columns=sample().columns))
    scored, metrics = train_and_score(invoice_features(cleaned), tmp_path / "ties.joblib")
    assert metrics["train_invoices"] == 9 and metrics["holdout_invoices"] == 1
    train_times = set(scored.loc[scored.subset.eq("train"), "invoice_date"])
    holdout_times = set(scored.loc[scored.subset.eq("holdout"), "invoice_date"])
    assert train_times.isdisjoint(holdout_times)
    assert max(train_times) < min(holdout_times)


def test_invoice_aggregate_overflow_cannot_reach_model():
    raw = pd.DataFrame([
        ["450001", "A", "x", 1, "12/1/2010 8:00", 1e308, None, "UK"],
        ["450001", "B", "y", 1, "12/1/2010 8:01", 1e308, None, "UK"],
    ], columns=sample().columns)
    cleaned, _ = clean_transactions(raw)
    assert cleaned["category"].eq("sale").all()  # Each line amount is finite.
    invoice = invoice_features(cleaned).iloc[0]
    assert "nonfinite_invoice_aggregate" in invoice["invoice_conflict_reason"]
    assert not invoice["eligible_for_model"]


def test_unparseable_quantity_and_price_are_invalid():
    raw = sample().iloc[[0, 2]].copy().reset_index(drop=True)
    raw["Quantity"] = raw["Quantity"].astype(object)
    raw["UnitPrice"] = raw["UnitPrice"].astype(object)
    raw.loc[0, "Quantity"] = "not-a-number"
    raw.loc[1, "UnitPrice"] = "not-a-number"
    cleaned, audit = clean_transactions(raw)
    assert cleaned["category"].eq("invalid").all()
    assert "quantity_nonfinite" in cleaned.loc[0, "invalid_reason"]
    assert "unit_price_nonfinite" in cleaned.loc[1, "invalid_reason"]
    assert audit["retained_rows"] == 0


def test_failed_reload_rolls_back_all_tables(tmp_path, monkeypatch):
    raw = sample().iloc[[0, 2, 4]].copy()
    cleaned, audit = clean_transactions(raw)
    features = invoice_features(cleaned)
    scored, metrics = train_and_score(features, tmp_path / "db_model.joblib")
    db_path = tmp_path / "atomic.sqlite"
    url = f"sqlite:///{db_path}"
    load_database(cleaned, scored, audit, metrics, tmp_path, url)
    reloaded = cleaned.copy()
    reloaded.loc[0, "amount_gbp"] = 7.0
    load_database(reloaded, scored, audit, metrics, tmp_path, url)
    with sqlite3.connect(db_path) as connection:
        before = [connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                  for table in ("processed_transactions", "sale_invoices", "anomaly_predictions", "data_quality_results", "model_metadata")]
        old_amount = connection.execute("SELECT amount_gbp FROM processed_transactions WHERE source_row_id=0").fetchone()[0]
    changed = cleaned.copy()
    changed.loc[0, "amount_gbp"] = 999.0
    original_to_sql = pd.DataFrame.to_sql

    def fail_on_invoice_insert(self, name, con, *args, **kwargs):
        if name == "sale_invoices":
            raise RuntimeError("injected insert failure")
        return original_to_sql(self, name, con, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_sql", fail_on_invoice_insert)
    with pytest.raises(RuntimeError, match="injected"):
        load_database(changed, scored, audit, metrics, tmp_path, url)
    with sqlite3.connect(db_path) as connection:
        after = [connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                 for table in ("processed_transactions", "sale_invoices", "anomaly_predictions", "data_quality_results", "model_metadata")]
        amount = connection.execute("SELECT amount_gbp FROM processed_transactions WHERE source_row_id=0").fetchone()[0]
    assert after == before
    assert amount == old_amount == 7.0
