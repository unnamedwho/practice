"""Score new invoice feature rows with the persisted training pipeline."""
from pathlib import Path
import joblib
import pandas as pd


def predict_invoices(features: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    model = artifact["pipeline"]
    result = features[["InvoiceNo"]].copy()
    result["anomaly_score"] = -model.decision_function(features[artifact["features"]])
    result["is_anomaly"] = model.predict(features[artifact["features"]]) == -1
    return result
