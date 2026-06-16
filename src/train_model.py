"""Chronological holdout and reproducible unsupervised invoice scoring."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from .feature_engineering import FEATURES

RANDOM_SEED = 42
N_ESTIMATORS = 200
CONTAMINATION = 0.02
SENSITIVITY_LEVELS = (0.01, 0.02, 0.05)


def train_and_score(invoices: pd.DataFrame, model_path: Path) -> tuple[pd.DataFrame, dict]:
    if len(invoices) < 2:
        raise ValueError("At least two eligible sale invoices are needed for train and holdout")
    if "eligible_for_model" in invoices and not invoices["eligible_for_model"].all():
        raise ValueError("Resolve or exclude invoice conflicts before training")
    ordered = invoices.sort_values(["invoice_date", "InvoiceNo"]).copy()
    if ordered["invoice_date"].isna().any() or ordered["invoice_date"].nunique() < 2:
        raise ValueError("Train and holdout need at least two distinct valid invoice timestamps")
    target = max(1, int(len(ordered) * .8))
    boundary = ordered["invoice_date"].iloc[target - 1]
    if boundary == ordered["invoice_date"].max():
        boundary = ordered["invoice_date"].drop_duplicates().iloc[-2]
    train_mask = ordered["invoice_date"].le(boundary)
    train = ordered.loc[train_mask]
    holdout = ordered.loc[~train_mask]
    model = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", RobustScaler()),
                      ("forest", IsolationForest(n_estimators=N_ESTIMATORS, contamination=CONTAMINATION,
                                                 random_state=RANDOM_SEED, n_jobs=-1))])
    model.fit(train[FEATURES])
    ordered["subset"] = np.where(train_mask, "train", "holdout")
    ordered["anomaly_score"] = -model.decision_function(ordered[FEATURES])
    ordered["is_anomaly"] = model.predict(ordered[FEATURES]) == -1
    # Simple, transparent baseline; comparison is descriptive without labels.
    rule_cut = float(train["total_gbp"].quantile(.99))
    ordered["rule_flag"] = ordered["total_gbp"].gt(rule_cut)
    holdout_rows = ordered.loc[ordered["subset"].eq("holdout")]
    # Same fitted forest and train-only score quantiles; these are operating-point
    # comparisons, not supervised quality measures or independently refitted models.
    raw_score = -model.score_samples(ordered[FEATURES])
    train_raw_score = raw_score[train_mask.to_numpy()]
    sensitivity = {}
    for level in SENSITIVITY_LEVELS:
        threshold = float(np.quantile(train_raw_score, 1 - level))
        flags = raw_score > threshold
        sensitivity[f"{level:.2f}"] = {"train_flagged": int(flags[train_mask.to_numpy()].sum()),
                                         "holdout_flagged": int(flags[~train_mask.to_numpy()].sum()),
                                         "train_raw_score_threshold": threshold}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "features": FEATURES, "trained_through": str(boundary)}, model_path)
    metrics = {"level": "sale invoice", "train_invoices": int(len(train)), "holdout_invoices": int(len(holdout)),
               "train_through": str(boundary), "holdout_from": str(holdout["invoice_date"].min()),
               "train_flagged": int(ordered.loc[ordered.subset.eq("train"), "is_anomaly"].sum()),
               "holdout_flagged": int(ordered.loc[ordered.subset.eq("holdout"), "is_anomaly"].sum()),
               "rule_threshold_gbp": rule_cut, "rule_flagged": int(ordered["rule_flag"].sum()),
               "rule_model_overlap": int((ordered["rule_flag"] & ordered["is_anomaly"]).sum()),
               "holdout_rule_flagged": int(holdout_rows["rule_flag"].sum()),
               "holdout_rule_model_overlap": int((holdout_rows["rule_flag"] & holdout_rows["is_anomaly"]).sum()),
               "holdout_model_only": int((holdout_rows["is_anomaly"] & ~holdout_rows["rule_flag"]).sum()),
               "holdout_rule_only": int((holdout_rows["rule_flag"] & ~holdout_rows["is_anomaly"]).sum()),
               "contamination_sensitivity": sensitivity,
               "random_seed": RANDOM_SEED, "n_estimators": N_ESTIMATORS, "contamination": CONTAMINATION,
               "features": FEATURES,
               "score_higher_means_more_unusual": True}
    return ordered, metrics
