"""Descriptive checks for invoice consistency and duplicate sensitivity."""
from pathlib import Path
import pandas as pd
from .feature_engineering import invoice_features
from .predict import predict_invoices


def invoice_consistency_summary(invoices: pd.DataFrame) -> dict:
    return {
        "sale_invoices_before_exclusion": int(len(invoices)),
        "multiple_customer_id_invoices": int(invoices["customer_id_count"].gt(1).sum()),
        "multiple_country_invoices": int(invoices["country_count"].gt(1).sum()),
        "multiple_timestamp_invoices": int(invoices["time_variation"].sum()),
        "time_span_over_24h_invoices": int(invoices["invoice_conflict_reason"].str.contains("time_span_over_24h", regex=False).sum()),
        "invalid_customer_id_invoices": int(invoices["customer_id_invalid_rows"].gt(0).sum()),
        "nonfinite_invoice_aggregate_invoices": int(invoices["invoice_conflict_reason"].str.contains("nonfinite_invoice_aggregate", regex=False).sum()),
        "excluded_from_model": int((~invoices["eligible_for_model"]).sum()),
        "date_rule": "latest line timestamp, assuming the input snapshot contains the complete invoice",
        "time_rule": "multiple timestamps within 24 hours are allowed; spans over 24 hours require review and are excluded",
    }


def duplicate_sensitivity(cleaned: pd.DataFrame, scored: pd.DataFrame, model_path: Path,
                          train_through: str, rule_threshold_gbp: float) -> dict:
    restorables = cleaned["category"].eq("duplicate") & cleaned["category_without_dedup"].isin(["sale", "return_or_cancellation"])
    alternate = cleaned.copy()
    alternate.loc[restorables, "category"] = alternate.loc[restorables, "category_without_dedup"]
    alt_invoices = invoice_features(alternate)
    alt_eligible = alt_invoices.loc[alt_invoices["eligible_for_model"]]
    alt_scores = predict_invoices(alt_eligible, model_path)
    current_flags = set(scored.loc[scored["is_anomaly"], "InvoiceNo"])
    alternate_flags = set(alt_scores.loc[alt_scores["is_anomaly"], "InvoiceNo"])
    boundary = pd.Timestamp(train_through)
    alt_holdout = alt_eligible.loc[alt_eligible["invoice_date"].gt(boundary)]
    alt_holdout_flags = alt_scores.set_index("InvoiceNo").loc[alt_holdout["InvoiceNo"], "is_anomaly"]
    return {
        "assumption": "Exact duplicate lines are removed after the first; some may be legitimate repeated items.",
        "restored_duplicate_lines": int(restorables.sum()),
        "restored_sale_lines": int((restorables & cleaned["category_without_dedup"].eq("sale")).sum()),
        "sale_total_gbp_deduplicated": float(cleaned.loc[cleaned["category"].eq("sale"), "amount_gbp"].sum()),
        "sale_total_gbp_with_duplicate_lines": float(alternate.loc[alternate["category"].eq("sale"), "amount_gbp"].sum()),
        "net_total_gbp_deduplicated": float(cleaned.loc[cleaned["category"].isin(["sale", "return_or_cancellation"]), "amount_gbp"].sum()),
        "net_total_gbp_with_duplicate_lines": float(alternate.loc[alternate["category"].isin(["sale", "return_or_cancellation"]), "amount_gbp"].sum()),
        "model_flags_deduplicated": len(current_flags),
        "model_flags_with_duplicate_lines_fixed_model": len(alternate_flags),
        "flags_only_with_duplicate_lines": len(alternate_flags - current_flags),
        "flags_only_after_deduplication": len(current_flags - alternate_flags),
        "holdout_model_flags_with_duplicate_lines_fixed_model": int(alt_holdout_flags.sum()),
        "holdout_rule_flags_with_duplicate_lines": int(alt_holdout["total_gbp"].gt(rule_threshold_gbp).sum()),
        "interpretation": "The same model trained on deduplicated invoices scores alternate aggregates; this isolates input sensitivity, not retrained model quality.",
    }
