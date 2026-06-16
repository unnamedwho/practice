"""End-to-end data, model, plots, reports, and database CLI."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data_loader import find_raw_data, load_transactions
from .data_cleaning import clean_transactions
from .feature_engineering import invoice_features
from .train_model import train_and_score
from .database import load_database
from .methodology import duplicate_sensitivity, invoice_consistency_summary
from .reproducibility import write_manifest


def save_plot(path: Path, title: str, xlabel: str, ylabel: str = "Count") -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_figures(raw: pd.DataFrame, cleaned: pd.DataFrame, scored: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    sales = cleaned.loc[cleaned.category.eq("sale")]
    plt.figure(figsize=(8, 4)); plt.hist(np.log1p(sales.amount_gbp.clip(lower=0)), bins=70)
    save_plot(folder / "sale_amount_log.png", "Sale line amount distribution", "log(1 + amount in GBP)")
    plt.figure(figsize=(8, 4)); plt.hist(cleaned.Quantity.dropna().clip(-100, 100), bins=80)
    save_plot(folder / "quantity.png", "Quantity distribution (clipped for display)", "Units, clipped to [-100, 100]")
    plt.figure(figsize=(8, 4)); plt.hist(cleaned.UnitPrice.dropna().clip(0, 100), bins=80)
    save_plot(folder / "unit_price.png", "Unit price distribution (clipped for display)", "GBP, clipped to [0, 100]")
    plt.figure(figsize=(9, 5)); cleaned.Country.value_counts().head(15).sort_values().plot.barh()
    plt.xscale("log")
    save_plot(folder / "countries.png", "Top 15 countries by source rows", "Rows (log scale)")
    plt.figure(figsize=(8, 4)); plt.hist(scored.anomaly_score, bins=70)
    save_plot(folder / "anomaly_scores.png", "Isolation Forest invoice scores", "Score (higher = more unusual)")
    plt.figure(figsize=(8, 4))
    for flagged, label in ((False, "Not flagged"), (True, "Flagged")):
        vals = np.log1p(scored.loc[scored.is_anomaly.eq(flagged), "total_gbp"].clip(lower=0))
        plt.hist(vals, bins=55, alpha=.65, label=label, density=True)
    plt.legend(); save_plot(folder / "invoice_amount_comparison.png", "Invoice total by model flag", "log(1 + invoice total in GBP)", "Density")
    plt.figure(figsize=(8, 4)); scored.groupby("country").is_anomaly.sum().sort_values(ascending=False).head(12).sort_values().plot.barh()
    save_plot(folder / "anomalies_by_country.png", "Flagged invoices by country", "Invoices")
    plt.figure(figsize=(8, 4));
    for flagged, label in ((False, "Not flagged"), (True, "Flagged")):
        plt.hist(np.log1p(scored.loc[scored.is_anomaly.eq(flagged), "line_count"]), bins=35, alpha=.6, label=label, density=True)
    plt.legend(); save_plot(folder / "line_count_comparison.png", "Line count by model flag", "log(1 + invoice lines)", "Density")


def run(root: Path, raw_path: Path | None = None, database: bool = True) -> dict:
    root = root.resolve()
    raw_path = raw_path or find_raw_data(root)
    raw = load_transactions(raw_path)
    cleaned, audit = clean_transactions(raw)
    invoices = invoice_features(cleaned)
    consistency = invoice_consistency_summary(invoices)
    eligible_invoices = invoices.loc[invoices["eligible_for_model"]].copy()
    model_path = root / "models" / "isolation_forest.joblib"
    scored, metrics = train_and_score(eligible_invoices, model_path)
    duplicate_impact = duplicate_sensitivity(cleaned, scored, model_path, metrics["train_through"], metrics["rule_threshold_gbp"])
    processed = root / "data" / "processed"
    reports = root / "reports"
    processed.mkdir(parents=True, exist_ok=True); reports.mkdir(exist_ok=True)
    cleaned.to_parquet(processed / "transactions_audit.parquet", index=False)
    invoices.to_parquet(processed / "invoice_quality.parquet", index=False)
    scored.to_parquet(processed / "invoice_predictions.parquet", index=False)
    pd.DataFrame([audit]).to_json(processed / "quality_summary.json", orient="records", indent=2)
    scored.sort_values("anomaly_score", ascending=False).head(25)[["InvoiceNo", "invoice_date", "country", "total_gbp", "line_count", "anomaly_score", "is_anomaly"]].to_csv(reports / "top_anomalous_invoices.csv", index=False)
    make_figures(raw, cleaned, scored, reports / "figures")
    dates = pd.to_datetime(raw.InvoiceDate, errors="coerce")
    eda = {"shape": list(raw.shape), "dtypes": {k: str(v) for k, v in raw.dtypes.items()},
           "date_min": str(dates.min()), "date_max": str(dates.max()),
           "invoice_count": int(raw.InvoiceNo.nunique()), "customer_count": int(raw.CustomerID.nunique()),
           "country_count": int(raw.Country.nunique()), "negative_quantity": int(pd.to_numeric(raw.Quantity, errors="coerce").lt(0).sum()),
           "cancelled_invoice_rows": int(raw.InvoiceNo.astype("string").str.match(r"^[Cc]\d{6}$", na=False).sum()),
           "quantity_describe": raw.Quantity.describe().to_dict(), "price_describe": raw.UnitPrice.describe().to_dict(),
           "country_top10": raw.Country.value_counts().head(10).to_dict(),
           "first_records": raw.head(5).astype(str).to_dict(orient="records"),
           "last_records": raw.tail(5).astype(str).to_dict(orient="records")}
    (processed / "eda_summary.json").write_text(json.dumps(eda, indent=2, default=str))
    db = load_database(cleaned, scored, audit, metrics, root) if database else {"status": "skipped"}
    manifest = write_manifest(root, raw_path, model_path, metrics)
    result = {"audit": audit, "invoice_consistency": consistency, "duplicate_sensitivity": duplicate_impact,
              "model": metrics, "database": db, "postgresql_tested": db.get("database_url_kind") == "postgresql",
              "source_sha256": manifest["source_sha256"]}
    (processed / "run_summary.json").write_text(json.dumps(result, indent=2))
    quality_text = f"""# Data quality report

Source: UCI Online Retail (ID 352), local file `{raw_path.name}`. Source SHA-256: `{manifest['source_sha256']}`.

| Measure | Before | Retained after cleaning |
|---|---:|---:|
| Rows | {audit['initial_rows']:,} | {audit['retained_rows']:,} |
| Exact duplicates | {audit['exact_duplicates']:,} | 0 |
| Invalid rows excluding duplicates | {audit['invalid_rows_excluding_duplicates']:,} | 0 |
| Returns/cancellations | — | {audit['return_rows']:,} |
| Regular sale rows | — | {audit['sale_rows']:,} |
| Review flags | — | {audit['review_flag_rows']:,} |

Missing values by column before cleaning: `{json.dumps(audit['missing_by_column'])}`.

Missing values among retained rows: `{json.dumps(audit['retained_missing_by_column'])}`.

Supplied invalid CustomerID values: {audit['customer_id_invalid_rows']:,}; missing CustomerID values: {audit['customer_id_missing_rows']:,}. A missing ID is accepted. A supplied nonintegral, nonfinite, nonpositive, out-of-exact-range, or nonnumeric ID is marked `customer_id_invalid` in the audit, converted to missing for numeric operations, and its invoice is excluded from modeling pending review. No identity is invented. Quantities beyond the exactly representable integer range (2^53) are invalid. Nonfinite quantity rows: {audit['nonfinite_quantity_rows']:,}; nonfinite unit-price rows: {audit['nonfinite_unit_price_rows']:,}; line-amount overflows: {audit['amount_overflow_rows']:,}. These cannot enter valid sale or return categories. {consistency['nonfinite_invoice_aggregate_invoices']:,} invoices with nonfinite aggregate values are excluded from ML.

Invoice consistency: {consistency['multiple_customer_id_invoices']:,} invoices with multiple known customer IDs, {consistency['multiple_country_invoices']:,} with multiple countries, {consistency['multiple_timestamp_invoices']:,} with multiple timestamps, {consistency['time_span_over_24h_invoices']:,} spanning more than 24 hours, and {consistency['excluded_from_model']:,} excluded from modeling. Different line times within 24 hours are allowed. The invoice date is the latest line timestamp, assuming the input snapshot contains the complete invoice. Conflicting customer or country values are not silently chosen; those invoices require review and are excluded from the model.

Exact duplicate removal is an **assumption**: identical lines may be legitimate repeated items. Keeping the {duplicate_impact['restored_duplicate_lines']:,} otherwise valid duplicate lines changes sale total from £{duplicate_impact['sale_total_gbp_deduplicated']:,.2f} to £{duplicate_impact['sale_total_gbp_with_duplicate_lines']:,.2f}, and signed net total from £{duplicate_impact['net_total_gbp_deduplicated']:,.2f} to £{duplicate_impact['net_total_gbp_with_duplicate_lines']:,.2f}. The ML sensitivity is documented in the ML report. Raw data remains untouched, and all rows remain in the audit Parquet.
"""
    (reports / "data_quality_report.md").write_text(quality_text)
    top = scored.sort_values("anomaly_score", ascending=False).head(5)
    top_lines = "\n".join(f"- {r.InvoiceNo}: £{r.total_gbp:,.2f}, {r.line_count} lines, score {r.anomaly_score:.3f}" for r in top.itertuples())
    sensitivity_lines = "\n".join(
        f"| {level} | {values['train_flagged']:,} | {values['holdout_flagged']:,} |"
        for level, values in metrics["contamination_sensitivity"].items()
    )
    ml_text = f"""# ML evaluation report

This educational prototype models complete sale invoices using public retail data. The Isolation Forest has {metrics['n_estimators']} trees, seed {metrics['random_seed']}, and contamination {metrics['contamination']:.2f}. All preprocessing is fitted only on {metrics['train_invoices']:,} training invoices. Training ends at **{metrics['train_through']}** and the {metrics['holdout_invoices']:,} holdout invoices begin at **{metrics['holdout_from']}**; no timestamp appears in both subsets.

| Holdout comparison | Invoices |
|---|---:|
| Isolation Forest flags | {metrics['holdout_flagged']:,} |
| Amount-rule flags | {metrics['holdout_rule_flagged']:,} |
| Both | {metrics['holdout_rule_model_overlap']:,} |
| Model only | {metrics['holdout_model_only']:,} |
| Rule only | {metrics['holdout_rule_only']:,} |

The amount rule is `total_gbp > £{metrics['rule_threshold_gbp']:,.2f}`, with the 99th-percentile threshold learned from training invoices. The model flags {metrics['train_flagged']:,} training invoices. These counts compare review queues, not predictive accuracy.

Sensitivity using the **same fitted forest** and different thresholds derived only from training scores:

| Assumed contamination | Training flags | Holdout flags |
|---:|---:|---:|
{sensitivity_lines}

If otherwise valid exact duplicate lines are retained, the fixed trained model flags {duplicate_impact['model_flags_with_duplicate_lines_fixed_model']:,} invoices versus {duplicate_impact['model_flags_deduplicated']:,} after deduplication. {duplicate_impact['flags_only_with_duplicate_lines']:,} flags appear only with duplicate lines and {duplicate_impact['flags_only_after_deduplication']:,} only after deduplication. In the alternate holdout, the fixed model flags {duplicate_impact['holdout_model_flags_with_duplicate_lines_fixed_model']:,} invoices and the same amount rule flags {duplicate_impact['holdout_rule_flags_with_duplicate_lines']:,}. This changes inputs to one fixed model; it is not a retraining experiment.

Top invoices for manual review:
{top_lines}

The listed amounts and line counts are observable characteristics, not confirmed reasons for a flag. Scores indicate statistical rarity, not fraud. There are no verified anomaly labels, so precision, recall, F1 and fraud-detection accuracy cannot be calculated. Returns are audited separately and are outside this sale-invoice model. A chronological holdout measures flag stability, not classification quality.
"""
    (reports / "ml_evaluation_report.md").write_text(ml_text)
    technical = f"""# Technical report

**Scope:** Educational prototype using public UCI Online Retail data, independent of ARQA BUH. Source SHA-256: `{manifest['source_sha256']}`.

## Processing and data quality

All {audit['initial_rows']:,} source lines remain in the audit output. The cleaning policy separates {audit['exact_duplicates']:,} exact duplicates and {audit['invalid_rows_excluding_duplicates']:,} other invalid rows, retaining {audit['sale_rows']:,} regular-sale and {audit['return_rows']:,} return/cancellation lines. Missing CustomerID is allowed; {audit['customer_id_invalid_rows']:,} supplied malformed IDs are flagged and their invoices withheld from ML. Nonfinite numeric inputs and overflow amounts are invalid. Duplicate removal is an assumption because identical lines may be real repeated positions. Its measured effect on sums and model flags appears in the two detailed reports.

## Invoice and model method

The input is treated as a complete invoice snapshot. The latest line time is the completion/availability time. Multiple line timestamps within 24 hours are permitted; spans beyond 24 hours, conflicting known customers or countries, and malformed customer IDs trigger review rather than a silent first-value choice. {consistency['multiple_timestamp_invoices']:,} invoices have multiple times, and {consistency['excluded_from_model']:,} invoices are excluded for consistency issues. The {metrics['train_invoices']:,} training invoices end at {metrics['train_through']}; the {metrics['holdout_invoices']:,} holdout invoices begin at {metrics['holdout_from']}. Preprocessing and the 200-tree Isolation Forest fit only on training invoices.

The model flags {metrics['train_flagged']:,} training and {metrics['holdout_flagged']:,} holdout invoices. On holdout, the training-derived amount rule flags {metrics['holdout_rule_flagged']:,}; {metrics['holdout_rule_model_overlap']:,} overlap. A model score is a review priority, not an explanation or confirmation of fraud. Contamination sensitivity and duplicate sensitivity are in [ml_evaluation_report.md](ml_evaluation_report.md).

## Storage and limits

The executed database dialect is `{db.get('database_url_kind', 'none')}` with {db.get('stored_transactions', 0):,} audited rows and {db.get('stored_predictions', 0):,} predictions. PostgreSQL {'was exercised in this run' if db.get('database_url_kind') == 'postgresql' else 'was not tested in this run; the isolated integration test requires TEST_POSTGRES_URL'}. The public UK retail data has no verified fraud labels, so precision, recall, F1, and fraud accuracy are unavailable. The Parquet mirror stores unit prices as float32, which may produce small monetary differences from the official source.
"""
    (reports / "technical_report.md").write_text(technical)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--no-database", action="store_true")
    args = parser.parse_args()
    run(args.root, args.raw, not args.no_database)


if __name__ == "__main__":
    main()
