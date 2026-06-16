# Financial Transaction Data Processing and Machine Learning-Based Anomaly Detection

An educational Information Systems portfolio prototype inspired by financial information system workflows. It uses the **public UCI Online Retail dataset (ID 352)**. It is independent of ARQA BUH and uses no company data or requirements. The source is retail invoice lines in GBP, so this is a demonstration of transaction processing rather than an accounting or fraud production system.

## Motivation and architecture

Reliable transaction analysis needs an auditable distinction between bad records, legitimate returns, and unusual values. The pipeline reads raw invoice lines, records quality findings, preserves all source rows with audit categories, builds sale-invoice features, fits an Isolation Forest on earlier invoices, scores later invoices, and writes reports and database tables.

```text
data/raw/ -> src/data_loader.py -> src/data_cleaning.py -> data/processed/transactions_audit.parquet
                                  -> src/feature_engineering.py -> src/train_model.py
                                  -> data/processed/invoice_predictions.parquet
                                  -> models/isolation_forest.joblib
                                  -> reports/ and reports/figures/
                                  -> src/database.py -> SQLite or PostgreSQL
```

The original [project.ipynb](project.ipynb) is retained and extended; [project_original.ipynb](project_original.ipynb) is a backup of its starting state. Run from the repository root. The [technical report](reports/technical_report.md), [data quality report](reports/data_quality_report.md), and [ML evaluation report](reports/ml_evaluation_report.md) explain executed results.

## Data source

UCI Machine Learning Repository, [Online Retail (ID 352)](https://archive.ics.uci.edu/dataset/352/online+retail), DOI `10.24432/C5BW33`. Source records span December 2010 through December 2011. The executed run uses a [public Parquet mirror](https://huggingface.co/datasets/pavan-naik/Online-Retail) of the 541,909-row dataset saved as `data/raw/online_retail_mirror.parquet` because the official UCI transfer was extremely slow. That mirror stores prices as `float32` and missing descriptions as the literal string `nan`; the loader restores those descriptions to missing values. Monetary results may differ slightly from an official CSV/XLSX run due to source precision. To use the official source from a clean checkout, download it once and choose it explicitly:

```bash
mkdir -p data/raw
curl -L --fail 'https://archive.ics.uci.edu/static/public/352/data.csv' -o data/raw/data.csv
python -m src.run_pipeline --raw data/raw/data.csv
```

Raw files are never modified. The source must contain InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country. The original notebook used `ucimlrepo.data.features`, which omits InvoiceNo and StockCode because UCI classifies them as index columns. This pipeline loads all eight columns.

## Stack and installation

The checked run used Python **3.14.7** in `.venv`. Use [requirements-tested.txt](requirements-tested.txt) for the exact verified direct dependency versions on a compatible platform; [reproducibility_manifest.json](data/processed/reproducibility_manifest.json) records all installed distribution versions, source and model checksums, model settings, and the time split. [requirements.txt](requirements.txt) keeps supported version ranges. On macOS in VS Code, select the environment used for installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run EDA, cleaning, training, predictions, and storage

```bash
python -m src.run_pipeline
```

This one command runs the full pipeline and writes EDA summaries, quality reports, model predictions, saved figures, and a SQLite validation database. Use `--raw 'data/raw/Online Retail.xlsx'` to choose a source explicitly. Use `--no-database` to skip database loading. Re-running replaces generated outputs, but does not change raw data or the original notebook backup.

To refresh and execute the notebook from the latest pipeline output, run `python -m src.build_notebook`, then open `project.ipynb` in VS Code. The builder registers a kernel for the active `.venv`, keeping notebook execution aligned with the pipeline environment. This command rewrites the generated walkthrough notebook; `project_original.ipynb` preserves the starting work.

The audit Parquet contains every source row with `source_row_id`, `category`, `invalid_reason`, `duplicate_of_earlier`, `review_flag`, and `amount_gbp`. The categories are `sale`, `return_or_cancellation`, `invalid`, and `duplicate`. Excluding identical rows after the first is an **assumption**: some identical lines may be legitimate repeated positions. The [quality report](reports/data_quality_report.md) and [ML report](reports/ml_evaluation_report.md) quantify its effect. Missing CustomerID is accepted. A supplied fractional, infinite, nonpositive, out-of-range, or nonnumeric CustomerID is marked `invalid` in `customer_id_status`, recorded as `customer_id_invalid` in the audit reason, and converted to missing for numeric use. IDs and quantities above 2^53 are rejected because float-source integer precision is not reliable there. The sale line with a malformed CustomerID is retained for audit, but its invoice is excluded from ML pending review. Nonfinite quantity, unit price, and line amount (including arithmetic overflow) make a row invalid. Nonfinite invoice aggregates also exclude an invoice from ML. Zero-price rows remain and receive a review flag. Returns and cancelled invoices remain in the audit output and database, but are not modeled as sale invoices.

Invoice-level features are log-transformed total amount, line count, total quantity, maximum unit price, hour, weekday, and whether CustomerID exists. Identifiers are never used as numeric predictors. Invoice consistency checks count distinct known customers, countries, and timestamps. Conflicting customers or countries and a time span exceeding 24 hours are marked for review and excluded from ML; small time differences between lines are allowed. Assuming the input snapshot contains a complete invoice, its **latest line timestamp** is its completion/availability time. The chronological split assigns all invoices at the same timestamp to one subset. The imputer, scaler, and Isolation Forest fit only on train. The model uses seed 42 and a 2% contamination assumption. `models/isolation_forest.joblib` is reusable; call `src.predict.predict_invoices(invoice_features(new_cleaned_rows), model_path)` on conflict-free sale invoices. Inputs must follow the same cleaning and feature definitions.

## Database

Without `DATABASE_URL`, the pipeline builds `data/processed/transactions.sqlite` and verifies stored counts. For PostgreSQL, create a database, set `DATABASE_URL` using [.env.example](.env.example), then run the same command. For example:

```bash
export DATABASE_URL='postgresql+psycopg://username:password@localhost:5432/financial_anomalies'
python -m src.run_pipeline
```

SQLAlchemy creates the tables and loads all audited rows, quality metrics, and invoice predictions. Deletion, insertion, and count verification happen in **one transaction on one connection**; a failed reload rolls back to the prior data. See [schema.sql](sql/schema.sql) and [analysis_queries.sql](sql/analysis_queries.sql) for PostgreSQL reference SQL and six analytical/verification queries. Credentials belong only in environment variables. Point `DATABASE_URL` at a dedicated project database because a successful run replaces the project's tables.

PostgreSQL was **not tested in this environment**: no server or `TEST_POSTGRES_URL` was available. For an integration run, start a local PostgreSQL server, create a disposable database and a user allowed to create schemas, install [requirements-tested.txt](requirements-tested.txt), then run:

```bash
export TEST_POSTGRES_URL='postgresql+psycopg://username:password@localhost:5432/financial_anomalies_test'
python -m pytest -q tests/test_postgres_integration.py
```

The test creates a uniquely named schema in that database, verifies loading, repeat loading, the invoice foreign key and stored contents, injects a failed reload and checks rollback, then removes **only its own schema**. Use `DATABASE_URL` separately for a full PostgreSQL pipeline run after this check. Never put credentials into tracked files or the archive.

## Tests

```bash
python -m pytest -q
```

The tests cover loading, empty input, duplicate and missing/invalid CustomerID handling, nonfinite values and amount overflow, returns, invoice conflicts and permitted time variation, timestamp-grouped chronological splitting, prediction equality before/after model serialization, and SQLite repeat-load rollback. The optional PostgreSQL test is skipped unless `TEST_POSTGRES_URL` is set.

## Actual results

The executed mirror run has 541,909 rows and eight columns, covering 2010-12-01 08:26 to 2011-12-09 12:50. There are 25,900 source invoices, 4,372 identified customers, 38 countries, 10,624 negative-quantity rows, and 9,288 rows with cancellation invoice IDs. Before cleaning, 1,454 descriptions and 135,080 customer IDs are missing. With all eight columns included, there are 5,268 exact duplicate rows; the original notebook's 6,007 count used only six feature columns and therefore counted different invoices with identical features as duplicates.

After duplicate and invalid-row separation, 536,638 lines are retained: 526,051 sale lines and 10,587 return/cancellation lines. Three rows are invalid apart from duplicates. Among 20,725 sale invoices, 42 have multiple timestamps within the allowed 24-hour span; none has a conflicting customer or country, and none was excluded from this run's ML input. The strict time split trains on **16,581** invoices through **2011-10-26 12:28** and holds out **4,144** from **2011-10-26 12:31**. The model flags **332** training and **119** holdout invoices. On holdout, the amount rule flags **92**, with **64** overlapping model flags. Using train-only thresholds of 1%, 2%, and 5% contamination on the same forest gives **71**, **119**, and **193** holdout flags.

Removing identical lines changes the sale total from **£10,655,622.43** with duplicate lines to **£10,631,048.69** after deduplication. In this run, the fixed trained model flags **451** invoices in either scenario; the absence of flag changes does not prove that deduplication is correct. SQLite stores and verifies 541,909 audit rows and 20,725 invoice and prediction rows. Eight charts were generated. These figures are reproduced in [run_summary.json](data/processed/run_summary.json), [eda_summary.json](data/processed/eda_summary.json), and the reports. Because the public dataset has no verified anomaly labels, precision, recall, F1, and fraud-detection accuracy are unavailable. A high model score indicates rarity, not fraud.

## Limitations and future work

This public UK retail dataset differs from Kazakhstan accounting data. The model excludes returns from sale-invoice scoring, and contamination is an operating assumption. A real deployment would need domain-specific validation, currency handling, customer review feedback, labeled cases where possible, drift monitoring, access controls, and scheduled ingestion. The observed amount or line count of a flagged invoice is a descriptive clue, not a confirmed cause of its model score.

## Shareable archive

Run `python -m src.package_project` to create [financial_anomaly_project1_clean.zip](dist/financial_anomaly_project1_clean.zip). It includes the local raw mirror, generated results, model, notebook, reports, code, and tests. It excludes `.venv`, `.git`, cache folders, `__MACOSX`, `.env`, and the archive itself. The working environment and Git history are untouched.
