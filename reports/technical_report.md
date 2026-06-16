# Technical report

**Scope:** Educational prototype using public UCI Online Retail data, independent of ARQA BUH. Source SHA-256: `0663e4d42cda4401cd76ee44bb278f3b23f02f576399925d4701a6cf07a9591c`.

## Processing and data quality

All 541,909 source lines remain in the audit output. The cleaning policy separates 5,268 exact duplicates and 3 other invalid rows, retaining 526,051 regular-sale and 10,587 return/cancellation lines. Missing CustomerID is allowed; 0 supplied malformed IDs are flagged and their invoices withheld from ML. Nonfinite numeric inputs and overflow amounts are invalid. Duplicate removal is an assumption because identical lines may be real repeated positions. Its measured effect on sums and model flags appears in the two detailed reports.

## Invoice and model method

The input is treated as a complete invoice snapshot. The latest line time is the completion/availability time. Multiple line timestamps within 24 hours are permitted; spans beyond 24 hours, conflicting known customers or countries, and malformed customer IDs trigger review rather than a silent first-value choice. 42 invoices have multiple times, and 0 invoices are excluded for consistency issues. The 16,581 training invoices end at 2011-10-26 12:28:00; the 4,144 holdout invoices begin at 2011-10-26 12:31:00. Preprocessing and the 200-tree Isolation Forest fit only on training invoices.

The model flags 332 training and 119 holdout invoices. On holdout, the training-derived amount rule flags 92; 64 overlap. A model score is a review priority, not an explanation or confirmation of fraud. Contamination sensitivity and duplicate sensitivity are in [ml_evaluation_report.md](ml_evaluation_report.md).

## Storage and limits

The executed database dialect is `sqlite` with 541,909 audited rows and 20,725 predictions. PostgreSQL was not tested in this run; the isolated integration test requires TEST_POSTGRES_URL. The public UK retail data has no verified fraud labels, so precision, recall, F1, and fraud accuracy are unavailable. The Parquet mirror stores unit prices as float32, which may produce small monetary differences from the official source.
