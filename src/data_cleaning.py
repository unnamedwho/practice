"""Conservative row validation that retains audit categories."""
import numpy as np
import pandas as pd

CLEANING_VERSION = "2.0"


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy().reset_index(drop=True)
    df.insert(0, "source_row_id", df.index.astype("int64"))
    missing = {str(k): int(v) for k, v in raw.isna().sum().items()}
    for col in ("InvoiceNo", "StockCode", "Description", "Country"):
        # Some public mirrors serialize absent descriptions as empty strings.
        missing[col] = int((raw[col].isna() | raw[col].astype("string").str.strip().eq("").fillna(False)).sum())
    duplicate = df.drop(columns="source_row_id").duplicated(keep="first")
    for col in ("InvoiceNo", "StockCode", "Description", "Country"):
        df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        df[col] = df[col].replace("", pd.NA)
    df["InvoiceNo"] = df["InvoiceNo"].str.upper()
    df["StockCode"] = df["StockCode"].str.upper()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=False)
    # CustomerID is optional. A supplied, malformed ID is retained as a row-level
    # review issue, but never silently converted into a legitimate missing ID.
    original_customer = df["CustomerID"]
    customer_missing = original_customer.isna() | original_customer.astype("string").str.strip().eq("").fillna(False)
    customer_number = pd.to_numeric(original_customer, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    with np.errstate(invalid="ignore"):
        # Beyond 2**53, conversion through a floating source may alter the ID.
        customer_valid = np.isfinite(customer_number) & (customer_number >= 1) & (customer_number <= 2**53) & (customer_number % 1 == 0)
    customer_invalid = ~customer_missing.to_numpy(dtype=bool) & ~customer_valid
    df["CustomerID"] = pd.Series(np.where(customer_valid, customer_number, np.nan), dtype="float64").astype("Int64")
    df["customer_id_status"] = np.select(
        [customer_missing.to_numpy(dtype=bool), customer_invalid], ["missing", "invalid"], default="valid"
    )
    cancelled = df["InvoiceNo"].str.match(r"^C\d{6}$", na=False)
    valid_invoice = df["InvoiceNo"].str.match(r"^(C)?\d{6}$", na=False)
    quantity = df["Quantity"].to_numpy(dtype="float64", na_value=np.nan)
    price = df["UnitPrice"].to_numpy(dtype="float64", na_value=np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        amount = quantity * price
        valid_qty = np.isfinite(quantity) & (quantity != 0) & (np.abs(quantity) <= 2**53) & (quantity % 1 == 0)
        valid_price = np.isfinite(price) & (price >= 0)
    valid_amount = np.isfinite(amount)
    amount_overflow = np.isfinite(quantity) & np.isfinite(price) & ~valid_amount
    # Zero prices can be promotional entries; retain but flag for review.
    essential = valid_invoice & df["StockCode"].notna() & df["InvoiceDate"].notna() & valid_qty & valid_price & valid_amount & df["Country"].notna()
    df["duplicate_of_earlier"] = duplicate
    df["cancelled_invoice"] = cancelled
    df["return_or_cancellation"] = cancelled | (quantity < 0)
    df["amount_gbp"] = amount
    df["invalid_reason"] = ""
    for failed, reason in ((~valid_invoice, "invoice_id;"), (df["StockCode"].isna(), "stock_code;"),
                           (df["InvoiceDate"].isna(), "date;"), (~np.isfinite(quantity), "quantity_nonfinite;"),
                           (np.isfinite(quantity) & ~valid_qty, "quantity_zero_nonintegral_or_out_of_range;"),
                           (~np.isfinite(price), "unit_price_nonfinite;"),
                           (np.isfinite(price) & ~valid_price, "unit_price_negative;"),
                           (amount_overflow, "amount_nonfinite;"),
                           (df["Country"].isna(), "country;"), (customer_invalid, "customer_id_invalid;")):
        df.loc[failed, "invalid_reason"] += reason
    df["category"] = "sale"
    df.loc[df["return_or_cancellation"], "category"] = "return_or_cancellation"
    df.loc[~essential, "category"] = "invalid"
    df["category_without_dedup"] = df["category"]
    df.loc[duplicate, "category"] = "duplicate"
    # Duplicate rows remain in the audit file; downstream retained rows exclude them.
    retained = df["category"].isin(["sale", "return_or_cancellation"])
    positive = df.loc[retained & df["amount_gbp"].gt(0), "amount_gbp"].abs()
    threshold = float(positive.quantile(.995)) if len(positive) else 0.0
    df["review_flag"] = retained & (df["UnitPrice"].eq(0) | df["amount_gbp"].abs().gt(threshold) | customer_invalid |
                                    (cancelled & df["Quantity"].gt(0)) | (~cancelled & df["Quantity"].lt(0)))
    retained_missing = {col: int(df.loc[retained, col].isna().sum()) for col in raw.columns}
    audit = {
        "initial_rows": int(len(df)), "missing_by_column": missing,
        "retained_missing_by_column": retained_missing,
        "exact_duplicates": int(duplicate.sum()), "invalid_rows_excluding_duplicates": int((~essential & ~duplicate).sum()),
        "customer_id_missing_rows": int(customer_missing.sum()),
        "customer_id_invalid_rows": int(customer_invalid.sum()),
        "nonfinite_quantity_rows": int((~np.isfinite(quantity)).sum()),
        "nonfinite_unit_price_rows": int((~np.isfinite(price)).sum()),
        "amount_overflow_rows": int(amount_overflow.sum()),
        "retained_rows": int(retained.sum()), "sale_rows": int(df["category"].eq("sale").sum()),
        "return_rows": int(df["category"].eq("return_or_cancellation").sum()),
        "review_flag_rows": int(df["review_flag"].sum()), "large_amount_threshold_gbp": threshold,
        "steps": ["Preserved original row IDs", "Marked exact raw duplicates, retaining first occurrence",
                  "Normalized text and parsed date/numeric fields", "Validated identifiers, quantity, price, and essential fields",
                  "Retained absent CustomerID; flagged supplied invalid IDs without imputing identity",
                  "Rejected nonfinite quantities, prices, and amounts from retained categories",
                  "Retained zero prices; separated sales, returns/cancellations, and invalid records",
                  "Flagged unusual amounts and sign inconsistencies for review"]}
    return df, audit
