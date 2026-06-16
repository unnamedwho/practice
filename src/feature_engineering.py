"""Invoice-level features: one complete invoice is one model observation."""
import numpy as np
import pandas as pd

FEATURES = ["log_total_gbp", "log_line_count", "log_total_quantity", "log_max_unit_price", "hour", "weekday", "has_customer"]


def invoice_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    sales = cleaned.loc[cleaned["category"].eq("sale")].copy()
    if sales.empty:
        return pd.DataFrame(columns=["InvoiceNo", "invoice_date", "first_line_date", "last_line_date",
            "country", "customer_id", "line_count", "total_gbp", "total_quantity", "max_unit_price",
            "customer_id_count", "country_count", "timestamp_count", "customer_id_invalid_rows",
            "time_variation", "invoice_conflict_reason", "eligible_for_model", *FEATURES])
    grouped = sales.groupby("InvoiceNo", sort=False, dropna=False)
    out = grouped.agg(first_line_date=("InvoiceDate", "min"), last_line_date=("InvoiceDate", "max"),
                      timestamp_count=("InvoiceDate", "nunique"), country=("Country", "first"),
                      country_count=("Country", "nunique"), customer_id_count=("CustomerID", "nunique"),
                      line_count=("source_row_id", "size"), total_gbp=("amount_gbp", "sum"),
                      total_quantity=("Quantity", "sum"), max_unit_price=("UnitPrice", "max"),
                      customer_id=("CustomerID", "first")).reset_index()
    if "customer_id_status" in sales:
        invalid_counts = sales.assign(_bad_customer=sales["customer_id_status"].eq("invalid").astype(int)).groupby("InvoiceNo")["_bad_customer"].sum()
        out["customer_id_invalid_rows"] = out["InvoiceNo"].map(invalid_counts).astype(int)
    else:
        out["customer_id_invalid_rows"] = 0
    out["time_variation"] = out["timestamp_count"].gt(1)
    long_span = (out["last_line_date"] - out["first_line_date"]).gt(pd.Timedelta(hours=24))
    out["invoice_conflict_reason"] = ""
    for mask, reason in ((out["customer_id_count"].gt(1), "multiple_customer_ids;"),
                         (out["country_count"].gt(1), "multiple_countries;"),
                         (long_span, "time_span_over_24h;"),
                         (out["customer_id_invalid_rows"].gt(0), "invalid_customer_id;")):
        out.loc[mask, "invoice_conflict_reason"] += reason
    finite_aggregates = np.isfinite(out[["total_gbp", "total_quantity", "max_unit_price"]].to_numpy(dtype="float64")).all(axis=1)
    out.loc[~finite_aggregates, "invoice_conflict_reason"] += "nonfinite_invoice_aggregate;"
    out["eligible_for_model"] = out["invoice_conflict_reason"].eq("")
    out.loc[out["customer_id_count"].gt(1), "customer_id"] = pd.NA
    out.loc[out["country_count"].gt(1), "country"] = pd.NA
    # An invoice is assumed complete in the input snapshot. The latest line time
    # is its availability time; small timestamp variation alone is not an error.
    out["invoice_date"] = out["last_line_date"]
    out["log_total_gbp"] = np.sign(out["total_gbp"]) * np.log1p(out["total_gbp"].abs())
    for source, target in (("line_count", "log_line_count"), ("total_quantity", "log_total_quantity"),
                           ("max_unit_price", "log_max_unit_price")):
        out[target] = np.log1p(out[source].clip(lower=0))
    out["hour"] = out["invoice_date"].dt.hour
    out["weekday"] = out["invoice_date"].dt.dayofweek
    out["has_customer"] = out["customer_id"].notna().astype(int)
    return out
