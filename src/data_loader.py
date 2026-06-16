"""Load and preserve the public UCI Online Retail source data."""
from pathlib import Path
import pandas as pd

REQUIRED = {"InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"}


def load_transactions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset missing: {path}")
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, dtype={"InvoiceNo": "string", "StockCode": "string"})
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"InvoiceNo": "string", "StockCode": "string"}, low_memory=False)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        # This public mirror serialized the original 1,454 missing descriptions
        # as the literal string "nan". Restore their missing-value meaning.
        frame["Description"] = frame["Description"].replace("nan", pd.NA)
    else:
        raise ValueError("Expected a CSV, XLSX, or Parquet file")
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(REQUIRED - set(frame.columns))}")
    return frame[list(REQUIRED_ORDER)].copy()


REQUIRED_ORDER = ("InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country")


def find_raw_data(root: Path) -> Path:
    for name in ("Online Retail.xlsx", "Online Retail.csv", "online_retail_mirror.parquet", "data.csv"):
        for folder in (root / "data" / "raw", root):
            candidate = folder / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError("Place the public UCI Online Retail CSV/XLSX in data/raw/. See README.")
