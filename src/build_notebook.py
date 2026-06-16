"""Rebuild the original notebook as an executed project walkthrough after a pipeline run."""
import json
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient


def build(root: Path) -> None:
    summary = json.loads((root / "data/processed/run_summary.json").read_text())
    audit, model = summary["audit"], summary["model"]
    consistency, duplicate = summary["invoice_consistency"], summary["duplicate_sensitivity"]
    kernel_name = "financial-project-venv"
    kernel_dir = Path(sys.prefix) / "share" / "jupyter" / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Financial Project (.venv)", "language": "python"
    }))
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell("# Financial Transaction Data Processing and ML-Based Anomaly Detection\n\nEducational prototype using public UCI Online Retail data (ID 352). The original starting notebook is preserved in `project_original.ipynb`. The identifiers omitted by `ucimlrepo.data.features` are loaded here from the full local source."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport sys\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nsys.path.insert(0, str(ROOT))\nfrom src.data_loader import find_raw_data, load_transactions\nfrom src.feature_engineering import invoice_features\nraw_path = find_raw_data(ROOT)\ndf = load_transactions(raw_path)\nprint('Source:', raw_path.name, 'Shape:', df.shape)"),
        nbf.v4.new_markdown_cell("## Raw data exploration\n\nThe next cell reports the actual dimensions, types, first and last records, missingness, duplicates, distributions, date range, invoices, customers, countries, negative quantities, and cancellation markers. Extreme values are reported without deleting them."),
        nbf.v4.new_code_cell("display(df.head(), df.tail())\nprint('Types:')\nprint(df.dtypes)\nprint('Missing by column:')\nprint(df.isna().sum())\nprint('Exact duplicates:', df.duplicated().sum())\nprint('Numeric summary:')\ndisplay(df[['Quantity', 'UnitPrice']].describe())\ndates = pd.to_datetime(df.InvoiceDate, errors='coerce')\nprint('Date range:', dates.min(), dates.max())\nprint('Invoices:', df.InvoiceNo.nunique(), 'Customers:', df.CustomerID.nunique())\nprint('Negative quantities:', (df.Quantity < 0).sum())\nprint('Cancelled invoice rows:', df.InvoiceNo.astype('string').str.match(r'^[Cc]\\d{6}$', na=False).sum())\nprint('Top countries:')\ndisplay(df.Country.value_counts().head(15))\nprint('Customer line count summary:')\ndisplay(df.groupby('CustomerID', dropna=True).size().describe())"),
        nbf.v4.new_markdown_cell(f"## Observed data quality findings\n\nThe executed pipeline found **{audit['initial_rows']:,}** source lines, **{audit['exact_duplicates']:,}** exact duplicates, and **{audit['invalid_rows_excluding_duplicates']:,}** invalid lines excluding duplicates. It retained **{audit['retained_rows']:,}** lines: **{audit['sale_rows']:,}** sales and **{audit['return_rows']:,}** returns/cancellations. **{audit['review_flag_rows']:,}** retained lines received review flags. Missing values before cleaning: `{audit['missing_by_column']}`. These figures come from the executed pipeline output."),
        nbf.v4.new_code_cell("cleaned = pd.read_parquet(ROOT / 'data/processed/transactions_audit.parquet')\nprint('Audit categories:')\ndisplay(cleaned.category.value_counts())\nprint('Review examples:')\ndisplay(cleaned.loc[cleaned.review_flag, ['source_row_id','InvoiceNo','Quantity','UnitPrice','amount_gbp','category']].head(10))\nprint('Invalid reasons:')\ndisplay(cleaned.loc[cleaned.category.eq('invalid'), 'invalid_reason'].value_counts().head(10))"),
        nbf.v4.new_markdown_cell(f"## Cleaning decisions and invoice consistency\n\nRaw rows remain untouched. Exact duplicate exclusion is an assumption because identical lines could be genuine repeated items. Missing CustomerID is accepted; a supplied malformed ID is audited and its invoice withheld from ML. Nonfinite quantity, price, or amount is invalid. Returns remain a separate category. Of {consistency['sale_invoices_before_exclusion']:,} sale invoices, {consistency['multiple_timestamp_invoices']:,} have multiple line times; variation within 24 hours is allowed. Conflicting known customers/countries or a span beyond 24 hours is flagged, and {consistency['excluded_from_model']:,} invoices are excluded. Assuming complete invoices, the latest line time is the model date."),
        nbf.v4.new_code_cell("invoices = pd.read_parquet(ROOT / 'data/processed/invoice_quality.parquet')\nscored = pd.read_parquet(ROOT / 'data/processed/invoice_predictions.parquet')\nprint('Sale invoices:', len(invoices), 'Eligible:', invoices.eligible_for_model.sum())\nprint('Invoice conflict reasons:')\ndisplay(invoices.invoice_conflict_reason.value_counts().head(10))\nprint('Model flags by subset:')\ndisplay(scored.groupby('subset').is_anomaly.agg(['sum','count']))\nprint('Holdout model/rule comparison:')\ndisplay(scored.loc[scored.subset.eq('holdout')].groupby(['is_anomaly','rule_flag']).size())\nprint('Highest anomaly scores:')\ndisplay(scored.sort_values('anomaly_score', ascending=False)[['InvoiceNo','invoice_date','country','total_gbp','line_count','anomaly_score','is_anomaly']].head(10))"),
        nbf.v4.new_markdown_cell(f"## Model assessment\n\nTraining: **{model['train_invoices']:,}** invoices through **{model['train_through']}**. Holdout: **{model['holdout_invoices']:,}** invoices from **{model['holdout_from']}**. Identical timestamps stay in one subset. Preprocessing is fitted on train only. On holdout, the Isolation Forest flags **{model['holdout_flagged']:,}** invoices, the training-derived amount rule flags **{model['holdout_rule_flagged']:,}**, and **{model['holdout_rule_model_overlap']:,}** overlap. At contamination thresholds 1%, 2%, and 5%, the same forest flags **{model['contamination_sensitivity']['0.01']['holdout_flagged']:,}**, **{model['contamination_sensitivity']['0.02']['holdout_flagged']:,}**, and **{model['contamination_sensitivity']['0.05']['holdout_flagged']:,}** holdout invoices. There are no verified anomaly labels, so precision, recall, F1, and fraud accuracy are unavailable."),
        nbf.v4.new_markdown_cell(f"## Duplicate sensitivity\n\nKeeping {duplicate['restored_duplicate_lines']:,} otherwise valid duplicate lines changes sale total from **£{duplicate['sale_total_gbp_deduplicated']:,.2f}** to **£{duplicate['sale_total_gbp_with_duplicate_lines']:,.2f}**. With the trained model held fixed, flagged invoices change from **{duplicate['model_flags_deduplicated']:,}** to **{duplicate['model_flags_with_duplicate_lines_fixed_model']:,}**; flags only in the alternate input: **{duplicate['flags_only_with_duplicate_lines']:,}**. This tests input sensitivity, not confirmed detection quality."),
        nbf.v4.new_markdown_cell("## Figures\n\nThe figures were generated by the executed pipeline and saved under `reports/figures/`. Clipping in two raw distributions is only for display; validation and modeling use original values."),
        nbf.v4.new_code_cell("for name in ['quantity.png','unit_price.png','countries.png','sale_amount_log.png','anomaly_scores.png','invoice_amount_comparison.png','anomalies_by_country.png','line_count_comparison.png']:\n    print(name)\n    display(Image(filename=str(ROOT / 'reports/figures' / name), width=650))"),
        nbf.v4.new_markdown_cell("## Storage, provenance, and limits\n\nDatabase loading is atomic. The local run verifies SQLite row counts; PostgreSQL requires a configured test server. The reproducibility manifest records raw SHA-256, Python and package versions, model settings, and the strict time boundary. Top-scoring invoice characteristics are not confirmed explanations or evidence of fraud. The public UK retail data is not representative of ARQA BUH operations or Kazakhstan accounting workflows."),
        nbf.v4.new_code_cell("summary = json.loads((ROOT / 'data/processed/run_summary.json').read_text())\nmanifest = json.loads((ROOT / 'data/processed/reproducibility_manifest.json').read_text())\nprint('Database:', summary['database'])\nprint('Source SHA-256:', manifest['source_sha256'])\nprint('Python:', manifest['python_version'])\nprint('Kernel executable:', sys.executable)")
    ]
    notebook.metadata.kernelspec = {"display_name": "Financial Project (.venv)", "language": "python", "name": kernel_name}
    notebook.metadata.language_info = {"name": "python"}
    client = NotebookClient(notebook, timeout=600, kernel_name=kernel_name, resources={"metadata": {"path": str(root)}})
    client.execute()
    nbf.write(notebook, root / "project.ipynb")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1])
