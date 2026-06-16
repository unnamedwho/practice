"""Capture source integrity, model settings, and the verified Python environment."""
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from .data_cleaning import CLEANING_VERSION

DIRECT_PACKAGES = ("pandas", "numpy", "pyarrow", "matplotlib", "scikit-learn", "joblib",
                   "SQLAlchemy", "psycopg", "pytest", "nbformat", "nbclient", "ipykernel", "openpyxl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path, raw_path: Path, model_path: Path, metrics: dict) -> dict:
    versions = {dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata.get("Name")}
    direct = {package: metadata.version(package) for package in DIRECT_PACKAGES}
    source = raw_path.resolve()
    manifest = {
        "source_file": str(source.relative_to(root)) if source.is_relative_to(root) else str(source),
        "source_sha256": sha256(source),
        "source_provenance": "UCI Online Retail ID 352; public Parquet mirror by pavan-naik" if source.suffix == ".parquet" else "UCI Online Retail ID 352; user-supplied local source",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "direct_dependency_versions": direct,
        "installed_distributions": dict(sorted(versions.items(), key=lambda item: item[0].lower())),
        "cleaning_version": CLEANING_VERSION,
        "model_file_sha256": sha256(model_path),
        "model_parameters": {key: metrics[key] for key in ("random_seed", "n_estimators", "contamination")},
        "features": metrics["features"],
        "train_through_inclusive": metrics["train_through"],
        "holdout_from_inclusive": metrics["holdout_from"],
        "split_rule": "all invoice completion timestamps <= train_through are train; later timestamps are holdout",
        "invoice_completion_rule": "latest line timestamp; input snapshot assumed complete",
    }
    (root / "data" / "processed" / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "requirements-tested.txt").write_text("# Verified direct dependencies for Python " + platform.python_version() + "\n" +
        "\n".join(f"{'psycopg[binary]' if package == 'psycopg' else package}=={direct[package]}"
                  for package in DIRECT_PACKAGES) + "\n")
    return manifest
