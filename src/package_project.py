"""Create a shareable archive without local environments, history, or caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

EXCLUDED_PARTS = {".venv", ".git", "__pycache__", "__MACOSX", ".pytest_cache",
                  ".ipynb_checkpoints", "dist"}
EXCLUDED_NAMES = {".env", ".DS_Store"}


def package(root: Path) -> Path:
    root = root.resolve()
    destination = root / "dist" / "financial_anomaly_project1_clean.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_symlink() or not path.is_file():
                continue
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.name in EXCLUDED_NAMES or path.suffix in {".pyc", ".pyo"}:
                continue
            archive.write(path, Path("financial-anomaly-project1") / relative)
    with ZipFile(destination) as archive:
        broken = archive.testzip()
        if broken:
            raise RuntimeError(f"Archive integrity failed at {broken}")
        names = archive.namelist()
        if any(any(part in EXCLUDED_PARTS for part in Path(name).parts) for name in names):
            raise RuntimeError("Archive contains an excluded directory")
    return destination


if __name__ == "__main__":
    print(package(Path(__file__).resolve().parents[1]))
