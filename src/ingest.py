"""
Boréal Marché — Data Ingestion Pipeline
========================================

Downloads the UCI Online Retail II dataset, verifies its integrity,
cleans it according to the rules in ``configs/pipeline.yaml``, converts
it to Parquet, and writes a provenance manifest.

Usage::

    python -m src.ingest            # run from the project root
    python -m src.ingest --force    # re-download even if file exists

The raw Excel file is large (~45 MB) and is **never** committed to Git.
It is tracked by DVC in ``data/raw/``.

**Honest provenance:** the underlying data is the UCI Online Retail II
public dataset, re-skinned as "Boréal Marché" for a pedagogical
capstone.  The DATA_CARD.md file states this plainly.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_config() -> dict:
    """Load and return the pipeline configuration dictionary."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Download & verify
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(cfg: dict, *, force: bool = False) -> Path:
    """Download the UCI zip, extract the xlsx, and return its path.

    Parameters
    ----------
    cfg : dict
        The ``data`` section of the pipeline config.
    force : bool
        If *True*, re-download even when the file already exists.

    Returns
    -------
    Path
        Absolute path to the extracted Excel file.
    """
    import urllib.request

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = raw_dir / cfg["raw_filename"]

    if xlsx_path.exists() and not force:
        print(f"[ingest] Found existing file: {xlsx_path}")
        return xlsx_path

    url = cfg["source_url"]
    print(f"[ingest] Downloading {url} …", flush=True)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
        zip_bytes = resp.read()
    print(f"[ingest] Downloaded {len(zip_bytes) / 1e6:.1f} MB")
    if url.lower().endswith(".xlsx"):
        xlsx_path.write_bytes(zip_bytes)
        return xlsx_path

    # Extract xlsx from the zip archive
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # The zip may contain the file in a subdirectory; find the xlsx
        xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx_names:
            raise FileNotFoundError("No .xlsx found inside the downloaded zip.")
        matches = [n for n in xlsx_names if Path(n).name == cfg["raw_filename"]]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {cfg['raw_filename']} in source archive")
        chosen = matches[0]
        print(f"[ingest] Extracting {chosen}")
        data = zf.read(chosen)
        xlsx_path.write_bytes(data)

    print(f"[ingest] Saved to {xlsx_path}")
    return xlsx_path


def verify_checksum(path: Path, expected_sha: str | None) -> str:
    """Verify the SHA-256 of the downloaded file.

    Parameters
    ----------
    path : Path
        File to verify.
    expected_sha : str or None
        Expected hex digest.  If *None*, verification is skipped but the
        computed digest is still returned so it can be recorded.

    Returns
    -------
    str
        The computed SHA-256 hex digest.
    """
    actual = _sha256(path)
    if not expected_sha:
        raise ValueError("A pinned source_sha256 is required before ingestion")
    if actual != expected_sha:
        raise ValueError(
            f"[ingest] SHA-256 mismatch; refusing to process input.\n"
            f"  expected: {expected_sha}\n"
            f"  actual:   {actual}\n"
            f"  Verify the official source before updating the pinned checksum."
        )
    else:
        status = "verified" if expected_sha else "recorded (no expected value)"
        print(f"[ingest] SHA-256 {status}: {actual[:16]}…")
    return actual


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
SERVICE_SKUS = {"POST", "DOT", "BANK CHARGES", "M", "PADS", "D", "C2", "CRUK", "S", "AMAZONFEE"}


def clean(df: pd.DataFrame, rules: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Apply cleaning rules and return (clean_df, report).

    Parameters
    ----------
    df : DataFrame
        Raw transaction data.
    rules : list[dict]
        Cleaning rule specifications from the config.

    Returns
    -------
    tuple[DataFrame, list[dict]]
        The cleaned DataFrame and a list of dicts documenting each rule
        application (name, description, rows_before, rows_after,
        rows_affected).
    """
    report: list[dict] = []
    df = df.rename(
        columns={"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID"}
    ).copy()
    required = {
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "Price",
        "Customer ID",
        "Country",
        "InvoiceDate",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Missing source columns: {sorted(required - set(df.columns))}")
    original_rows = len(df)
    for col in ["Quantity", "Price", "Customer ID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    for rule in rules:
        name = rule["name"]
        before = len(df)
        modified = 0

        if name == "remove_invalid_schema":
            valid = df[["Invoice", "StockCode", "InvoiceDate"]].notna().all(axis=1)
            valid &= np.isfinite(df[["Quantity", "Price"]]).all(axis=1)
            valid &= df["Quantity"].mod(1).eq(0)
            valid &= df["Customer ID"].isna() | (
                np.isfinite(df["Customer ID"]) & df["Customer ID"].mod(1).eq(0)
            )
            df = df.loc[valid].copy()

        elif name == "remove_zero_quantity":
            df = df.loc[df["Quantity"] != 0].copy()

        elif name == "remove_non_positive_price":
            df = df[df["Price"] > 0]

        elif name == "remove_null_customer":
            df = df[df["Customer ID"].notna()]

        elif name == "remove_service_skus":
            df = df[~df["StockCode"].astype(str).str.upper().isin(SERVICE_SKUS)]

        elif name == "canonicalise_descriptions":
            canonical = (
                df["Description"]
                .fillna("unknown")
                .astype(str)
                .str.lower()
                .str.strip()
                .apply(lambda s: re.sub(r"\s+", " ", s))
            )
            modified = int((canonical != df["Description"]).sum())
            df = df.copy()
            df["Description"] = canonical.replace("", "unknown")

        elif name == "cap_extreme_values":
            df = df.copy()
            qmax, pmax = rule["max_abs_quantity"], rule["max_unit_price"]
            df["WasCapped"] = (df["Quantity"].abs() > qmax) | (df["Price"] > pmax)
            modified = int(df["WasCapped"].sum())
            df["Quantity"] = df["Quantity"].clip(-qmax, qmax)
            df["Price"] = df["Price"].clip(upper=pmax)

        elif name == "flag_events":
            df = df.copy()
            df["IsCancellation"] = df["Invoice"].astype(str).str.upper().str.startswith("C")
            df["IsReturn"] = df["Quantity"] < 0
            df["IsPurchase"] = ~df["IsCancellation"] & (df["Quantity"] > 0)
            modified = len(df)

        else:
            raise ValueError(f"Unknown cleaning rule: {name}")

        after = len(df)
        entry = {
            "rule": name,
            "description": rule.get("description", ""),
            "rows_before": before,
            "rows_after": after,
            "rows_removed": before - after,
            "rows_modified": modified,
            "rows_affected": before - after + modified,
            "pct_of_original": 100 * (before - after + modified) / max(original_rows, 1),
            "pct_of_step_input": 100 * (before - after + modified) / max(before, 1),
        }
        report.append(entry)
        print(f"[ingest]   {name}: {before:,} → {after:,}  (−{before - after:,} rows)")

    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Parquet conversion & manifest
# ---------------------------------------------------------------------------
def to_parquet(df: pd.DataFrame, cfg: dict, actual_sha: str, cleaning_report: list[dict]) -> Path:
    """Write cleaned data to Parquet and produce the provenance manifest.

    Parameters
    ----------
    df : DataFrame
        Cleaned transaction data.
    cfg : dict
        The full pipeline config (top-level).
    actual_sha : str
        SHA-256 of the original raw file.
    cleaning_report : list[dict]
        Row-level cleaning report.

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    data_cfg = cfg["data"]
    parquet_path = PROJECT_ROOT / data_cfg["parquet_path"]
    manifest_path = PROJECT_ROOT / data_cfg["manifest_path"]
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Explicit schema: enforce types before writing
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype(str)
    df["Quantity"] = df["Quantity"].astype(int)
    df["Price"] = df["Price"].astype(float)
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["Country"] = df["Country"].astype(str)
    # InvoiceDate should already be datetime
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Add a line-total column used downstream
    df["LineTotal"] = df["Quantity"] * df["Price"]

    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"[ingest] Parquet written: {parquet_path}  ({len(df):,} rows)")

    # Manifest
    manifest = {
        "source": data_cfg["source_url"],
        "source_sha256": actual_sha,
        "processed_sha256": _sha256(parquet_path),
        "config_sha256": _sha256(CONFIG_PATH),
        "retrieval_date": datetime.now(UTC).isoformat(),
        "row_count": len(df),
        "original_row_count": cleaning_report[0]["rows_before"],
        "purchase_rows": int(df["IsPurchase"].sum()),
        "cancellation_rows": int(df["IsCancellation"].sum()),
        "return_rows": int(df["IsReturn"].sum()),
        "date_min": str(df["InvoiceDate"].min()),
        "date_max": str(df["InvoiceDate"].max()),
        "currency": "GBP (original source; no currency conversion)",
        "columns": list(df.columns),
        "licence": "CC BY 4.0 (UCI Machine Learning Repository)",
        "provenance": (
            "This is the UCI Online Retail II dataset, re-skinned as "
            f"'{data_cfg['reskin']['company_name']}' for a pedagogical "
            "capstone project at Vanier College.  It is NOT proprietary "
            "client data."
        ),
        "cleaning_report": cleaning_report,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    report_path = parquet_path.parent / "cleaning_report.json"
    report_path.write_text(
        json.dumps(
            {
                "original_row_count": manifest["original_row_count"],
                "final_row_count": len(df),
                "retained_pct": 100 * len(df) / manifest["original_row_count"],
                "rules": cleaning_report,
                "outlier_policy": "Fixed caps of absolute quantity 10000 and price GBP 1000; "
                "not statistical errors or learned quantiles. Raw workbook preserved. "
                "Wholesale sensitivity analysis remains future work.",
            },
            indent=2,
        )
    )
    print(f"[ingest] Manifest written: {manifest_path}")

    return parquet_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(*, force: bool = False) -> Path:
    """Execute the full ingestion pipeline.

    Returns
    -------
    Path
        Path to the final Parquet file.
    """
    cfg = load_config()
    data_cfg = cfg["data"]

    print("=" * 70)
    print("Boréal Marché — Data Ingestion")
    print("=" * 70)

    # 1. Download
    xlsx_path = download_dataset(data_cfg, force=force)

    # 2. Verify
    actual_sha = verify_checksum(xlsx_path, data_cfg.get("source_sha256"))

    # 3. Load both sheets (the dataset has Year 2009-2010 and Year 2010-2011)
    print("[ingest] Reading Excel sheets …")
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    print(f"[ingest] Loaded {len(df):,} rows from {len(sheets)} sheet(s)")

    # 4. Clean
    print("[ingest] Cleaning …")
    df, report = clean(df, cfg["cleaning"]["rules"])

    # 5. Write Parquet + manifest
    parquet_path = to_parquet(df, cfg, actual_sha, report)

    print("=" * 70)
    print("[ingest] Done.")
    return parquet_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI wrapper so the module can be run with ``python -m src.ingest``."""
    parser = argparse.ArgumentParser(description="Boréal Marché data ingestion")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the raw file exists",
    )
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
