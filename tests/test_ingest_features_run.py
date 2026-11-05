"""
Ingestion and feature-engineering run on a SYNTHETIC workbook.

Every row below is generated, not taken from online_retail_II.xlsx. Items, customers
and prices are placeholders. The workbook has the real file's column layout and two
sheets so the real ingest/feature code paths run end to end on a clean CI checkout,
which does not have the DVC-tracked raw data.
"""

from __future__ import annotations

import copy
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import features, ingest
from src.contracts import ENGINEERED_FEATURE_NAMES, FEATURE_NAMES

COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def _synthetic_workbook(path):
    rng = np.random.default_rng(11)
    rows, invoice = [], 100000
    start, end = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09")
    span_days = (end - start).days
    for customer in range(12000, 12150):
        country = "United Kingdom" if customer % 5 else "France"
        for _ in range(int(rng.integers(2, 12))):
            invoice += 1
            stamp = start + pd.Timedelta(days=int(rng.integers(0, span_days)), hours=10)
            for _ in range(int(rng.integers(1, 4))):
                sku = f"SKU{int(rng.integers(1, 15))}"
                rows.append(
                    [str(invoice), sku, f"Item {sku}", int(rng.integers(1, 6)), stamp,
                     float(rng.integers(1, 20)), customer, country]
                )  # fmt: skip
    # Rows each cleaning rule must remove or reclassify.
    stamp = pd.Timestamp("2010-03-01 10:00")
    rows += [
        ["900001", "SKU1", "Item SKU1", 0, stamp, 2.0, 12000, "United Kingdom"],
        ["900002", "SKU1", "Item SKU1", 1, stamp, 0.0, 12000, "United Kingdom"],
        ["900003", "SKU1", "Item SKU1", 1, stamp, 2.0, None, "United Kingdom"],
        ["900004", "POST", "Postage", 1, stamp, 2.0, 12000, "United Kingdom"],
        ["C900005", "SKU1", "Item SKU1", -1, stamp, 2.0, 12000, "United Kingdom"],
    ]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    cut = pd.Timestamp("2010-12-01")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame[frame["InvoiceDate"] < cut].to_excel(writer, sheet_name="Year 2009-2010", index=False)
        frame[frame["InvoiceDate"] >= cut].to_excel(
            writer, sheet_name="Year 2010-2011", index=False
        )
    return len(frame)


@pytest.fixture
def workbook(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    path = raw / "online_retail_II.xlsx"
    total = _synthetic_workbook(path)
    cfg = copy.deepcopy(ingest.load_config())
    cfg["data"]["source_sha256"] = ingest._sha256(path)
    return tmp_path, path, cfg, total


@pytest.fixture
def ingested(workbook, monkeypatch):
    root, path, cfg, total = workbook
    monkeypatch.setattr(ingest, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingest, "load_config", lambda: cfg)
    return root, path, cfg, total, ingest.run()


def test_run_ingests_cleans_and_writes_manifest(ingested):
    root, path, cfg, total, parquet = ingested
    manifest = json.loads((root / cfg["data"]["manifest_path"]).read_text())
    assert manifest["source_sha256"] == cfg["data"]["source_sha256"]
    assert manifest["original_row_count"] == total
    assert manifest["row_count"] < total, "invalid rows must be removed"
    assert manifest["cancellation_rows"] >= 1
    frame = pd.read_parquet(parquet)
    assert (frame["Price"] > 0).all() and frame["Customer ID"].notna().all()
    assert "POST" not in set(frame["StockCode"])
    report = json.loads((parquet.parent / "cleaning_report.json").read_text())
    assert report["rules"] and report["final_row_count"] == len(frame)


def test_features_run_builds_contract_on_ingested_data(ingested, monkeypatch):
    root, *_ = ingested
    monkeypatch.setattr(features, "PROJECT_ROOT", root)
    matrix, y = features.run()
    assert list(matrix.columns) == ENGINEERED_FEATURE_NAMES
    assert matrix.index.equals(y.index) and set(y.unique()) <= {0, 1}
    processed = root / "data" / "processed"
    metadata = json.loads((processed / "feature_metadata.json").read_text())
    assert metadata["n_features"] == 25 and metadata["n_customers"] == len(matrix)
    store = pd.read_parquet(processed / "customer_features.parquet")
    assert list(store.columns) == FEATURE_NAMES


def test_features_reject_missing_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "PROJECT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="src.ingest"):
        features.run()


def test_features_reject_incomplete_label_window(ingested, monkeypatch):
    root, *_ = ingested
    frame = pd.read_parquet(root / "data/processed/transactions.parquet")
    frame[frame["InvoiceDate"] < "2011-06-01"].to_parquet(
        root / "data/processed/transactions.parquet"
    )
    monkeypatch.setattr(features, "PROJECT_ROOT", root)
    with pytest.raises(ValueError, match="Incomplete label window"):
        features.run()


def test_checksum_requires_pinned_value(workbook):
    _, path, _, _ = workbook
    with pytest.raises(ValueError, match="pinned"):
        ingest.verify_checksum(path, None)


def test_checksum_accepts_matching_value(workbook):
    _, path, cfg, _ = workbook
    assert (
        ingest.verify_checksum(path, cfg["data"]["source_sha256"]) == cfg["data"]["source_sha256"]
    )


def test_download_skips_existing_file(workbook):
    root, path, cfg, _ = workbook
    data_cfg = copy.deepcopy(cfg["data"])
    ingest.PROJECT_ROOT, original = root, ingest.PROJECT_ROOT
    try:
        assert ingest.download_dataset(data_cfg) == path
    finally:
        ingest.PROJECT_ROOT = original


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _download(tmp_path, monkeypatch, url, payload, filename="online_retail_II.xlsx"):
    monkeypatch.setattr(ingest, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _FakeResponse(payload))
    return ingest.download_dataset({"source_url": url, "raw_filename": filename})


def test_download_saves_direct_xlsx(tmp_path, monkeypatch):
    saved = _download(tmp_path, monkeypatch, "https://example.test/file.xlsx", b"workbook-bytes")
    assert saved.read_bytes() == b"workbook-bytes"


def test_download_extracts_xlsx_from_zip(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nested/online_retail_II.xlsx", b"inside-zip")
    saved = _download(tmp_path, monkeypatch, "https://example.test/file.zip", buffer.getvalue())
    assert saved.read_bytes() == b"inside-zip"


def test_download_rejects_zip_without_workbook(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", b"nothing useful")
    with pytest.raises(FileNotFoundError, match="No .xlsx"):
        _download(tmp_path, monkeypatch, "https://example.test/file.zip", buffer.getvalue())


def test_download_rejects_ambiguous_archive(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("other.xlsx", b"x")
    with pytest.raises(ValueError, match="exactly one"):
        _download(tmp_path, monkeypatch, "https://example.test/file.zip", buffer.getvalue())


def test_clean_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing source columns"):
        ingest.clean(pd.DataFrame({"Invoice": ["1"]}), [])
