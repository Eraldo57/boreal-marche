"""
RAG index build and query tests with a FAKE encoder and a placeholder corpus.

The embedding model is replaced by a deterministic hash-based encoder so the test needs
no model download (CI has no model cache). The corpus text is a short invented
placeholder, not the project's policy documents. This verifies the indexing and
retrieval mechanics, not retrieval quality, which the golden-set evaluation covers.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pytest

from src.rag import index

ENGLISH = (
    "# Placeholder returns\n\n"
    "Customers may return unopened items within thirty days of delivery. "
    "A receipt is required for every return. Refunds go to the original payment method.\n\n"
    "Perishable goods cannot be returned for food-safety reasons."
)
FRENCH = (
    "# Politique fictive\n\n"
    "Nous livrons dans les régions pour les clients qui ont une commande. "
    "Vous pouvez suivre votre colis dans votre compte et nous vous écrivons par courriel. "
    "Les frais sont affichés avant que vous ne payiez cette commande."
)


class FakeEncoder:
    """Deterministic unit vectors: identical text gives identical vectors."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
            vector = np.random.default_rng(seed).normal(size=16)
            vectors.append(vector / np.linalg.norm(vector))
        return np.array(vectors)


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "returns_en.md").write_text(ENGLISH, encoding="utf-8")
    (folder / "livraison_fr.md").write_text(FRENCH, encoding="utf-8")
    return folder


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    monkeypatch.setattr(index, "embedding_model", lambda name: FakeEncoder())


def test_build_index_writes_manifest_and_serves_queries(corpus, tmp_path):
    db = tmp_path / "db"
    built = index.build_index(corpus, db)
    manifest = json.loads((db / "manifest.json").read_text())
    assert manifest["document_count"] == 2
    assert manifest["chunk_count"] == built.collection.count() >= 2
    assert manifest["corpus_sha256"] == index.corpus_hash(corpus)

    exact_chunk = built.collection.get(include=["documents"])["documents"][0]
    hits = built.query(exact_chunk, top_k=3)
    assert hits[0]["text"] == exact_chunk and hits[0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert {h["language"] for h in hits} <= {"en", "fr"}


def test_languages_are_tagged_per_document(corpus, tmp_path):
    built = index.build_index(corpus, tmp_path / "db")
    metas = built.collection.get(include=["metadatas"])["metadatas"]
    by_source = {m["source"]: m["language"] for m in metas}
    assert by_source == {"returns_en.md": "en", "livraison_fr.md": "fr"}


def test_second_build_is_skipped_and_force_rebuilds(corpus, tmp_path):
    db = tmp_path / "db"
    first = index.build_index(corpus, db)
    count = first.collection.count()
    assert index.build_index(corpus, db).collection.count() == count
    assert index.build_index(corpus, db, force=True).collection.count() == count


def test_changed_corpus_makes_index_stale(corpus, tmp_path):
    db = tmp_path / "db"
    index.build_index(corpus, db)
    (corpus / "returns_en.md").write_text(ENGLISH + "\n\nExtra clause.", encoding="utf-8")
    with pytest.raises(ValueError, match="Stale RAG index"):
        index.RagIndex(db, corpus)


def test_changed_chunk_config_is_rejected(corpus, tmp_path):
    db = tmp_path / "db"
    index.build_index(corpus, db)
    manifest = json.loads((db / "manifest.json").read_text())
    manifest["chunk_size"] += 1
    (db / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Chunk configuration"):
        index.RagIndex(db, corpus)


def test_manifest_count_mismatch_is_rejected(corpus, tmp_path):
    db = tmp_path / "db"
    index.build_index(corpus, db)
    manifest = json.loads((db / "manifest.json").read_text())
    manifest["chunk_count"] += 5
    (db / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="count mismatch"):
        index.RagIndex(db, corpus)


def test_empty_corpus_is_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        index.build_index(empty, tmp_path / "db")


def test_query_index_function_returns_scored_hits(corpus, tmp_path):
    db = tmp_path / "db"
    built = index.build_index(corpus, db)
    chunk = built.collection.get(include=["documents"])["documents"][0]
    hits = index.query_index(chunk, db_dir=db, n_results=2)
    assert len(hits) == 2 and hits[0]["score"] >= hits[1]["score"]
    assert set(hits[0]) == {"text", "source", "chunk_index", "language", "score"}


def test_command_line_entry_point_builds_index(corpus, tmp_path, monkeypatch):
    db = tmp_path / "cli_db"
    monkeypatch.setattr(sys, "argv", ["index", "--corpus", str(corpus), "--db", str(db), "--force"])
    index.main()
    assert (db / "manifest.json").exists()


def test_detect_language_distinguishes_french_from_english():
    assert index._detect_language(FRENCH) == "fr"
    assert index._detect_language(ENGLISH) == "en"


def test_chunk_text_splits_long_sentences_and_adds_overlap():
    long_text = "word " * 400
    chunks = index.chunk_text(long_text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 3
    assert all(len(chunk) <= 100 + 10 + 1 for chunk in chunks)


def test_chunk_text_hard_splits_a_single_unbroken_token():
    chunks = index.chunk_text("x" * 250, chunk_size=100, chunk_overlap=0)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_chunk_text_splits_paragraphs_and_sentences():
    text = "First sentence here. Second sentence here.\n\nAnother paragraph that is fairly long."
    chunks = index.chunk_text(text, chunk_size=30, chunk_overlap=0)
    assert len(chunks) >= 3


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_chunk_text_rejects_invalid_window(size, overlap):
    with pytest.raises(ValueError):
        index.chunk_text("anything", chunk_size=size, chunk_overlap=overlap)
