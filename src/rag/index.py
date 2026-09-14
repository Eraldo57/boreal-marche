"""
Document indexing for the Boréal Marché RAG assistant.

Loads Markdown policy documents from docs/corpus/, chunks them with configurable
size and overlap, embeds each chunk with a multilingual sentence-transformer,
and stores the vectors in a persistent ChromaDB collection.

Usage (CLI):
    python -m src.rag.index                     # default paths
    python -m src.rag.index --corpus docs/corpus --db data/chromadb
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "docs" / "corpus"
DEFAULT_DB = ROOT / "data" / "chromadb"
CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"
COLLECTION_NAME = "boreal_policy"


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_assistant_config() -> dict[str, Any]:
    """Read the ``assistant`` section from pipeline.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("assistant", {})


def _detect_language(text: str) -> str:
    """Heuristic language detection — French vs English.

    Counts common French stop-words.  Good enough for chunked policy docs
    where we know only two languages are present.
    """
    french_markers = {
        "le", "la", "les", "de", "du", "des", "un", "une", "et", "est",
        "en", "dans", "pour", "sur", "avec", "par", "au", "aux", "ce",
        "cette", "ces", "nous", "vous", "sont", "qui", "que", "ne", "pas",
        "ou", "ses", "sa", "son", "leur", "leurs", "ont", "peut", "être",
        "votre", "vos", "notre", "nos",
    }
    words = set(re.findall(r"\b[a-zàâéèêëïîôùûüç]+\b", text.lower()))
    score = len(words & french_markers)
    return "fr" if score >= 5 else "en"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Split *text* into overlapping windows of roughly *chunk_size* characters.

    Splitting happens at paragraph boundaries (``\\n\\n``) first, then at
    sentence boundaries (``. ``), and finally at word boundaries.  This keeps
    tables and headings intact when possible.
    """
    # Split on double-newlines first (paragraphs / table rows).
    paragraphs = re.split(r"\n{2,}", text)

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para.strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # The paragraph itself may exceed chunk_size — split it further.
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    candidate = (current + " " + sent).strip() if current else sent
                    if len(candidate) <= chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        # Single huge sentence — hard-split at word boundary.
                        while len(sent) > chunk_size:
                            cut = sent[:chunk_size].rfind(" ")
                            if cut == -1:
                                cut = chunk_size
                            chunks.append(sent[:cut].strip())
                            sent = sent[cut:].strip()
                        current = sent
            else:
                current = para.strip()

    if current:
        chunks.append(current)

    # Apply overlap by carrying the tail of the previous chunk forward.
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append(tail.strip() + " " + chunks[i])
        chunks = overlapped

    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_index(
    corpus_dir: Path = DEFAULT_CORPUS,
    db_dir: Path = DEFAULT_DB,
    *,
    force: bool = False,
) -> int:
    """Index every ``.md`` file in *corpus_dir* into ChromaDB.

    Returns the total number of chunks indexed.
    """
    cfg = _load_assistant_config()
    chunk_size: int = cfg.get("chunk_size", 512)
    chunk_overlap: int = cfg.get("chunk_overlap", 64)
    model_name: str = cfg.get("embedding_model", "all-MiniLM-L6-v2")

    md_files = sorted(corpus_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {corpus_dir}")
    logger.info("Found %d documents in %s", len(md_files), corpus_dir)

    # Prepare ChromaDB.
    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))

    if force:
        # Drop the existing collection so we start fresh.
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # If the collection already has data and we're not forcing, skip.
    if collection.count() > 0 and not force:
        logger.info(
            "Collection '%s' already has %d chunks — skipping (use --force to rebuild).",
            COLLECTION_NAME,
            collection.count(),
        )
        return collection.count()

    # Load the embedding model.
    logger.info("Loading embedding model '%s'…", model_name)
    model = SentenceTransformer(model_name)

    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metas: list[dict[str, str | int]] = []

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        source = md_path.name
        language = _detect_language(text)
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info("  %s → %d chunks (%s)", source, len(chunks), language)

        for idx, chunk in enumerate(chunks):
            doc_id = hashlib.sha256(f"{source}:{idx}".encode()).hexdigest()[:16]
            all_ids.append(doc_id)
            all_texts.append(chunk)
            all_metas.append({
                "source": source,
                "chunk_index": idx,
                "language": language,
            })

    # Embed all chunks in one batch.
    logger.info("Embedding %d chunks…", len(all_texts))
    embeddings = model.encode(all_texts, show_progress_bar=True, normalize_embeddings=True)

    # Upsert into ChromaDB (handles both insert and update).
    BATCH = 500
    for start in range(0, len(all_ids), BATCH):
        end = start + BATCH
        collection.upsert(
            ids=all_ids[start:end],
            documents=all_texts[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=all_metas[start:end],
        )

    total = collection.count()
    logger.info("Index complete — %d chunks in collection '%s'.", total, COLLECTION_NAME)
    return total


def query_index(
    question: str,
    *,
    db_dir: Path = DEFAULT_DB,
    n_results: int | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top-*n* chunks most similar to *question*.

    Returns a list of dicts with keys ``text``, ``source``, ``score``,
    ``language``, and ``chunk_index``.
    """
    cfg = _load_assistant_config()
    n_results = n_results or cfg.get("max_chunks_per_query", 5)
    model_name = model_name or cfg.get("embedding_model", "all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(COLLECTION_NAME)

    model = SentenceTransformer(model_name)
    q_emb = model.encode([question], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, Any]] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns cosine *distance* (0 = identical); convert to
        # similarity for the rest of the pipeline.
        score = 1.0 - dist
        hits.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "language": meta["language"],
            "score": round(score, 4),
        })

    return hits


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI wrapper for ``python -m src.rag.index``."""
    parser = argparse.ArgumentParser(description="Build the Boréal Marché document index.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to the corpus directory (default: docs/corpus).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to the ChromaDB directory (default: data/chromadb).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and rebuild the collection even if it already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    build_index(args.corpus, args.db, force=args.force)


if __name__ == "__main__":
    main()
