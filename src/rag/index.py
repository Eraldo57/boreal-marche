"""
Document indexing for the Boréal Marché RAG assistant.

Loads Markdown policy documents from docs/corpus/, splits each into sections (a "## "
heading and everything under it), cuts the sections into small chunks, embeds each chunk
with a multilingual sentence-transformer and stores the vectors in a persistent ChromaDB
collection. A chunk is matched by its small, titled text, but a query returns the whole
section it belongs to ("small to big"): the encoder reads only 128 tokens, so matching must
use small pieces, while a fact often sits in a table or list that a small piece would cut.

Usage (CLI):
    python -m src.rag.index                     # default paths
    python -m src.rag.index --corpus docs/corpus --db data/chromadb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "docs" / "corpus"
DEFAULT_DB = ROOT / "data" / "chromadb"
CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"
COLLECTION_NAME = "boreal_policy"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SECTION_LAYOUT = 2  # bumped when the index layout changes; an older index must be rebuilt
SECTIONS_FILE = "sections.json"  # section texts, stored beside the manifest
CHUNKS_PER_PASSAGE = 3  # chunks fetched per returned section, so sections can be deduplicated


def corpus_hash(corpus_dir: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(Path(corpus_dir).glob("*.md")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


@lru_cache(maxsize=2)
def embedding_model(name: str) -> Any:
    """Load once per process. Explicit repo cache works in local/Docker runs."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, cache_folder=str(ROOT / "data/model_cache"), device="cpu")


class RagIndex:
    """Long-lived query object, shared by API requests."""

    def __init__(
        self, db_dir: Path | str = DEFAULT_DB, corpus_dir: Path | str = DEFAULT_CORPUS
    ) -> None:
        import chromadb

        self.db_dir = Path(db_dir)
        metadata = json.loads((self.db_dir / "manifest.json").read_text())
        cfg = _load_assistant_config()
        if metadata["embedding_model"] != cfg["embedding_model"] or metadata[
            "corpus_sha256"
        ] != corpus_hash(Path(corpus_dir)):
            raise ValueError("Stale RAG index; run python -m src.rag.index --force")
        if (
            metadata["chunk_size"] != cfg["chunk_size"]
            or metadata["chunk_overlap"] != cfg["chunk_overlap"]
        ):
            raise ValueError("Chunk configuration changed; rebuild RAG index")
        if metadata.get("layout") != SECTION_LAYOUT:
            raise ValueError("Index layout changed; rebuild RAG index (--force)")
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = self.client.get_collection(COLLECTION_NAME)
        if self.collection.count() != metadata["chunk_count"] or not self.collection.count():
            raise ValueError("RAG collection/manifest count mismatch")
        self.model = embedding_model(metadata["embedding_model"])
        self.sections = json.loads((self.db_dir / SECTIONS_FILE).read_text(encoding="utf-8"))
        self.metadata = metadata

    def query(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """The *top_k* best sections for *question*, best first.

        ``score`` is the best chunk's similarity; ``chunk_index`` is the section's position
        in its document.
        """
        return search(self.collection, self.model, self.sections, question, top_k)


def search(
    collection: Any, model: Any, sections: dict[str, Any], question: str, top_k: int
) -> list[dict[str, Any]]:
    """Match small chunks, return the distinct sections they belong to (best chunk first)."""
    vectors = model.encode([question], normalize_embeddings=True).tolist()
    result: Any = collection.query(
        query_embeddings=vectors,
        n_results=min(top_k * CHUNKS_PER_PASSAGE, collection.count()),
        include=["metadatas", "distances"],
    )
    hits: dict[tuple[str, int], dict[str, Any]] = {}
    for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
        key = (meta["source"], meta["section"])
        if key not in hits:  # results come best first, so the first chunk of a section is its best
            hits[key] = {
                "text": sections[meta["source"]][meta["section"]]["text"],
                "source": meta["source"],
                "score": float(1 - distance),
                "language": meta["language"],
                "chunk_index": meta["section"],
            }
    return list(hits.values())[:top_k]


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _load_assistant_config() -> dict[str, Any]:
    """Read the ``assistant`` section from pipeline.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    assistant: dict[str, Any] = cfg.get("assistant", {})
    return assistant


def _detect_language(text: str) -> str:
    """Heuristic language detection — French vs English.

    Counts common French stop-words.  Good enough for chunked policy docs
    where we know only two languages are present.
    """
    french_markers = {
        "le",
        "la",
        "les",
        "de",
        "du",
        "des",
        "un",
        "une",
        "et",
        "est",
        "en",
        "dans",
        "pour",
        "sur",
        "avec",
        "par",
        "au",
        "aux",
        "ce",
        "cette",
        "ces",
        "nous",
        "vous",
        "sont",
        "qui",
        "que",
        "ne",
        "pas",
        "ou",
        "ses",
        "sa",
        "son",
        "leur",
        "leurs",
        "ont",
        "peut",
        "être",
        "votre",
        "vos",
        "notre",
        "nos",
    }
    words = set(re.findall(r"\b[a-zàâéèêëïîôùûüç]+\b", text.lower()))
    score = len(words & french_markers)
    return "fr" if score >= 5 else "en"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

_DATE_LINE = re.compile(r"^\*(Last updated|Dernière mise à jour)[^*]*\*$")


def split_sections(text: str) -> list[dict[str, str]]:
    """Split a Markdown policy into sections: each "## " heading and everything under it.

    Text before the first heading, apart from the title and the date line, becomes an
    "Introduction" section; a document without "## " headings is that one section. Every
    section carries its document's title, and its ``text`` starts with its heading line.
    """
    title_match = re.match(r"#\s+(.+)", text)
    title = title_match.group(1).strip() if title_match else ""
    first, *rest = re.split(r"(?m)^(?=## )", text)
    intro = "\n".join(
        line
        for line in first.splitlines()
        if not line.startswith("# ") and not _DATE_LINE.match(line.strip())
    ).strip()
    sections = []
    if intro:
        sections.append(
            {"title": title, "heading": "Introduction", "text": f"## Introduction\n{intro}"}
        )
    for part in rest:
        part = part.strip()
        heading = part.splitlines()[0].lstrip("# ").strip()
        sections.append({"title": title, "heading": heading, "text": part})
    return sections


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
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size")
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
) -> RagIndex:
    """Index every ``.md`` file in *corpus_dir* into ChromaDB.

    Returns the same query object used by the API lifespan.
    """
    import chromadb

    corpus_dir, db_dir = Path(corpus_dir), Path(db_dir)
    cfg = _load_assistant_config()
    chunk_size: int = cfg.get("chunk_size", 512)
    chunk_overlap: int = cfg.get("chunk_overlap", 64)
    model_name: str = cfg.get("embedding_model", DEFAULT_MODEL)

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
        return RagIndex(db_dir, corpus_dir)

    # Load the embedding model.
    logger.info("Loading embedding model '%s'…", model_name)
    model = embedding_model(model_name)

    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metas: list[Any] = []
    all_sections: dict[str, list[dict[str, str]]] = {}

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        source = md_path.name
        language = _detect_language(text)
        sections = split_sections(text)
        all_sections[source] = sections
        count = 0
        for position, section in enumerate(sections):
            body = section["text"].split("\n", 1)[1] if "\n" in section["text"] else ""
            # The title and heading go in front of every chunk, so a table row or a list item
            # still says which policy and section it belongs to.
            head = f"{section['title']} — {section['heading']}\n"
            for piece in chunk_text(body, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                doc_id = hashlib.sha256(f"{source}:{count}".encode()).hexdigest()[:16]
                all_ids.append(doc_id)
                all_texts.append(head + piece)
                all_metas.append(
                    {
                        "source": source,
                        "chunk_index": count,
                        "section": position,
                        "language": language,
                    }
                )
                count += 1
        logger.info("  %s → %d sections, %d chunks (%s)", source, len(sections), count, language)

    # Embed all chunks in one batch.
    logger.info("Embedding %d chunks…", len(all_texts))
    embeddings = model.encode(all_texts, show_progress_bar=True, normalize_embeddings=True)

    # Upsert into ChromaDB (handles both insert and update).
    batch_size = 500
    for start in range(0, len(all_ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=all_ids[start:end],
            documents=all_texts[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=all_metas[start:end],
        )

    total = collection.count()
    (db_dir / SECTIONS_FILE).write_text(
        json.dumps(all_sections, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (db_dir / "manifest.json").write_text(
        json.dumps(
            {
                "embedding_model": model_name,
                "corpus_sha256": corpus_hash(corpus_dir),
                "chunk_count": total,
                "document_count": len(md_files),
                "section_count": sum(len(v) for v in all_sections.values()),
                "layout": SECTION_LAYOUT,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            indent=2,
        )
    )
    logger.info("Index complete — %d chunks in collection '%s'.", total, COLLECTION_NAME)
    return RagIndex(db_dir, corpus_dir)


def query_index(
    question: str,
    *,
    db_dir: Path = DEFAULT_DB,
    n_results: int | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top-*n* sections most similar to *question*.

    Returns a list of dicts with keys ``text``, ``source``, ``score``,
    ``language``, and ``chunk_index`` (the section's position in its document).
    """
    cfg = _load_assistant_config()
    n_results = n_results or cfg.get("max_chunks_per_query", 5)
    model_name = model_name or cfg.get("embedding_model", DEFAULT_MODEL)

    import chromadb

    collection = chromadb.PersistentClient(path=str(db_dir)).get_collection(COLLECTION_NAME)
    sections = json.loads((Path(db_dir) / SECTIONS_FILE).read_text(encoding="utf-8"))
    hits = search(collection, embedding_model(model_name), sections, question, n_results)
    return [{**hit, "score": round(hit["score"], 4)} for hit in hits]


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
