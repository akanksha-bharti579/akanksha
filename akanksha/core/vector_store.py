"""
Persistent Hybrid Vector Store using ChromaDB with Reciprocal Rank Fusion (RRF).
"""

from pathlib import Path
from typing import Optional
import json
import re
from pydantic import BaseModel, Field
from akanksha.core.text_splitter import DocumentChunk

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class SearchResult(BaseModel):
    """Result of a vector and semantic search query."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    source_title: str
    source_path: str
    page_number: Optional[int] = None
    similarity_score: float
    metadata: dict = Field(default_factory=dict)


class VectorStore:
    """
    Hybrid Vector Store with ChromaDB backend and BM25-style keyword fallback.
    Provides persistent local storage, document management, and hybrid retrieval.
    """

    def __init__(self, persist_dir: Path | str, collection_name: str = "second_brain"):
        self.persist_dir = Path(persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # In-memory document chunk registry for metadata & keyword ranking
        self.chunks_cache: dict[str, DocumentChunk] = {}
        self.cache_file = self.persist_dir / "chunks_registry.json"
        self._load_cache()

        if HAS_CHROMADB:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir / "chroma"),
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.client = None
            self.collection = None

    def _load_cache(self) -> None:
        """Load document chunk cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        chunk = DocumentChunk(**item)
                        self.chunks_cache[chunk.id] = chunk
            except Exception:
                self.chunks_cache = {}

    def _save_cache(self) -> None:
        """Save document chunk cache to disk."""
        data = [chunk.model_dump() for chunk in self.chunks_cache.values()]
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Index a batch of DocumentChunks into the vector database."""
        if not chunks:
            return 0

        # Update cache
        for chunk in chunks:
            self.chunks_cache[chunk.id] = chunk
        self._save_cache()

        if HAS_CHROMADB and self.collection is not None:
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [
                {
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "source_title": chunk.source_title,
                    "source_path": chunk.source_path,
                    "page_number": chunk.page_number or 0,
                    "file_type": str(chunk.metadata.get("file_type", "unknown")),
                }
                for chunk in chunks
            ]
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        source_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Hybrid search combining ChromaDB dense vector retrieval and BM25 keyword matching.
        """
        query = query.strip()
        if not query or not self.chunks_cache:
            return []

        dense_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}

        # 1. Dense Vector Search via ChromaDB
        if HAS_CHROMADB and self.collection is not None:
            try:
                where_clause = None
                if source_filter:
                    where_clause = {"source_title": source_filter}

                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k * 2, len(self.chunks_cache)),
                    where=where_clause,
                )

                if results and results["ids"] and results["ids"][0]:
                    ids = results["ids"][0]
                    distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
                    for chunk_id, dist in zip(ids, distances):
                        # Cosine distance to similarity (1 - dist)
                        similarity = max(0.0, 1.0 - float(dist))
                        dense_scores[chunk_id] = similarity
            except Exception as e:
                # Fallback to pure keyword if vector query fails
                dense_scores = {}

        # 2. Keyword Match Scoring (BM25 token overlap approximation)
        query_tokens = set(re.findall(r"\w+", query.lower()))
        for chunk_id, chunk in self.chunks_cache.items():
            if source_filter and chunk.source_title != source_filter:
                continue

            chunk_tokens = re.findall(r"\w+", chunk.text.lower())
            if not chunk_tokens:
                continue

            matches = sum(1 for tok in query_tokens if tok in chunk_tokens)
            if matches > 0:
                # Token overlap ratio with logarithmic scale
                ratio = matches / len(query_tokens)
                keyword_scores[chunk_id] = min(1.0, ratio)

        # 3. Reciprocal Rank Fusion / Weighted Hybrid Score
        all_candidate_ids = set(dense_scores.keys()) | set(keyword_scores.keys())
        combined_results: list[SearchResult] = []

        for cid in all_candidate_ids:
            chunk = self.chunks_cache.get(cid)
            if not chunk:
                continue

            d_score = dense_scores.get(cid, 0.0)
            k_score = keyword_scores.get(cid, 0.0)

            # 70% dense vector + 30% exact keyword match
            if d_score > 0 and k_score > 0:
                final_score = (0.7 * d_score) + (0.3 * k_score)
            elif d_score > 0:
                final_score = d_score * 0.9
            else:
                final_score = k_score * 0.65

            combined_results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    source_title=chunk.source_title,
                    source_path=chunk.source_path,
                    page_number=chunk.page_number,
                    similarity_score=round(final_score, 4),
                    metadata=chunk.metadata,
                )
            )

        # Sort descending by similarity score
        combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return combined_results[:top_k]

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks associated with a doc_id from cache and ChromaDB."""
        chunks_to_remove = [
            cid for cid, chunk in self.chunks_cache.items() if chunk.doc_id == doc_id or chunk.source_title == doc_id
        ]
        if not chunks_to_remove:
            return 0

        for cid in chunks_to_remove:
            del self.chunks_cache[cid]
        self._save_cache()

        if HAS_CHROMADB and self.collection is not None:
            try:
                self.collection.delete(ids=chunks_to_remove)
            except Exception:
                pass

        return len(chunks_to_remove)

    def list_documents(self) -> list[dict]:
        """Return a summary of all unique ingested documents and their chunk counts."""
        docs_summary: dict[str, dict] = {}
        for chunk in self.chunks_cache.values():
            key = chunk.source_title
            if key not in docs_summary:
                docs_summary[key] = {
                    "doc_id": chunk.doc_id,
                    "title": chunk.source_title,
                    "source_path": chunk.source_path,
                    "file_type": chunk.metadata.get("file_type", "unknown"),
                    "total_chunks": 0,
                    "total_pages": chunk.metadata.get("total_pages", 1),
                    "character_count": 0,
                }
            docs_summary[key]["total_chunks"] += 1
            docs_summary[key]["character_count"] += len(chunk.text)

        return list(docs_summary.values())

    def clear(self) -> None:
        """Clear all indexed data."""
        self.chunks_cache.clear()
        self._save_cache()
        if HAS_CHROMADB and self.collection is not None:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    self.collection_name, metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass
