"""
SecondBrain: Unified RAG engine for ingestion, hybrid semantic search, and grounded Q&A.
"""

from pathlib import Path
from typing import Optional
import time
from pydantic import BaseModel, Field

from akanksha.core.config import BrainConfig, default_config
from akanksha.core.document_loader import DocumentLoader, Document
from akanksha.core.text_splitter import RecursiveTextSplitter, DocumentChunk
from akanksha.core.vector_store import VectorStore, SearchResult
from akanksha.core.llm_provider import get_llm_backend, BaseLLM


class SourceCitation(BaseModel):
    """Citation metadata for an answer."""
    title: str
    source_path: str
    page_number: Optional[int] = None
    similarity_score: float
    snippet: str


class QueryResponse(BaseModel):
    """Complete response returned by SecondBrain.query()."""
    query: str
    answer: str
    sources: list[SourceCitation]
    latency_ms: float
    total_chunks_searched: int


class SecondBrain:
    """
    Main orchestration class for Akanksha's AI Second Brain.
    Handles document ingestion, vector indexing, hybrid retrieval, and grounded synthesis.
    """

    def __init__(self, config: Optional[BrainConfig] = None):
        self.config = config or default_config
        self.config.ensure_directories()

        self.splitter = RecursiveTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.vector_store = VectorStore(
            persist_dir=self.config.index_dir,
            collection_name="second_brain",
        )
        self.llm = self._init_llm()

    def _init_llm(self) -> BaseLLM:
        """Initialize LLM based on current config."""
        provider = self.config.llm_provider.lower()
        if provider == "gemini":
            return get_llm_backend(
                provider_name="gemini",
                api_key=self.config.gemini_api_key,
                model=self.config.gemini_model,
            )
        elif provider == "openai":
            return get_llm_backend(
                provider_name="openai",
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
                model=self.config.openai_model,
            )
        elif provider == "ollama":
            return get_llm_backend(
                provider_name="ollama",
                base_url=self.config.ollama_base_url,
                model=self.config.ollama_model,
            )
        else:
            return get_llm_backend(provider_name="offline")

    def set_provider(self, provider: str, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """Switch the LLM provider dynamically at runtime."""
        self.config.llm_provider = provider
        if api_key:
            if provider == "gemini":
                self.config.gemini_api_key = api_key
            elif provider == "openai":
                self.config.openai_api_key = api_key
        if model:
            if provider == "gemini":
                self.config.gemini_model = model
            elif provider == "openai":
                self.config.openai_model = model
            elif provider == "ollama":
                self.config.ollama_model = model
        self.llm = self._init_llm()

    def ingest_file(self, file_path: str | Path) -> dict:
        """Load, chunk, and index a single document file."""
        path = Path(file_path).resolve()
        start_time = time.perf_counter()

        docs = DocumentLoader.load_file(path)
        if not docs:
            return {"status": "empty", "chunks_indexed": 0, "file_name": path.name}

        chunks = self.splitter.split_documents(docs)
        indexed_count = self.vector_store.add_chunks(chunks)

        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "status": "success",
            "file_name": path.name,
            "pages": len(docs) if docs[0].file_type == "pdf" else 1,
            "chunks_indexed": indexed_count,
            "duration_ms": elapsed,
        }

    def ingest_directory(self, dir_path: str | Path, recursive: bool = True) -> dict:
        """Ingest all supported documents inside a folder."""
        path = Path(dir_path).resolve()
        start_time = time.perf_counter()

        docs = DocumentLoader.load_directory(
            path,
            supported_extensions=self.config.supported_extensions,
            recursive=recursive,
        )
        if not docs:
            return {"status": "empty", "files_found": 0, "chunks_indexed": 0}

        chunks = self.splitter.split_documents(docs)
        indexed_count = self.vector_store.add_chunks(chunks)

        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        unique_files = len({d.source_path for d in docs})

        return {
            "status": "success",
            "files_indexed": unique_files,
            "chunks_indexed": indexed_count,
            "duration_ms": elapsed,
        }

    def search(self, query: str, top_k: Optional[int] = None) -> list[SearchResult]:
        """Perform semantic & keyword search across the knowledge base."""
        k = top_k or self.config.top_k
        return self.vector_store.search(query=query, top_k=k)

    def query(self, question: str, top_k: Optional[int] = None) -> QueryResponse:
        """
        Execute full RAG pipeline:
        1. Hybrid retrieve top-k chunks
        2. Format grounded prompt
        3. Synthesize response with LLM
        4. Return answer + source citations
        """
        start_time = time.perf_counter()
        k = top_k or self.config.top_k

        results = self.search(question, top_k=k)
        if not results:
            return QueryResponse(
                query=question,
                answer="No relevant documents or notes found in your Second Brain. Ingest some files first to start querying!",
                sources=[],
                latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
                total_chunks_searched=len(self.vector_store.chunks_cache),
            )

        # Build context blocks
        context_parts: list[str] = []
        citations: list[SourceCitation] = []

        for r in results:
            page_str = f", Page {r.page_number}" if r.page_number else ""
            source_header = f"[{r.source_title}{page_str}] (Score: {r.similarity_score})"
            context_parts.append(f"{source_header}\n{r.text}")

            citations.append(
                SourceCitation(
                    title=r.source_title,
                    source_path=r.source_path,
                    page_number=r.page_number,
                    similarity_score=r.similarity_score,
                    snippet=r.text[:240].strip() + ("..." if len(r.text) > 240 else ""),
                )
            )

        context_str = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are Akanksha's AI Second Brain and Research Assistant. "
            "Your objective is to provide precise, insightful, and well-structured answers "
            "based strictly on the user's provided research papers, documents, and notes. "
            "Always cite the source document and page number when referencing facts. "
            "If the context does not contain the answer, truthfully state that."
        )

        user_prompt = (
            f"=== CONTEXT ===\n"
            f"{context_str}\n\n"
            f"=== USER QUESTION ===\n"
            f"{question}\n\n"
            f"Provide a clear, grounded answer with citations."
        )

        answer = self.llm.generate(prompt=user_prompt, system_prompt=system_prompt)
        latency = round((time.perf_counter() - start_time) * 1000, 2)

        return QueryResponse(
            query=question,
            answer=answer,
            sources=citations,
            latency_ms=latency,
            total_chunks_searched=len(self.vector_store.chunks_cache),
        )

    def summarize(self, document_title: str) -> str:
        """Generate an executive summary of a specific ingested document."""
        chunks = [
            c for c in self.vector_store.chunks_cache.values()
            if c.source_title.lower() == document_title.lower() or c.doc_id == document_title
        ]
        if not chunks:
            return f"Document '{document_title}' not found in the index."

        chunks.sort(key=lambda c: (c.page_number or 0, c.chunk_index))
        # Select representative chunks across the document
        sample_chunks = chunks[:8]
        combined_text = "\n\n".join([c.text for c in sample_chunks])

        prompt = (
            f"=== CONTEXT ===\n{combined_text}\n\n"
            f"=== USER QUESTION ===\n"
            f"Provide an executive summary of '{document_title}', including:\n"
            f"1. Core Topic & Purpose\n"
            f"2. Key Methodologies or Main Points\n"
            f"3. Practical Takeaways & Conclusions"
        )
        return self.llm.generate(prompt=prompt)

    def list_documents(self) -> list[dict]:
        """List all indexed documents and metrics."""
        return self.vector_store.list_documents()

    def delete_document(self, doc_id: str) -> int:
        """Remove a document and its chunks from the Second Brain."""
        return self.vector_store.delete_document(doc_id)

    def clear(self) -> None:
        """Wipe the second brain index completely."""
        self.vector_store.clear()
