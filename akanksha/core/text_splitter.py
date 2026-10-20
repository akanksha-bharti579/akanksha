"""
Recursive character text splitting for RAG chunking with overlap.
"""

from typing import Optional
from pydantic import BaseModel, Field
from akanksha.core.document_loader import Document


class DocumentChunk(BaseModel):
    """A chunk of text derived from a Document, ready for embedding and indexing."""
    id: str
    doc_id: str
    chunk_index: int
    text: str
    source_title: str
    source_path: str
    page_number: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


class RecursiveTextSplitter:
    """
    Recursively splits text on natural boundaries (paragraphs, newlines, sentences, words)
    while respecting target chunk size and overlap.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        separators: Optional[list[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        """Split a string into chunks of approximately `chunk_size` characters with overlap."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            # Base case: split strictly by character slice
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start += self.chunk_size - self.chunk_overlap
            return chunks

        sep = separators[0]
        remaining_separators = separators[1:]

        if sep == "":
            splits = list(text)
        else:
            splits = text.split(sep)

        chunks: list[str] = []
        current_doc: list[str] = []
        total_len = 0

        for s in splits:
            s_len = len(s) + (len(sep) if current_doc else 0)
            if total_len + s_len <= self.chunk_size:
                current_doc.append(s)
                total_len += s_len
            else:
                if current_doc:
                    joined = sep.join(current_doc).strip()
                    if joined:
                        chunks.append(joined)
                    # Handle overlap by taking tail elements from previous chunk
                    overlap_doc: list[str] = []
                    overlap_len = 0
                    for elem in reversed(current_doc):
                        e_len = len(elem) + len(sep)
                        if overlap_len + e_len <= self.chunk_overlap:
                            overlap_doc.insert(0, elem)
                            overlap_len += e_len
                        else:
                            break
                    current_doc = overlap_doc
                    total_len = overlap_len

                # If a single item is longer than chunk_size, recursively split it with finer separator
                if len(s) > self.chunk_size:
                    sub_chunks = self._split(s, remaining_separators)
                    chunks.extend(sub_chunks)
                else:
                    current_doc.append(s)
                    total_len += len(s) + (len(sep) if current_doc else 0)

        if current_doc:
            joined = sep.join(current_doc).strip()
            if joined:
                chunks.append(joined)

        return chunks

    def split_document(self, doc: Document) -> list[DocumentChunk]:
        """Split a Document into DocumentChunks carrying parent metadata."""
        text_chunks = self.split_text(doc.content)
        chunks: list[DocumentChunk] = []

        for idx, text in enumerate(text_chunks):
            chunk_id = f"{doc.id}_c{idx}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    doc_id=doc.id,
                    chunk_index=idx,
                    text=text,
                    source_title=doc.title,
                    source_path=doc.source_path,
                    page_number=doc.page_number,
                    metadata={
                        **doc.metadata,
                        "file_type": doc.file_type,
                    },
                )
            )

        return chunks

    def split_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        """Split a collection of Documents into DocumentChunks."""
        all_chunks: list[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
