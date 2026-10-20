"""
Akanksha AI - Personal AI Second Brain & Research Assistant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A modern, local Retrieval-Augmented Generation (RAG) system to
organize, index, search, and converse with your documents, research
papers, and study notes.
"""

from akanksha.core.rag_engine import SecondBrain
from akanksha.core.document_loader import DocumentLoader, Document
from akanksha.core.text_splitter import RecursiveTextSplitter, DocumentChunk
from akanksha.core.vector_store import VectorStore

__version__ = "0.1.0"
__author__ = "Akanksha Bharti"
__all__ = [
    "SecondBrain",
    "DocumentLoader",
    "Document",
    "RecursiveTextSplitter",
    "DocumentChunk",
    "VectorStore",
    "__version__",
]
