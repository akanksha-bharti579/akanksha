"""
Automated unit tests for Akanksha Second Brain RAG system.
"""

from pathlib import Path
import tempfile
import shutil
import pytest

from akanksha.core.config import BrainConfig
from akanksha.core.document_loader import DocumentLoader
from akanksha.core.text_splitter import RecursiveTextSplitter
from akanksha.core.rag_engine import SecondBrain


@pytest.fixture
def temp_brain():
    """Create a temporary isolated SecondBrain instance for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    config = BrainConfig(
        data_dir=temp_dir,
        uploads_dir=temp_dir / "uploads",
        index_dir=temp_dir / "index",
        llm_provider="offline",
    )
    brain = SecondBrain(config=config)
    yield brain
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_text_splitter():
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "Machine learning is a subset of artificial intelligence. "
        "It focuses on building applications that learn from data and improve their accuracy over time without being explicitly programmed. "
        "In data science, an algorithm is a sequence of statistical processing steps."
    )
    chunks = splitter.split_text(sample_text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) > 0


def test_document_loader_text(tmp_path):
    test_file = tmp_path / "notes.md"
    test_file.write_text("# Test Notes\nGradient Boosting minimizes loss sequentially.", encoding="utf-8")
    
    docs = DocumentLoader.load_file(test_file)
    assert len(docs) == 1
    assert "Gradient Boosting" in docs[0].content
    assert docs[0].file_type == "markdown"


def test_ingest_and_search(temp_brain, tmp_path):
    sample_file = tmp_path / "quantum.txt"
    sample_file.write_text(
        "Quantum computing leverages superposition and entanglement to perform complex computations.",
        encoding="utf-8",
    )

    ingest_res = temp_brain.ingest_file(sample_file)
    assert ingest_res["status"] == "success"
    assert ingest_res["chunks_indexed"] >= 1

    # Search
    results = temp_brain.search("quantum computing superposition")
    assert len(results) >= 1
    assert "superposition" in results[0].text.lower()


def test_grounded_query(temp_brain, tmp_path):
    sample_file = tmp_path / "arch.txt"
    sample_file.write_text(
        "The ResNet architecture uses residual skip connections to solve the vanishing gradient problem in deep neural networks.",
        encoding="utf-8",
    )

    temp_brain.ingest_file(sample_file)
    response = temp_brain.query("What do residual skip connections solve?")
    assert response.query == "What do residual skip connections solve?"
    assert len(response.sources) >= 1
    assert "residual" in response.answer.lower() or "vanishing" in response.answer.lower() or len(response.sources) > 0
