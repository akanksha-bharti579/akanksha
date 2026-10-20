"""
Quickstart Example: Using Akanksha's AI Second Brain programmatically in Python.
"""

import sys
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from akanksha import SecondBrain

def main():
    print("=" * 60)
    print("[*] Akanksha AI Second Brain - Python SDK Quickstart")
    print("=" * 60)

    # 1. Initialize Second Brain
    brain = SecondBrain()

    # 2. Ingest the sample study notes
    sample_doc = Path(__file__).parent / "deep_learning_and_rag_notes.md"
    print(f"\n📂 Ingesting sample document: {sample_doc.name}...")
    ingest_result = brain.ingest_file(sample_doc)
    print(f"✔ Ingested {ingest_result['chunks_indexed']} chunks in {ingest_result['duration_ms']}ms!")

    # 3. Perform a semantic & hybrid search
    query_1 = "How does Gradient Boosting differ from Random Forest?"
    print(f"\n🔍 Searching for: '{query_1}'...")
    search_results = brain.search(query_1, top_k=2)
    for idx, r in enumerate(search_results, 1):
        print(f"  [{idx}] {r.source_title} (Similarity: {r.similarity_score:.3f})")
        print(f"      Snippet: {r.text[:120]}...\n")

    # 4. Ask a grounded question with automatic source citations
    query_2 = "What is the Scaled Dot-Product Attention formula and its components?"
    print(f"\n🤖 Asking grounded question: '{query_2}'...")
    response = brain.query(query_2)
    print("\n--- Answer ---")
    print(response.answer)
    print("\n--- Source Citations ---")
    for src in response.sources:
        print(f"• {src.title} (Confidence: {src.similarity_score:.3f})")
    print(f"\n⏱ Query completed in {response.latency_ms}ms")


if __name__ == "__main__":
    main()
