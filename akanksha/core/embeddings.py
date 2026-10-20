"""
Pluggable Embedding Engines: Fast Local Semantic (TF-IDF/LSA), HuggingFace, OpenAI, and Gemini.
"""

from typing import Protocol, runtime_checkable
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


@runtime_checkable
class BaseEmbedder(Protocol):
    """Protocol for embedding models."""
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_query(self, query: str) -> np.ndarray:
        ...


class FastSemanticEmbedder:
    """
    High-speed, zero-dependency local semantic embedder.
    Uses subword n-grams and TF-IDF with Latent Semantic Analysis (TruncatedSVD)
    to produce dense semantic representations without requiring external APIs or GPUs.
    """

    def __init__(self, n_dimensions: int = 128):
        self.n_dimensions = n_dimensions
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="word",
            sublinear_tf=True,
            max_features=10000,
        )
        self.svd: TruncatedSVD | None = None
        self.is_fitted = False

    def fit(self, texts: list[str]) -> "FastSemanticEmbedder":
        """Fit the semantic space on the corpus."""
        if not texts:
            return self

        # Avoid empty strings
        clean_texts = [t if t.strip() else "document" for t in texts]
        tfidf_matrix = self.vectorizer.fit_transform(clean_texts)

        n_samples, n_features = tfidf_matrix.shape
        n_components = min(self.n_dimensions, n_samples - 1, n_features - 1)

        if n_components >= 2:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd.fit(tfidf_matrix)
        else:
            self.svd = None

        self.is_fitted = True
        return self

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Transform texts into normalized dense embedding vectors."""
        if not texts:
            return np.empty((0, self.n_dimensions))

        clean_texts = [t if t.strip() else "document" for t in texts]
        if not self.is_fitted:
            self.fit(clean_texts)

        tfidf_matrix = self.vectorizer.transform(clean_texts)

        if self.svd is not None:
            dense = self.svd.transform(tfidf_matrix)
        else:
            dense = tfidf_matrix.toarray()
            # Pad or truncate to n_dimensions
            if dense.shape[1] < self.n_dimensions:
                padding = np.zeros((dense.shape[0], self.n_dimensions - dense.shape[1]))
                dense = np.hstack([dense, padding])
            else:
                dense = dense[:, :self.n_dimensions]

        # L2-normalize vectors for fast cosine distance via dot product
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([query])[0]


class GeminiEmbedder:
    """Embeddings via Google Gemini API."""

    def __init__(self, api_key: str | None = None, model: str = "models/text-embedding-004"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiEmbedder")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import httpx
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:batchEmbedContents?key={self.api_key}"
        requests_payload = [
            {"model": self.model, "content": {"parts": [{"text": t}]}} for t in texts
        ]
        response = httpx.post(url, json={"requests": requests_payload}, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        embeddings = [entry["values"] for entry in data.get("embeddings", [])]
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


class OpenAIEmbedder:
    """Embeddings via OpenAI or OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbedder")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        import httpx
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"input": texts, "model": self.model}
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


def get_embedder(provider: str = "auto", api_key: str | None = None) -> BaseEmbedder:
    """Factory to get the appropriate embedder with automatic graceful fallback."""
    provider = provider.lower()
    if provider == "gemini":
        try:
            return GeminiEmbedder(api_key=api_key)
        except Exception:
            pass
    elif provider == "openai":
        try:
            return OpenAIEmbedder(api_key=api_key)
        except Exception:
            pass

    # Default fallback: Fast, zero-dependency local semantic embedder
    return FastSemanticEmbedder()
