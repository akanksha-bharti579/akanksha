"""
Configuration settings for Akanksha Second Brain.
"""

from pathlib import Path
import os
from typing import Optional
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = BASE_DIR / ".brain_data"
DEFAULT_UPLOADS_DIR = DEFAULT_DATA_DIR / "documents"
DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "index"

class BrainConfig(BaseModel):
    """Global configuration for the Second Brain engine."""
    
    # Storage Paths
    data_dir: Path = Field(default=DEFAULT_DATA_DIR)
    uploads_dir: Path = Field(default=DEFAULT_UPLOADS_DIR)
    index_dir: Path = Field(default=DEFAULT_INDEX_DIR)
    
    # Text chunking settings
    chunk_size: int = Field(default=600, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=120, description="Overlap between consecutive chunks")
    
    # Retrieval settings
    top_k: int = Field(default=4, description="Number of context passages to retrieve")
    min_similarity_score: float = Field(default=0.15, description="Minimum similarity threshold")
    
    # Supported document extensions
    supported_extensions: list[str] = Field(
        default=[".pdf", ".txt", ".md", ".py", ".json", ".csv", ".js", ".ts", ".html", ".ipynb"]
    )
    
    # LLM Settings
    # Options: "offline", "gemini", "openai", "ollama"
    llm_provider: str = Field(
        default=os.getenv("AKANKSHA_LLM_PROVIDER", "offline")
    )
    
    # API Keys & Endpoints (from environment variables or settings)
    gemini_api_key: Optional[str] = Field(default=os.getenv("GEMINI_API_KEY"))
    gemini_model: str = Field(default="gemini-1.5-flash")
    
    openai_api_key: Optional[str] = Field(default=os.getenv("OPENAI_API_KEY"))
    openai_base_url: Optional[str] = Field(default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_model: str = Field(default="gpt-4o-mini")
    
    ollama_base_url: str = Field(default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = Field(default="llama3")
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


default_config = BrainConfig()
