"""
Document ingestion engine supporting PDFs, Markdown, Code, Notebooks, and Text.
"""

from pathlib import Path
from typing import Optional
import json
import hashlib
from pydantic import BaseModel, Field
import pypdf


class Document(BaseModel):
    """Represents an ingested document or document section."""
    id: str
    title: str
    content: str
    source_path: str
    file_type: str
    page_number: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


class DocumentLoader:
    """Loads and extracts text and metadata from supported file formats."""

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of a file for caching and change detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    @classmethod
    def load_file(cls, file_path: str | Path) -> list[Document]:
        """Load a file and return a list of Document objects (split by page for PDFs)."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        file_hash = cls.compute_file_hash(path)

        if ext == ".pdf":
            return cls._load_pdf(path, file_hash)
        elif ext in [".md", ".markdown"]:
            return cls._load_text(path, file_hash, file_type="markdown")
        elif ext in [".txt", ".text"]:
            return cls._load_text(path, file_hash, file_type="text")
        elif ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".json", ".csv"]:
            return cls._load_code(path, file_hash, ext)
        elif ext == ".ipynb":
            return cls._load_notebook(path, file_hash)
        else:
            # Fallback to UTF-8 text loader
            return cls._load_text(path, file_hash, file_type="unknown")

    @classmethod
    def _load_pdf(cls, path: Path, file_hash: str) -> list[Document]:
        """Extract text from PDF page by page using pypdf."""
        documents: list[Document] = []
        try:
            reader = pypdf.PdfReader(str(path))
            total_pages = len(reader.pages)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                clean_text = text.strip()
                if clean_text:
                    doc_id = f"{file_hash}_p{page_idx + 1}"
                    documents.append(
                        Document(
                            id=doc_id,
                            title=path.name,
                            content=clean_text,
                            source_path=str(path),
                            file_type="pdf",
                            page_number=page_idx + 1,
                            metadata={
                                "file_name": path.name,
                                "total_pages": total_pages,
                                "page_number": page_idx + 1,
                                "file_hash": file_hash,
                                "file_size": path.stat().st_size,
                            },
                        )
                    )
        except Exception as e:
            raise RuntimeError(f"Error reading PDF {path.name}: {e}") from e

        return documents

    @classmethod
    def _load_text(cls, path: Path, file_hash: str, file_type: str) -> list[Document]:
        """Read standard text/markdown files."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
        except Exception as e:
            raise RuntimeError(f"Error reading text file {path.name}: {e}") from e

        if not content:
            return []

        return [
            Document(
                id=f"{file_hash}_0",
                title=path.name,
                content=content,
                source_path=str(path),
                file_type=file_type,
                metadata={
                    "file_name": path.name,
                    "file_hash": file_hash,
                    "file_size": path.stat().st_size,
                },
            )
        ]

    @classmethod
    def _load_code(cls, path: Path, file_hash: str, ext: str) -> list[Document]:
        """Read code files, formatting as annotated code blocks."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                code_text = f.read().strip()
        except Exception as e:
            raise RuntimeError(f"Error reading code file {path.name}: {e}") from e

        if not code_text:
            return []

        annotated = f"```{ext.lstrip('.')}\n// File: {path.name}\n{code_text}\n```"
        return [
            Document(
                id=f"{file_hash}_0",
                title=path.name,
                content=annotated,
                source_path=str(path),
                file_type="code",
                metadata={
                    "file_name": path.name,
                    "extension": ext,
                    "file_hash": file_hash,
                    "file_size": path.stat().st_size,
                },
            )
        ]

    @classmethod
    def _load_notebook(cls, path: Path, file_hash: str) -> list[Document]:
        """Extract markdown and code from Jupyter Notebook (.ipynb)."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                nb_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error parsing notebook {path.name}: {e}") from e

        extracted_parts: list[str] = []
        cells = nb_data.get("cells", [])
        for cell_idx, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", [])).strip()
            if not source:
                continue

            if cell_type == "markdown":
                extracted_parts.append(source)
            elif cell_type == "code":
                extracted_parts.append(f"```python\n# [Cell {cell_idx + 1}]\n{source}\n```")

        combined = "\n\n".join(extracted_parts).strip()
        if not combined:
            return []

        return [
            Document(
                id=f"{file_hash}_0",
                title=path.name,
                content=combined,
                source_path=str(path),
                file_type="jupyter_notebook",
                metadata={
                    "file_name": path.name,
                    "cell_count": len(cells),
                    "file_hash": file_hash,
                    "file_size": path.stat().st_size,
                },
            )
        ]

    @classmethod
    def load_directory(
        cls,
        dir_path: str | Path,
        supported_extensions: Optional[list[str]] = None,
        recursive: bool = True,
    ) -> list[Document]:
        """Scan and load all supported documents from a directory."""
        directory = Path(dir_path).resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        if supported_extensions is None:
            supported_extensions = [
                ".pdf", ".txt", ".md", ".markdown", ".py", ".json", ".csv", ".ipynb", ".js", ".html"
            ]

        valid_exts = {ext.lower() for ext in supported_extensions}
        pattern = "**/*" if recursive else "*"
        all_docs: list[Document] = []

        for p in directory.glob(pattern):
            if p.is_file() and p.suffix.lower() in valid_exts and not p.name.startswith("."):
                try:
                    docs = cls.load_file(p)
                    all_docs.extend(docs)
                except Exception as err:
                    # Non-fatal: log and skip corrupted files
                    print(f"Skipping {p.name}: {err}")

        return all_docs
