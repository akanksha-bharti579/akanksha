"""
Fast Starlette backend serving REST API and Web UI for Akanksha's Second Brain.
"""

from pathlib import Path
import json
import shutil
from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from akanksha.core.rag_engine import SecondBrain
from akanksha.core.config import default_config

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

brain = SecondBrain(default_config)


async def api_status(request: Request) -> JSONResponse:
    """Get system health, stats, and provider."""
    docs = brain.list_documents()
    total_chunks = len(brain.vector_store.chunks_cache)
    return JSONResponse({
        "status": "online",
        "name": "Akanksha AI Second Brain",
        "documents_count": len(docs),
        "total_chunks": total_chunks,
        "llm_provider": brain.config.llm_provider,
        "storage_dir": str(brain.config.data_dir),
    })


async def api_list_documents(request: Request) -> JSONResponse:
    """List all indexed documents."""
    docs = brain.list_documents()
    return JSONResponse({"documents": docs})


async def api_upload(request: Request) -> JSONResponse:
    """Handle document file upload and automatic indexing."""
    try:
        form = await request.form()
        file_field = form.get("file")
        if not file_field:
            return JSONResponse({"error": "No file uploaded"}, status_code=400)

        filename = file_field.filename
        content = await file_field.read()

        # Save to uploads dir
        save_path = brain.config.uploads_dir / filename
        with open(save_path, "wb") as f:
            f.write(content)

        # Ingest file
        result = brain.ingest_file(save_path)
        return JSONResponse({"success": True, "result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_query(request: Request) -> JSONResponse:
    """Execute grounded RAG query."""
    try:
        data = await request.json()
        query_text = data.get("query", "").strip()
        provider = data.get("provider")
        api_key = data.get("api_key")
        model = data.get("model")

        if not query_text:
            return JSONResponse({"error": "Query cannot be empty"}, status_code=400)

        if provider:
            brain.set_provider(provider=provider, api_key=api_key, model=model)

        response = brain.query(question=query_text)
        return JSONResponse(response.model_dump())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_summarize(request: Request) -> JSONResponse:
    """Summarize a specific document."""
    try:
        data = await request.json()
        title = data.get("title", "").strip()
        if not title:
            return JSONResponse({"error": "Document title required"}, status_code=400)

        summary = brain.summarize(title)
        return JSONResponse({"title": title, "summary": summary})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_delete_document(request: Request) -> JSONResponse:
    """Delete a document from index."""
    title = request.path_params.get("title", "")
    count = brain.delete_document(title)
    return JSONResponse({"deleted": count > 0, "chunks_removed": count})


async def serve_index(request: Request) -> FileResponse:
    """Serve single-page application index.html."""
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)


routes = [
    Route("/api/status", api_status, methods=["GET"]),
    Route("/api/documents", api_list_documents, methods=["GET"]),
    Route("/api/upload", api_upload, methods=["POST"]),
    Route("/api/query", api_query, methods=["POST"]),
    Route("/api/summarize", api_summarize, methods=["POST"]),
    Route("/api/documents/{title:path}", api_delete_document, methods=["DELETE"]),
    Route("/", serve_index, methods=["GET"]),
    Mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static"),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=False, routes=routes, middleware=middleware)
