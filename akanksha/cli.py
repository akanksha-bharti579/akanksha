import sys
import os
import argparse
from pathlib import Path

# Configure UTF-8 for Windows PowerShell and CMD
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box

from akanksha.core.rag_engine import SecondBrain
from akanksha.core.config import default_config

# Safe console for Windows
console = Console(force_terminal=True)


def print_banner() -> None:
    """Display rich welcome banner."""
    banner_text = (
        "[bold cyan]Akanksha's AI Second Brain & Research Assistant[/bold cyan]\n"
        "[dim]Local Retrieval-Augmented Generation (RAG) | Semantic Search | Grounded Q&A[/dim]"
    )
    console.print(Panel(banner_text, border_style="cyan", box=box.ROUNDED))


def cmd_ingest(brain: SecondBrain, target_path: str) -> None:
    """Ingest a file or directory of documents."""
    path = Path(target_path).resolve()
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] Path does not exist: {path}")
        return

    with console.status(f"[bold green]Ingesting {path.name}...[/bold green]"):
        if path.is_file():
            res = brain.ingest_file(path)
            console.print(
                f"[bold green][+] Ingested:[/bold green] {res['file_name']} "
                f"([cyan]{res.get('chunks_indexed', 0)} chunks[/cyan], {res.get('duration_ms', 0)}ms)"
            )
        elif path.is_dir():
            res = brain.ingest_directory(path)
            console.print(
                f"[bold green][+] Ingested Directory:[/bold green] {path.name} "
                f"([cyan]{res.get('files_indexed', 0)} files[/cyan], "
                f"[cyan]{res.get('chunks_indexed', 0)} total chunks[/cyan], {res.get('duration_ms', 0)}ms)"
            )


def cmd_list(brain: SecondBrain) -> None:
    """Display table of all indexed documents."""
    docs = brain.list_documents()
    if not docs:
        console.print("[yellow]Your Second Brain is empty. Run `akanksha ingest <file>` to add notes or papers.[/yellow]")
        return

    table = Table(title="[bold]Indexed Knowledge Base[/bold]", box=box.ROUNDED, border_style="cyan")
    table.add_column("Document Title", style="bold white")
    table.add_column("Type", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Characters", justify="right", style="magenta")

    for d in docs:
        table.add_row(
            d["title"],
            d["file_type"],
            str(d["total_chunks"]),
            f"{d['character_count']:,}",
        )

    console.print(table)


def cmd_search(brain: SecondBrain, query: str, top_k: int = 4) -> None:
    """Perform hybrid search and display ranked snippets."""
    with console.status(f"[bold green]Searching for '{query}'...[/bold green]"):
        results = brain.search(query, top_k=top_k)

    if not results:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    console.print(f"\n[bold cyan]Top {len(results)} matches for:[/bold cyan] [italic]\"{query}\"[/italic]\n")
    for idx, r in enumerate(results, 1):
        page_info = f" | Page {r.page_number}" if r.page_number else ""
        header = f"#{idx} [bold yellow]{r.source_title}[/bold yellow]{page_info} (Score: [bold green]{r.similarity_score:.3f}[/bold green])"
        console.print(Panel(r.text, title=header, border_style="blue", box=box.ROUNDED))


def cmd_ask(brain: SecondBrain, question: str, top_k: int = 4) -> None:
    """Ask a question and receive a grounded answer with citations."""
    with console.status("[bold green]Thinking and synthesizing grounded answer...[/bold green]"):
        resp = brain.query(question, top_k=top_k)

    console.print()
    console.print(Panel(Markdown(resp.answer), title="[bold green]Assistant Answer[/bold green]", border_style="green", box=box.ROUNDED))

    if resp.sources:
        console.print("\n[bold cyan][*] Source Citations:[/bold cyan]")
        for idx, src in enumerate(resp.sources, 1):
            page_info = f" (Page {src.page_number})" if src.page_number else ""
            console.print(
                f"  [{idx}] [bold]{src.title}[/bold]{page_info} - "
                f"[dim]Score: {src.similarity_score:.3f}[/dim]\n"
                f"      [dim italic]\"{src.snippet}\"[/dim italic]"
            )
    console.print(f"\n[dim]Latency: {resp.latency_ms}ms | Searched {resp.total_chunks_searched} chunks[/dim]\n")


def cmd_chat(brain: SecondBrain) -> None:
    """Start an interactive chat session."""
    print_banner()
    console.print("[bold yellow]Interactive Chat Session Started.[/bold yellow]")
    console.print("[dim]Type your question or 'exit' / 'quit' to end.\n[/dim]")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[bold cyan]Goodbye! Keep learning.[/bold cyan]")
                break

            cmd_ask(brain, user_input)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Session closed.[/bold cyan]")
            break


def cmd_web(port: int = 8000, host: str = "127.0.0.1") -> None:
    """Launch the Web UI dashboard."""
    import uvicorn
    from akanksha.web.server import app

    print_banner()
    console.print(f"[bold green]Starting Web UI on:[/bold green] [bold cyan]http://{host}:{port}[/bold cyan]")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    """CLI Argument Parser entrypoint."""
    parser = argparse.ArgumentParser(
        prog="akanksha",
        description="Akanksha's AI Second Brain & Research Assistant",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest
    ingest_p = subparsers.add_parser("ingest", help="Ingest a document or directory")
    ingest_p.add_argument("path", help="Path to file or folder")

    # List
    subparsers.add_parser("list", help="List all indexed documents")

    # Search
    search_p = subparsers.add_parser("search", help="Perform semantic search")
    search_p.add_argument("query", help="Search query string")
    search_p.add_argument("-k", "--top-k", type=int, default=4, help="Number of results")

    # Ask
    ask_p = subparsers.add_parser("ask", help="Ask a question grounded in your documents")
    ask_p.add_argument("question", help="Question to ask")
    ask_p.add_argument("-k", "--top-k", type=int, default=4, help="Top passages to retrieve")

    # Summarize
    sum_p = subparsers.add_parser("summarize", help="Executive summary of a document")
    sum_p.add_argument("title", help="Document title or file name")

    # Chat
    subparsers.add_parser("chat", help="Start an interactive terminal chat")

    # Web UI
    web_p = subparsers.add_parser("web", help="Launch interactive Web UI")
    web_p.add_argument("-p", "--port", type=int, default=8000, help="Web server port")
    web_p.add_argument("-H", "--host", default="127.0.0.1", help="Host interface")

    # Clear
    subparsers.add_parser("clear", help="Clear all stored documents and vectors")

    args = parser.parse_args()

    # If no arguments provided, open chat
    if not args.command:
        brain = SecondBrain()
        cmd_chat(brain)
        return

    brain = SecondBrain()

    if args.command == "ingest":
        cmd_ingest(brain, args.path)
    elif args.command == "list":
        cmd_list(brain)
    elif args.command == "search":
        cmd_search(brain, args.query, top_k=args.top_k)
    elif args.command == "ask":
        cmd_ask(brain, args.question, top_k=args.top_k)
    elif args.command == "summarize":
        with console.status(f"[bold green]Summarizing {args.title}...[/bold green]"):
            summary = brain.summarize(args.title)
        console.print(Panel(Markdown(summary), title=f"📋 Summary: {args.title}", border_style="cyan"))
    elif args.command == "chat":
        cmd_chat(brain)
    elif args.command == "web":
        cmd_web(port=args.port, host=args.host)
    elif args.command == "clear":
        brain.clear()
        console.print("[bold red]✔ Cleared all indexed documents and vector stores.[/bold red]")


if __name__ == "__main__":
    main()
