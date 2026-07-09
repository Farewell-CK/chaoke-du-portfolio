import json
import typer
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from portfolio_cli.client import get_client

app = typer.Typer(name="portfolio-cli", help="Manage your portfolio website")
console = Console()


@app.command()
def status():
    """Check website status."""
    client = get_client()
    try:
        r = client.get("/api/search?q=test")
        rprint(f"[green]Status: Online[/green] ({r.status_code})")
        rprint(f"URL: {client.base_url}")
    except Exception as e:
        rprint(f"[red]Status: Offline[/red] ({e})")


@app.command()
def stats():
    """Show website statistics."""
    client = get_client()
    try:
        r = client.get("/api/admin/messages")
        if r.status_code == 200:
            messages = r.json()
            pending = sum(1 for m in messages if m.get("status") == "pending")
            rprint(f"[cyan]Messages:[/cyan] {len(messages)} total, {pending} pending")
        else:
            rprint(f"[yellow]Messages: auth required[/yellow]")

        r2 = client.get("/api/admin/contacts")
        if r2.status_code == 200:
            contacts = r2.json()
            rprint(f"[cyan]Contacts:[/cyan] {len(contacts)}")
        else:
            rprint(f"[yellow]Contacts: auth required[/yellow]")
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")


# --- Messages ---
messages_app = typer.Typer(help="Manage guestbook messages")
app.add_typer(messages_app, name="message")


@messages_app.command("list")
def message_list():
    """List all messages."""
    client = get_client()
    r = client.get("/api/admin/messages")
    if r.status_code != 200:
        rprint(f"[red]Error: {r.text}[/red]")
        return

    messages = r.json()
    table = Table(title="Messages")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Content", style="white")
    table.add_column("Date", style="dim")

    for m in messages:
        table.add_row(
            str(m["id"]),
            m["name"],
            m["status"],
            m["content"][:50] + "..." if len(m["content"]) > 50 else m["content"],
            m["created_at"][:10],
        )

    console.print(table)


@messages_app.command("approve")
def message_approve(id: int):
    """Approve a message."""
    client = get_client()
    r = client.patch(f"/api/messages/{id}", json={"status": "approved"})
    if r.status_code == 200:
        rprint(f"[green]Message {id} approved[/green]")
    else:
        rprint(f"[red]Error: {r.text}[/red]")


@messages_app.command("delete")
def message_delete(id: int):
    """Delete a message."""
    client = get_client()
    r = client.delete(f"/api/messages/{id}")
    if r.status_code == 200:
        rprint(f"[green]Message {id} deleted[/green]")
    else:
        rprint(f"[red]Error: {r.text}[/red]")


# --- Posts ---
posts_app = typer.Typer(help="Manage blog posts")
app.add_typer(posts_app, name="post")


@posts_app.command("list")
def post_list():
    """List blog posts (from content directory)."""
    import os
    from pathlib import Path

    blog_dir = Path(os.environ.get("PORTFOLIO_CONTENT_DIR", "content/blog"))
    if not blog_dir.exists():
        rprint(f"[red]Blog directory not found: {blog_dir}[/red]")
        return

    table = Table(title="Blog Posts")
    table.add_column("Slug", style="cyan")
    table.add_column("File", style="dim")

    for f in sorted(blog_dir.glob("*.md")):
        table.add_row(f.stem, str(f))

    console.print(table)


# --- Search ---
@app.command()
def search(query: str):
    """Search the website."""
    client = get_client()
    r = client.get(f"/api/search?q={query}")
    if r.status_code != 200:
        rprint(f"[red]Error: {r.text}[/red]")
        return

    results = r.json().get("results", [])
    if not results:
        rprint("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f'Search: "{query}"')
    table.add_column("Type", style="dim")
    table.add_column("Title", style="cyan")
    table.add_column("Description", style="white")

    for res in results:
        table.add_row(res["type"], res["title"], res["description"][:60])

    console.print(table)


if __name__ == "__main__":
    app()
