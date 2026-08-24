import json
from typing import Optional
import typer
from rich.panel import Panel
from rich.table import Table

from .config import settings
from .tools.registry import default_registry
from .agent.harness import AgentHarness
from .ui.console import AgentUI, console

app = typer.Typer(help="FDE AI Agent Harness & Developer CLI")


@app.command()
def status() -> None:
    """Check configuration, API key connectivity, and registered tools."""
    table = Table(title="FDE Agent Harness Status", border_style="magenta")
    table.add_column("Setting / Component", style="cyan")
    table.add_column("Value / State", style="white")

    key_status = (
        "[bold green]Configured (Ready for Live Claude API)[/bold green]"
        if settings.has_api_key
        else "[bold yellow]Missing / Empty (Operating in Mock Mode)[/bold yellow]"
    )
    table.add_row("Anthropic API Key", key_status)
    table.add_row("Configured Model", settings.model)
    table.add_row("Max Turns / Limits", str(settings.max_turns))
    table.add_row("Registered Tools", f"{len(default_registry.list_tools())} tools active")

    console.print(table)

    # Show registered tools summary
    tools_table = Table(title="Registered Tools", border_style="yellow")
    tools_table.add_column("Tool Name", style="bold yellow")
    tools_table.add_column("Description", style="white")

    for tool in default_registry.list_tools():
        tools_table.add_row(tool.name, tool.description)

    console.print(tools_table)
    console.print("\n[dim]Tip: Run 'uv run main.py chat \"<your query>\"' to test the execution loop.[/dim]")


@app.command()
def tools() -> None:
    """Inspect tool schemas formatted for Anthropic Claude function calling."""
    console.print("[bold cyan]Anthropic Function Calling Tool Schemas:[/bold cyan]\n")
    schemas = default_registry.get_schemas()
    for schema in schemas:
        console.print(
            Panel(
                json.dumps(schema, indent=2),
                title=f"[bold yellow]Tool: {schema['name']}[/bold yellow]",
                border_style="yellow",
            )
        )


@app.command()
def chat(
    message: str = typer.Argument(..., help="The prompt to send to the AI agent."),
    mock: bool = typer.Option(False, "--mock", help="Force mock engine even if API key is present."),
    live: bool = typer.Option(False, "--live", help="Force live API (errors if key is missing)."),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show turn boundaries and message state history."),
    max_turns: Optional[int] = typer.Option(None, "--max-turns", "-m", help="Override max loop iterations."),
) -> None:
    """Send a query to the agent harness and execute any required tools."""
    if live and not settings.has_api_key:
        console.print("[bold red]Error:[/bold red] Live mode requested but ANTHROPIC_API_KEY is not set in .env.")
        raise typer.Exit(code=1)

    force_mock = mock or (not live and not settings.has_api_key)
    if max_turns:
        settings.max_turns = max_turns

    harness = AgentHarness(config=settings, registry=default_registry, force_mock=force_mock)
    AgentUI.print_banner(mode=harness.mode, model=settings.model)

    response = harness.run(prompt=message, debug=debug)

    # Show execution metrics
    console.print(
        f"\n[dim]📊 Execution Metrics: {response.turns_taken} turn(s) | "
        f"Tokens: ~{response.total_input_tokens} in / ~{response.total_output_tokens} out | "
        f"Mode: {response.mode}[/dim]\n"
    )


@app.command()
def repl(
    mock: bool = typer.Option(False, "--mock", help="Force mock engine."),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show debug state on every turn."),
) -> None:
    """Start an interactive multi-turn chat session with persistent context."""
    force_mock = mock or not settings.has_api_key
    harness = AgentHarness(config=settings, registry=default_registry, force_mock=force_mock)

    AgentUI.print_banner(mode=harness.mode, model=settings.model)
    console.print("[dim]Interactive REPL session started. Type 'exit', 'quit', or 'clear' to manage session.[/dim]\n")

    history = []
    while True:
        try:
            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Exiting session. Goodbye![/dim]")
                break
            if user_input.lower() == "clear":
                history = []
                console.print("[dim]Conversation history cleared.[/dim]")
                continue

            response = harness.run(prompt=user_input, conversation_history=history, debug=debug)
            # Persist state
            history = response.messages

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated.[/dim]")
            break


def main() -> None:
    app()
