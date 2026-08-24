import json
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "user": "bold cyan",
    "agent": "bold magenta",
    "tool": "bold yellow",
})

console = Console(theme=custom_theme)


class AgentUI:
    """Rich visual output helper for the agent execution lifecycle."""

    @staticmethod
    def print_banner(mode: str, model: str) -> None:
        """Display the CLI banner and engine mode."""
        mode_badge = (
            "[bold green]LIVE API[/bold green]"
            if mode == "live"
            else "[bold yellow]MOCK ENGINE (Keyless Mode)[/bold yellow]"
        )
        console.print(
            Panel(
                f"[bold white]FDE AI Agent Harness[/bold white] | Engine: {mode_badge} | Model: [cyan]{model}[/cyan]\n"
                f"[dim]Building from scratch: Raw ReAct loop, tool schemas, state persistence.[/dim]",
                border_style="magenta",
                padding=(0, 2),
            )
        )

    @staticmethod
    def print_user_prompt(prompt: str) -> None:
        """Render the user prompt."""
        console.print(f"\n[user]🧑 User:[/user] {prompt}")

    @staticmethod
    def print_turn_header(turn: int, max_turns: int) -> None:
        """Display turn boundary in verbose/debug mode."""
        console.print(f"\n[dim]─── [Turn {turn}/{max_turns}] Model Step ───────────────────────────────────────────[/dim]")

    @staticmethod
    def print_tool_call(tool_name: str, tool_id: str, tool_input: Dict[str, Any]) -> None:
        """Render a tool call invocation with arguments formatted nicely."""
        formatted_json = json.dumps(tool_input, indent=2)
        syntax = Syntax(formatted_json, "json", theme="monokai", line_numbers=False)
        console.print(
            Panel(
                syntax,
                title=f"[tool]⚙️ Tool Call:[/tool] [bold white]{tool_name}[/bold white] [dim]({tool_id})[/dim]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    @staticmethod
    def print_tool_result(tool_name: str, tool_id: str, result_str: str) -> None:
        """Render the tool output received from local execution."""
        # Truncate if extremely long
        display_text = result_str if len(result_str) < 1500 else result_str[:1500] + "\n... [truncated]"
        try:
            parsed = json.loads(result_str)
            syntax = Syntax(json.dumps(parsed, indent=2), "json", theme="monokai", line_numbers=False)
            content = syntax
        except Exception:
            content = display_text

        console.print(
            Panel(
                content,
                title=f"[success]✔️ Tool Result:[/success] [bold white]{tool_name}[/bold white]",
                border_style="green",
                padding=(0, 1),
            )
        )

    @staticmethod
    def print_agent_response(text: str) -> None:
        """Render the agent's final text response in formatted Markdown."""
        console.print(f"\n[agent]🤖 Claude Agent:[/agent]")
        console.print(Markdown(text))

    @staticmethod
    def print_debug_messages(messages: List[Dict[str, Any]]) -> None:
        """Render raw message history JSON for deep inspection."""
        table = Table(title="Message History (State Inspection)", border_style="dim")
        table.add_column("Step", justify="center", style="cyan", no_wrap=True)
        table.add_column("Role", style="magenta", no_wrap=True)
        table.add_column("Content Summary", style="white")

        for idx, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                summary = f"[{len(content)} block(s)] " + ", ".join([b.get("type", "unknown") for b in content if isinstance(b, dict)])
            else:
                summary = str(content)[:80] + ("..." if len(str(content)) > 80 else "")
            table.add_row(str(idx), role, summary)

        console.print(table)
