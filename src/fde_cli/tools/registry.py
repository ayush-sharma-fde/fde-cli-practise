import json
import logging
from typing import Any, Callable, Dict, List, Optional
from .base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry managing available tools and dynamic dispatching."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a Python function as an agent tool."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool = Tool(
                name=name,
                description=description,
                input_schema=input_schema,
                func=func,
            )
            self._tools[name] = tool
            return func

        return decorator

    def add_tool(self, tool: Tool) -> None:
        """Add an instantiated Tool object to the registry."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Export all registered tools in Anthropic-compatible tool schema format."""
        return [tool.to_anthropic_schema() for tool in self._tools.values()]

    def list_tools(self) -> List[Tool]:
        """Return all registered Tool objects."""
        return list(self._tools.values())

    def dispatch(self, name: str, input_args: Dict[str, Any]) -> str:
        """
        Execute a tool by name with arguments and return a string result.
        Catches exceptions and returns clean error descriptions to the LLM.
        """
        tool = self.get_tool(name)
        if not tool:
            return f"Error: Tool '{name}' is not registered in the agent harness."

        try:
            result = tool.execute(**input_args)
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2)
            return str(result)
        except TypeError as e:
            return f"Argument Error calling tool '{name}': {str(e)}"
        except Exception as e:
            return f"Execution Error in tool '{name}': {str(e)}"


# Global default registry
default_registry = ToolRegistry()
