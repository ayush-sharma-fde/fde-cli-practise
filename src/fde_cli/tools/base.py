from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class Tool:
    """Represents a callable tool with an Anthropic-compatible JSON schema."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[..., Any]

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Format the tool schema according to Anthropic Claude function calling specs."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def execute(self, **kwargs) -> Any:
        """Execute the underlying function with supplied arguments."""
        return self.func(**kwargs)
