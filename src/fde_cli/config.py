import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


@dataclass
class Settings:
    """Application configuration and runtime settings."""
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()
    max_turns: int = int(os.getenv("MAX_TURNS", "10"))
    system_prompt: str = (
        "You are an expert Forward Deployed Engineer (FDE) AI Assistant. "
        "You have access to specialized tools to inspect systems, compute calculations, "
        "and query databases. Use tools when needed to give accurate and complete answers."
    )

    @property
    def has_api_key(self) -> bool:
        """Check if a valid non-empty Anthropic API key is provided."""
        return bool(
            self.anthropic_api_key
            and not self.anthropic_api_key.startswith("your_")
            and len(self.anthropic_api_key) > 10
        )


settings = Settings()
