import sys
from pathlib import Path

# Ensure src directory is in Python path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fde_cli.cli import app

if __name__ == "__main__":
    app()