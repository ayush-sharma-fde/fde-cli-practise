import math
import platform
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict
from .registry import ToolRegistry


# -----------------------------------------------------------------------------
# In-Memory SQLite Sample Database for SQL Query Tool Demo
# -----------------------------------------------------------------------------
def get_sample_db() -> sqlite3.Connection:
    """Initialize an in-memory SQLite database with sample FDE tables."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create sample tables: customers, deployments, server_metrics
    cursor.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            tier TEXT NOT NULL,
            monthly_spend REAL NOT NULL,
            status TEXT NOT NULL
        );
    """)
    cursor.executemany("""
        INSERT INTO customers (id, name, tier, monthly_spend, status) VALUES (?, ?, ?, ?, ?)
    """, [
        (1, "Acme Corp", "Enterprise", 12500.0, "active"),
        (2, "Stark Industries", "Enterprise", 45000.0, "active"),
        (3, "Wayne Enterprises", "Enterprise", 38000.0, "active"),
        (4, "Cyberdyne Systems", "Growth", 4200.0, "paused"),
        (5, "Pied Piper", "Startup", 1500.0, "active"),
    ])

    cursor.execute("""
        CREATE TABLE deployments (
            id INTEGER PRIMARY KEY,
            service TEXT NOT NULL,
            environment TEXT NOT NULL,
            version TEXT NOT NULL,
            status TEXT NOT NULL,
            deployed_at TEXT NOT NULL
        );
    """)
    cursor.executemany("""
        INSERT INTO deployments (id, service, environment, version, status, deployed_at) VALUES (?, ?, ?, ?, ?, ?)
    """, [
        (101, "auth-service", "production", "v2.4.1", "healthy", "2026-08-20 14:32:00"),
        (102, "payment-gateway", "production", "v1.9.0", "healthy", "2026-08-19 09:15:00"),
        (103, "agent-harness-api", "staging", "v0.3.0-rc1", "deploying", "2026-08-21 10:00:00"),
        (104, "analytics-worker", "production", "v3.1.2", "degraded", "2026-08-21 08:45:00"),
    ])

    conn.commit()
    return conn


# Shared in-memory DB connection
_DB_CONN = get_sample_db()


# -----------------------------------------------------------------------------
# Tool Functions
# -----------------------------------------------------------------------------
def calculator(expression: str) -> Dict[str, Any]:
    """Safely evaluates a mathematical expression using Python's math library."""
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

    try:
        # Strip dangerous builtins for safety
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": f"Calculation error: {str(e)}"}


def get_system_info() -> Dict[str, Any]:
    """Returns local system runtime information."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": sys.version.split()[0],
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_sql_query(query: str) -> Dict[str, Any]:
    """Executes a read-only SQL query against the system database."""
    query_clean = query.strip()
    if not query_clean.lower().startswith(("select", "pragma", "explain")):
        return {"error": "Only SELECT or PRAGMA read queries are permitted."}

    try:
        cursor = _DB_CONN.cursor()
        cursor.execute(query_clean)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        
        # Format as list of dicts for clean JSON representation
        results = [dict(zip(columns, row)) for row in rows]
        return {
            "query": query_clean,
            "row_count": len(results),
            "columns": columns,
            "data": results,
        }
    except sqlite3.Error as e:
        return {"query": query_clean, "error": f"SQLite Error: {str(e)}"}


# -----------------------------------------------------------------------------
# Registration Helper
# -----------------------------------------------------------------------------
def register_default_tools(registry: ToolRegistry) -> None:
    """Register all built-in system and calculation tools to the given registry."""
    registry.register(
        name="calculator",
        description="Calculate mathematical expressions. Useful for arithmetic, statistics, formulas, and numeric conversions.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression string to evaluate (e.g. '128 * 4' or 'sqrt(144) + 10')",
                }
            },
            "required": ["expression"],
        },
    )(calculator)

    registry.register(
        name="get_system_info",
        description="Retrieve runtime system metrics, OS details, Python version, and current system timestamp.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    )(get_system_info)

    registry.register(
        name="run_sql_query",
        description="Execute a SQL SELECT query against the analytics database. Tables: 'customers' (id, name, tier, monthly_spend, status), 'deployments' (id, service, environment, version, status, deployed_at).",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The SQL query string (e.g., 'SELECT name, monthly_spend FROM customers WHERE tier = \"Enterprise\";')",
                }
            },
            "required": ["query"],
        },
    )(run_sql_query)
