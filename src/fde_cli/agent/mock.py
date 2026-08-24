import json
import re
import uuid
from typing import Any, Dict, List, Optional
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage


class MockClaudeEngine:
    """
    Simulates Anthropic Claude API responses locally without requiring an API key.
    Produces authentic `anthropic.types.Message` objects to exercise the exact
    same tool-calling while-loop lifecycle as the live API.
    """

    def __init__(self, model_name: str = "mock-claude-3-5-sonnet") -> None:
        self.model_name = model_name

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Message:
        """
        Simulate Claude's reasoning step based on conversation history and registered tools.
        """
        last_msg = messages[-1]
        last_content = last_msg.get("content", "")

        # ---------------------------------------------------------------------
        # Step A: Check if the last message was a tool result
        # ---------------------------------------------------------------------
        if isinstance(last_content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in last_content
        ):
            # Extract tool results and formulate final answer
            tool_results = [b for b in last_content if isinstance(b, dict) and b.get("type") == "tool_result"]
            summary_parts = []
            for tr in tool_results:
                content_str = tr.get("content", "")
                try:
                    data = json.loads(content_str)
                    if "result" in data:
                        summary_parts.append(f"The calculation result is **{data['result']}**.")
                    elif "data" in data:
                        rows = data["data"]
                        summary_parts.append(
                            f"Successfully queried the database. Found **{len(rows)}** record(s):\n"
                            f"```json\n{json.dumps(rows, indent=2)}\n```"
                        )
                    elif "os" in data:
                        summary_parts.append(
                            f"System information: Running on **{data['os']} {data['os_release']}** "
                            f"({data['architecture']}) with Python **{data['python_version']}** at `{data['current_time']}`."
                        )
                    else:
                        summary_parts.append(f"Tool executed successfully:\n```json\n{json.dumps(data, indent=2)}\n```")
                except Exception:
                    summary_parts.append(f"Tool output received: {content_str}")

            final_text = (
                "Based on the tool execution, here are the findings:\n\n"
                + "\n\n".join(summary_parts)
                + "\n\n*(Execution simulated locally via Mock Claude Engine. Ready for live API key)*"
            )

            return Message(
                id=f"msg_mock_{uuid.uuid4().hex[:8]}",
                content=[TextBlock(text=final_text, type="text")],
                model=self.model_name,
                role="assistant",
                stop_reason="end_turn",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=150, output_tokens=85),
            )

        # ---------------------------------------------------------------------
        # Step B: Parse User Prompt and decide if a tool should be called
        # ---------------------------------------------------------------------
        prompt_text = str(last_content).lower()

        # 1. Math / Calculation pattern
        math_match = re.search(r"(\d+\s*[\+\-\*\/\^%]\s*\d+(?:\s*[\+\-\*\/\^%]\s*\d+)*)", prompt_text)
        if ("calculate" in prompt_text or "what is" in prompt_text or math_match) and any(
            t.get("name") == "calculator" for t in (tools or [])
        ):
            expr = math_match.group(1) if math_match else "42 * 10"
            tool_id = f"toolu_mock_{uuid.uuid4().hex[:8]}"
            return Message(
                id=f"msg_mock_{uuid.uuid4().hex[:8]}",
                content=[
                    TextBlock(text=f"I need to calculate the expression `{expr}` using the calculator tool.", type="text"),
                    ToolUseBlock(
                        id=tool_id,
                        input={"expression": expr},
                        name="calculator",
                        type="tool_use",
                    ),
                ],
                model=self.model_name,
                role="assistant",
                stop_reason="tool_use",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=85, output_tokens=42),
            )

        # 2. SQL / Database pattern
        if any(w in prompt_text for w in ["sql", "database", "customers", "deployments", "table", "query", "spend"]) and any(
            t.get("name") == "run_sql_query" for t in (tools or [])
        ):
            if "deployment" in prompt_text or "service" in prompt_text:
                sql = "SELECT service, environment, version, status FROM deployments WHERE status != 'healthy';"
            elif "enterprise" in prompt_text or "spend" in prompt_text:
                sql = "SELECT name, tier, monthly_spend FROM customers WHERE tier = 'Enterprise' ORDER BY monthly_spend DESC;"
            else:
                sql = "SELECT * FROM customers LIMIT 5;"

            tool_id = f"toolu_mock_{uuid.uuid4().hex[:8]}"
            return Message(
                id=f"msg_mock_{uuid.uuid4().hex[:8]}",
                content=[
                    TextBlock(text=f"Querying the internal database with SQL: `{sql}`", type="text"),
                    ToolUseBlock(
                        id=tool_id,
                        input={"query": sql},
                        name="run_sql_query",
                        type="tool_use",
                    ),
                ],
                model=self.model_name,
                role="assistant",
                stop_reason="tool_use",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=110, output_tokens=55),
            )

        # 3. System Info pattern
        if any(w in prompt_text for w in ["system", "os", "platform", "version", "time", "specs"]) and any(
            t.get("name") == "get_system_info" for t in (tools or [])
        ):
            tool_id = f"toolu_mock_{uuid.uuid4().hex[:8]}"
            return Message(
                id=f"msg_mock_{uuid.uuid4().hex[:8]}",
                content=[
                    TextBlock(text="Let me inspect the host system environment.", type="text"),
                    ToolUseBlock(
                        id=tool_id,
                        input={},
                        name="get_system_info",
                        type="tool_use",
                    ),
                ],
                model=self.model_name,
                role="assistant",
                stop_reason="tool_use",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=90, output_tokens=30),
            )

        # ---------------------------------------------------------------------
        # Step C: Conversational response (no tool needed)
        # ---------------------------------------------------------------------
        response_text = (
            f"Hello! I am your FDE AI Agent running in local **Mock Mode**.\n\n"
            f"I have **{len(tools or [])}** tools loaded in my registry:\n"
            + "\n".join([f"- `{t.get('name')}`: {t.get('description', '')}" for t in (tools or [])])
            + "\n\nTry asking me:\n"
            "- *'Calculate 256 * 16 + 42'*\n"
            "- *'Query customers with Enterprise tier'*\n"
            "- *'What are the unhealthy deployments?'*\n"
            "- *'What is the current system info and time?'*\n\n"
            "*(Once your Anthropic API key is added to `.env`, this harness will query Claude 3.5/3.7 Sonnet directly!)*"
        )

        return Message(
            id=f"msg_mock_{uuid.uuid4().hex[:8]}",
            content=[TextBlock(text=response_text, type="text")],
            model=self.model_name,
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
            type="message",
            usage=Usage(input_tokens=75, output_tokens=120),
        )
