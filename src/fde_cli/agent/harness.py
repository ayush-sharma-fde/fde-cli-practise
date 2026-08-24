import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import anthropic
from anthropic.types import TextBlock, ToolUseBlock

from ..config import Settings, settings
from ..tools.registry import ToolRegistry, default_registry
from ..ui.console import AgentUI, console
from .mock import MockClaudeEngine

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Represents the final outcome of an agent invocation."""
    final_text: str
    turns_taken: int
    total_input_tokens: int
    total_output_tokens: int
    messages: List[Dict[str, Any]]
    mode: str


class AgentHarness:
    """
    The Core AI Agent Harness.
    Executes a raw while-loop ReAct execution cycle with Anthropic Claude
    (or local Mock Engine when running in keyless mode).
    """

    def __init__(
        self,
        config: Optional[Settings] = None,
        registry: Optional[ToolRegistry] = None,
        force_mock: bool = False,
    ) -> None:
        self.config = config or settings
        self.registry = registry or default_registry
        self.force_mock = force_mock

        # Determine runtime mode (Live Anthropic API vs Mock Engine)
        if not self.force_mock and self.config.has_api_key:
            self.mode = "live"
            self.client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            self.mock_engine = None
        else:
            self.mode = "mock"
            self.client = None
            self.mock_engine = MockClaudeEngine(model_name=self.config.model)

    def _call_model(self, messages: List[Dict[str, Any]]) -> Any:
        """Dispatch message history to Anthropic API or Mock Engine."""
        tools_schema = self.registry.get_schemas()

        if self.mode == "live" and self.client:
            return self.client.messages.create(
                model=self.config.model,
                max_tokens=4096,
                system=self.config.system_prompt,
                messages=messages,
                tools=tools_schema if tools_schema else None,
            )
        else:
            return self.mock_engine.create_message(
                messages=messages,
                tools=tools_schema,
                system=self.config.system_prompt,
            )

    def run(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        debug: bool = False,
    ) -> AgentResponse:
        """
        Execute the raw Agent ReAct while-loop.
        
        Lifecycle:
        1. Append User prompt to conversation history.
        2. Enter while loop (guardrailed by max_turns).
        3. Send messages to Model.
        4. If stop_reason == 'end_turn': Model is done. Extract text and exit loop.
        5. If stop_reason == 'tool_use':
           a. Append assistant response (with tool_use blocks) to history.
           b. Execute each tool call locally via the ToolRegistry.
           c. Package outputs into 'tool_result' content blocks.
           d. Append user response with 'tool_result' blocks to history.
           e. Repeat loop.
        """
        messages: List[Dict[str, Any]] = list(conversation_history or [])
        messages.append({"role": "user", "content": prompt})

        AgentUI.print_user_prompt(prompt)

        turn = 0
        total_input_tokens = 0
        total_output_tokens = 0
        final_text = ""

        while turn < self.config.max_turns:
            turn += 1
            if debug:
                AgentUI.print_turn_header(turn, self.config.max_turns)

            # -----------------------------------------------------------------
            # 1. Call the LLM
            # -----------------------------------------------------------------
            try:
                response = self._call_model(messages)
            except Exception as e:
                console.print(f"[error]API Call Error:[/error] {str(e)}")
                return AgentResponse(
                    final_text=f"Error during model call: {str(e)}",
                    turns_taken=turn,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    messages=messages,
                    mode=self.mode,
                )

            # Track token usage
            if hasattr(response, "usage") and response.usage:
                total_input_tokens += getattr(response.usage, "input_tokens", 0)
                total_output_tokens += getattr(response.usage, "output_tokens", 0)

            # -----------------------------------------------------------------
            # 2. Inspect Assistant Response Blocks
            # -----------------------------------------------------------------
            assistant_content_blocks = []
            tool_use_blocks: List[ToolUseBlock] = []
            text_parts: List[str] = []

            for block in response.content:
                if isinstance(block, TextBlock) or (hasattr(block, "type") and block.type == "text"):
                    text_content = block.text if hasattr(block, "text") else str(block)
                    text_parts.append(text_content)
                    assistant_content_blocks.append({"type": "text", "text": text_content})
                elif isinstance(block, ToolUseBlock) or (hasattr(block, "type") and block.type == "tool_use"):
                    tool_use_blocks.append(block)
                    assistant_content_blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            # Print partial thoughts/text if any before tool call
            if text_parts and tool_use_blocks:
                for text in text_parts:
                    console.print(f"[dim italic]Thinking: {text}[/dim italic]")

            # -----------------------------------------------------------------
            # 3. Check Termination Condition: stop_reason == "end_turn"
            # -----------------------------------------------------------------
            if response.stop_reason == "end_turn" or not tool_use_blocks:
                final_text = "\n\n".join(text_parts) if text_parts else "No response generated."
                # Append assistant's final response to history
                messages.append({"role": "assistant", "content": assistant_content_blocks})
                AgentUI.print_agent_response(final_text)
                break

            # -----------------------------------------------------------------
            # 4. Handle Tool Execution: stop_reason == "tool_use"
            # -----------------------------------------------------------------
            # First, append the assistant's request (with its tool_use blocks) to history
            messages.append({"role": "assistant", "content": assistant_content_blocks})

            tool_result_blocks = []
            for tool_call in tool_use_blocks:
                # Render tool call in terminal
                AgentUI.print_tool_call(tool_call.name, tool_call.id, tool_call.input)

                # Execute tool locally via registry
                result_str = self.registry.dispatch(tool_call.name, tool_call.input)

                # Render tool result in terminal
                AgentUI.print_tool_result(tool_call.name, tool_call.id, result_str)

                # Format Anthropic tool_result block
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": result_str,
                })

            # Append the user message containing tool results back to conversation history
            messages.append({
                "role": "user",
                "content": tool_result_blocks,
            })

            # The while-loop continues to the next turn!

        else:
            console.print(f"[warning]Reached max turns ({self.config.max_turns}) limit.[/warning]")

        if debug:
            AgentUI.print_debug_messages(messages)

        return AgentResponse(
            final_text=final_text,
            turns_taken=turn,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            messages=messages,
            mode=self.mode,
        )
