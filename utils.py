from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: str | dict[str, Any] | None


def collect_tools(messages):
    """Extract function tool calls in order, excluding the final output tool."""
    tool_calls = []
    for message in messages:
        for part in message.parts:
            if part.part_kind != "tool-call":
                continue
            if part.tool_name == "final_result":
                continue
            tool_calls.append(ToolCall(part.tool_name, part.args))
    return tool_calls
