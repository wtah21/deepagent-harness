"""
Shared state for the DeepAgent Harness system.
Every node (main agent + all sub-agents) reads/writes this same schema.
Files, todos, and execution log are shared across all agents.
Uses custom reducers to allow concurrent/multiple updates in a single step.
"""
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.managed.is_last_step import RemainingSteps


class Todo(TypedDict):
    """A single planning item with status tracking."""
    content: str
    status: Literal["pending", "in_progress", "completed"]


class AgentConfig(TypedDict):
    """Configuration for a custom agent."""
    name: str
    description: str
    system_prompt: str
    tools: list[str]
    icon: str
    color: str
    model: str
    max_tokens: int
    temperature: float
    is_builtin: bool


class HarnessStage(TypedDict):
    """A single stage in a multi-agent harness pipeline."""
    agent_name: str
    instructions: str
    pass_files: bool


class HarnessConfig(TypedDict):
    """Configuration for a multi-agent workflow harness."""
    name: str
    description: str
    strategy: Literal["sequential", "parallel", "review_loop"]
    stages: list[HarnessStage]
    max_review_rounds: int
    icon: str
    color: str


class ExecutionLogEntry(TypedDict):
    """Audit trail entry for agent actions."""
    timestamp: str
    agent_name: str
    action: str
    summary: str
    status: Literal["success", "error", "pending"]


# ── Reducers for State Channels ──────────────────────────────────────

def merge_files(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """Reducer to merge virtual filesystem updates, preventing InvalidUpdateError."""
    if left is None:
        left = {}
    if right is None:
        return left
    merged = dict(left)
    merged.update(right)
    return merged


def merge_todos(left: list[Todo], right: list[Todo]) -> list[Todo]:
    """Reducer for todos list to allow multiple writes in a single step."""
    if right is not None:
        return right
    return left if left is not None else []


def merge_execution_log(left: list[ExecutionLogEntry], right: list[ExecutionLogEntry]) -> list[ExecutionLogEntry]:
    """Reducer for audit execution log entries to allow multiple updates."""
    if right is not None:
        return right
    return left if left is not None else []


# ── State Schema ──────────────────────────────────────────────────────

class DeepAgentState(TypedDict):
    """The shared state for the entire DeepAgent system."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    todos: Annotated[list[Todo], merge_todos]
    files: Annotated[dict[str, str], merge_files]
    remaining_steps: RemainingSteps
    execution_log: Annotated[list[ExecutionLogEntry], merge_execution_log]