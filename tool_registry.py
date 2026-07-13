"""
Central registry of all available tools for DeepAgent Harness.
Each tool has metadata: name, description, category, risk_level.
The AgentFactory uses this to resolve tool name strings → actual tool functions.
"""
from typing import Callable, Any

from config import TOOL_CATEGORIES


class ToolMeta:
    """Metadata wrapper for a tool function."""

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        category: str,
        risk_level: str = "safe",
    ):
        self.func = func
        self.name = name
        self.description = description
        self.category = category
        self.risk_level = risk_level  # safe, moderate, dangerous

    def to_dict(self) -> dict[str, str]:
        """Serialize metadata for UI display."""
        cat_info = TOOL_CATEGORIES.get(self.category, {})
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "category_label": cat_info.get("label", self.category),
            "category_icon": cat_info.get("icon", "🔧"),
            "risk_level": self.risk_level,
        }


# ── The Registry ─────────────────────────────────────────────────────────
# Populated at import time by _register_all_tools() below.
# Maps tool name → ToolMeta.
TOOL_REGISTRY: dict[str, ToolMeta] = {}


def register_tool(
    func: Callable,
    name: str,
    description: str,
    category: str,
    risk_level: str = "safe",
) -> None:
    """Register a tool in the global registry."""
    TOOL_REGISTRY[name] = ToolMeta(func, name, description, category, risk_level)


def get_tool(name: str) -> Callable | None:
    """Get a tool function by name."""
    meta = TOOL_REGISTRY.get(name)
    return meta.func if meta else None


def get_tools_by_names(names: list[str]) -> list[Callable]:
    """Resolve a list of tool names to their functions. Skips unknown names."""
    tools = []
    for name in names:
        meta = TOOL_REGISTRY.get(name)
        if meta:
            tools.append(meta.func)
    return tools


def get_tools_by_category(category: str) -> list[ToolMeta]:
    """Get all tools in a category."""
    return [m for m in TOOL_REGISTRY.values() if m.category == category]


def list_all_tools() -> list[dict[str, str]]:
    """List all registered tools as dicts for UI display."""
    return [m.to_dict() for m in TOOL_REGISTRY.values()]


def _register_all_tools() -> None:
    """Import and register every tool. Called once at module load."""
    from tools import (
        write_todos, ls, read_file, write_file, edit_file,
        execute_python, save_to_disk, read_from_disk,
        web_search, read_url, shell_command, diff_files,
        install_package,
    )

    # ── Filesystem tools ─────────────────────────────────────────────
    register_tool(ls, "ls", "List all files in the virtual filesystem", "filesystem")
    register_tool(read_file, "read_file", "Read a file from the virtual filesystem", "filesystem")
    register_tool(write_file, "write_file", "Create or overwrite a virtual file", "filesystem")
    register_tool(edit_file, "edit_file", "Replace a substring in an existing virtual file", "filesystem")
    register_tool(save_to_disk, "save_to_disk", "Export a virtual file to the real filesystem", "filesystem", "dangerous")
    register_tool(read_from_disk, "read_from_disk", "Import a real file into the virtual filesystem", "filesystem", "moderate")
    register_tool(diff_files, "diff_files", "Show unified diff between two virtual files", "filesystem")

    # ── Execution tools ──────────────────────────────────────────────
    register_tool(execute_python, "execute_python", "Run Python code in an isolated subprocess", "execution", "moderate")
    register_tool(shell_command, "shell_command", "Run a shell command (sandboxed)", "execution", "dangerous")
    register_tool(install_package, "install_package", "Install a Python package via pip", "execution", "dangerous")

    # ── Web tools ────────────────────────────────────────────────────
    register_tool(web_search, "web_search", "Search the web via DuckDuckGo (free, no API key)", "web")
    register_tool(read_url, "read_url", "Fetch and extract text content from a URL", "web")

    # ── Planning tools ───────────────────────────────────────────────
    register_tool(write_todos, "write_todos", "Create or update the task plan", "planning")

    # ── Delegation tools (registered separately by subagents.py) ─────
    # The `task` tool is added by subagents.py after this module loads,
    # to avoid circular imports.


# Auto-register on import
_register_all_tools()
