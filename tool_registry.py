"""
DeepAgent Harness 所有可用工具的中央注册表。
每个工具都带有元数据：名称、描述、分类、风险等级。
AgentFactory 借助它把工具名字符串解析为实际的工具函数。
"""
from typing import Callable, Any

from config import TOOL_CATEGORIES
from tool_output_guard import wrap_tool


class ToolMeta:
    """工具函数的元数据包装类。"""

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
        self.risk_level = risk_level  # 可选值：safe、moderate、dangerous

    def to_dict(self) -> dict[str, str]:
        """将元数据序列化，供 UI 展示。"""
        cat_info = TOOL_CATEGORIES.get(self.category, {})
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "category_label": cat_info.get("label", self.category),
            "category_icon": cat_info.get("icon", "🔧"),
            "risk_level": self.risk_level,
        }


# ── 工具注册表 ───────────────────────────────────────────────────────────
# 在导入时由下方的 _register_all_tools() 填充。
# 映射关系为：工具名称 → ToolMeta。
TOOL_REGISTRY: dict[str, ToolMeta] = {}


def register_tool(
    func: Callable,
    name: str,
    description: str,
    category: str,
    risk_level: str = "safe",
) -> None:
    """将工具注册到全局注册表中；统一套一层超长输出守卫。"""
    TOOL_REGISTRY[name] = ToolMeta(wrap_tool(func), name, description, category, risk_level)


def unregister_tool(name: str) -> None:
    """从注册表中移除一个工具（MCP 配置变更时用于清理旧工具）。"""
    TOOL_REGISTRY.pop(name, None)


def get_tool(name: str) -> Callable | None:
    """按名称获取工具函数。"""
    meta = TOOL_REGISTRY.get(name)
    return meta.func if meta else None


def get_tools_by_names(names: list[str]) -> list[Callable]:
    """把一组工具名称解析为对应的函数，未知名称会被跳过。"""
    tools = []
    for name in names:
        meta = TOOL_REGISTRY.get(name)
        if meta:
            tools.append(meta.func)
    return tools


def get_tools_by_category(category: str) -> list[ToolMeta]:
    """获取某个分类下的所有工具。"""
    return [m for m in TOOL_REGISTRY.values() if m.category == category]


def list_all_tools() -> list[dict[str, str]]:
    """以字典列表的形式列出所有已注册工具，供 UI 展示。"""
    return [m.to_dict() for m in TOOL_REGISTRY.values()]


def _register_all_tools() -> None:
    """导入并注册每一个工具。模块加载时仅调用一次。"""
    from tools import (
        write_todos, ls, read_file, write_file, edit_file,
        execute_python, read_from_disk,
        web_search, read_url, shell_command, diff_files,
        install_package,
    )

    # ── 文件系统工具 ─────────────────────────────────────────────────
    register_tool(ls, "ls", "List all files in the virtual filesystem", "filesystem")
    register_tool(read_file, "read_file", "Read a file from the virtual filesystem", "filesystem")
    register_tool(write_file, "write_file", "Create or overwrite a virtual file", "filesystem")
    register_tool(edit_file, "edit_file", "Replace a substring in an existing virtual file", "filesystem")
    # save_to_disk 已移除：agent 不得直写本地文件系统，导出一律走 UI「导入到本地」按钮
    register_tool(read_from_disk, "read_from_disk", "Import a real file into the virtual filesystem", "filesystem", "moderate")
    register_tool(diff_files, "diff_files", "Show unified diff between two virtual files", "filesystem")

    # ── 代码执行工具 ─────────────────────────────────────────────────
    register_tool(execute_python, "execute_python", "Run Python code in an isolated subprocess", "execution", "moderate")
    register_tool(shell_command, "shell_command", "Run a shell command (sandboxed)", "execution", "dangerous")
    register_tool(install_package, "install_package", "Install a Python package via pip", "execution", "dangerous")

    # ── 网络工具 ─────────────────────────────────────────────────────
    register_tool(web_search, "web_search", "Search the web via DuckDuckGo (free, no API key)", "web")
    register_tool(read_url, "read_url", "Fetch and extract text content from a URL", "web")

    # ── 规划工具 ─────────────────────────────────────────────────────
    register_tool(write_todos, "write_todos", "Create or update the task plan", "planning")

    # ── 委派工具（由 subagents.py 单独注册）──────────────────────────
    # `task` 工具会在本模块加载完成后由 subagents.py 添加，
    # 目的是避免循环导入。

    # ── 外部 MCP 工具：读取配置并连接服务器后注册 ───────────────────
    _register_mcp_tools()

    # ── 外部 A2A Agent：按缓存的 Agent Card 注册为工具 ──────────────
    _register_a2a_agents()


def _register_mcp_tools() -> None:
    """连接配置中的 MCP 服务器并注册其工具；失败只记录日志，不影响内置工具。"""
    from config import debug_log

    try:
        from mcp_client import load_and_register

        report = load_and_register()
        failed = {name: info.get("error") for name, info in report.items() if not info.get("ok")}
        if failed:
            debug_log(f"[tool-registry] MCP 部分服务器连接失败：{failed}")
    except Exception as exc:  # 防止 MCP 依赖缺失拖垮整个应用
        debug_log(f"[tool-registry] MCP 初始化失败：{type(exc).__name__}: {exc}")


def _register_a2a_agents() -> None:
    """注册配置中的远端 A2A Agent；失败只记录日志，不影响其它工具。"""
    from config import debug_log

    try:
        from a2a_client import reload_agents

        report = reload_agents()
        failed = {name: info.get("error") for name, info in report.items() if not info.get("ok")}
        if failed:
            debug_log(f"[tool-registry] A2A 部分 Agent 注册失败：{failed}")
    except Exception as exc:
        debug_log(f"[tool-registry] A2A 初始化失败：{type(exc).__name__}: {exc}")


# 导入时自动注册
_register_all_tools()
