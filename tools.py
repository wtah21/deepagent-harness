"""
DeepAgent Harness 的状态修改类与实用工具集。
包含虚拟文件系统操作、子进程执行、网页搜索，
以及基于 LangGraph Command 更新的状态管理工具。
"""
import os
import sys
import shutil
import subprocess
import tempfile
import difflib
from typing import Annotated, Optional
import urllib.request
from bs4 import BeautifulSoup

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from config import EXECUTION_TIMEOUT
from persistence import WorkspaceStore
from state import workspace_of
import docker_runtime


# ── 规划工具 ─────────────────────────────────────────────────────────

@tool
def write_todos(todos: list[dict], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """创建或替换任务计划。任何非平凡任务都应先调用本工具。
    todos: 由 {"content": str, "status": "pending"|"in_progress"|"completed"} 组成的列表。
    """
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Plan updated: {len(todos)} todos.", tool_call_id=tool_call_id)],
        }
    )


# ── 文件系统工具 ─────────────────────────────────────────────────────

@tool
def ls(
    state: Annotated[dict, InjectedState],
    offset: int = 0,
    limit: int = 100,
) -> str:
    """分页列出虚拟文件系统中的文件。默认返回第一页（最多 100 个）。
    文件很多时先用默认页，没找到目标再传 offset 翻页，不要一次拉取全部。
    """
    files = sorted(state.get("files", {}).keys())
    total = len(files)
    if total == 0:
        return "(no files yet)"

    limit = max(1, min(int(limit), 500))
    offset = max(0, min(int(offset), max(0, total - 1)))
    page = files[offset:offset + limit]
    header = f"[files {offset + 1}-{offset + len(page)} / {total}]"
    remaining = total - (offset + len(page))
    footer = f"\n[还有 {remaining} 个文件，下一页请传 offset={offset + len(page)}]" if remaining > 0 else ""
    return header + "\n" + "\n".join(page) + footer


@tool
def read_file(
    path: str,
    state: Annotated[dict, InjectedState],
    start_line: int = 1,
    max_lines: int = 0,
) -> str:
    """读取虚拟文件内容，支持按行分页。默认读取全文。
    文件很大时先传 max_lines（例如 200）读取第一段，需要时再用 start_line 继续。
    """
    content = WorkspaceStore.read(workspace_of(state), path)
    if content is None:
        files = state.get("files", {})
        return f"Error: '{path}' not found. Existing files: {list(files.keys())}"

    if max_lines and int(max_lines) > 0:
        lines = content.splitlines()
        total = len(lines)
        start = max(1, int(start_line))
        begin = min(start - 1, total)
        chunk = lines[begin:begin + int(max_lines)]
        end = begin + len(chunk)
        header = f"[{path} 第 {begin + 1}-{end} 行 / 共 {total} 行]"
        footer = f"\n[还有 {total - end} 行，继续读取请传 start_line={end + 1}]" if end < total else ""
        return header + "\n" + "\n".join(chunk) + footer
    return content


@tool
def write_file(
    path: str,
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """在虚拟文件系统中新建文件，或完全覆盖已有文件。"""
    # 子智能体可能给容器内绝对路径（/app/workspace/x.py、/reports/x.md），统一成工作区相对路径
    normalized = WorkspaceStore.normalize_path(path)
    if not normalized:
        # 返回可读的错误而不是抛异常：一个非法路径不应让整条流水线崩溃
        return Command(update={"messages": [ToolMessage(
            f"Error: 非法路径 '{path}'，请使用工作区内的相对路径（例如 reports/notes.md）。",
            tool_call_id=tool_call_id,
        )]})
    path = normalized
    entry = WorkspaceStore.write(workspace_of(state), path, content)
    # 只把索引写回 state，文件内容留在工作区表
    return Command(
        update={
            "files": {path: entry},
            "messages": [ToolMessage(f"Wrote {len(content)} chars to {path} in virtual filesystem.", tool_call_id=tool_call_id)],
        }
    )


@tool
def edit_file(
    path: str,
    old_str: str,
    new_str: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """在已有的虚拟文件中，把一段精确且唯一的子串（old_str）替换为 new_str。
    如果 old_str 没有恰好出现一次，会直接报错，因此不确定时应先重新读取文件。
    """
    workspace_id = workspace_of(state)
    # 子智能体可能给容器内绝对路径（/app/workspace/x.py、/reports/x.md），统一成工作区相对路径
    normalized = WorkspaceStore.normalize_path(path)
    if not normalized:
        return Command(update={"messages": [ToolMessage(
            f"Error: 非法路径 '{path}'，请使用工作区内的相对路径（例如 reports/notes.md）。",
            tool_call_id=tool_call_id,
        )]})
    path = normalized
    content = WorkspaceStore.read(workspace_id, path)
    if content is None:
        return Command(update={"messages": [ToolMessage(f"Error: '{path}' not found in virtual filesystem.", tool_call_id=tool_call_id)]})

    count = content.count(old_str)
    if count != 1:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: old_str appears {count} times in {path}, must be exactly 1. "
                        "Include more surrounding context to make it unique.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    entry = WorkspaceStore.write(workspace_id, path, content.replace(old_str, new_str))
    return Command(
        update={
            "files": {path: entry},
            "messages": [ToolMessage(f"Edited virtual file {path}", tool_call_id=tool_call_id)],
        }
    )


@tool
def read_from_disk(
    real_path: str,
    virtual_path: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """从本地文件系统导入一个真实文件到虚拟文件系统。"""
    try:
        if not os.path.exists(real_path):
            return Command(update={"messages": [ToolMessage(f"Error: real path '{real_path}' does not exist.", tool_call_id=tool_call_id)]})
        
        with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # 虚拟路径同样归一化，避免索引里出现容器内绝对路径
        virtual_path = WorkspaceStore.normalize_path(virtual_path) or virtual_path
        entry = WorkspaceStore.write(workspace_of(state), virtual_path, content)
        return Command(
            update={
                "files": {virtual_path: entry},
                "messages": [ToolMessage(f"Imported '{real_path}' into virtual file '{virtual_path}'", tool_call_id=tool_call_id)]
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [ToolMessage(f"Error importing file: {str(e)}", tool_call_id=tool_call_id)]
            }
        )


@tool
def diff_files(
    path1: str,
    path2: str,
    state: Annotated[dict, InjectedState],
    max_lines: int = 400,
) -> str:
    """展示两个虚拟文件之间的统一 diff（unified diff）。默认最多返回 400 行，
    diff 很大时调大 max_lines 前请先确认是否真的需要全部内容。
    """
    workspace_id = workspace_of(state)
    content1 = WorkspaceStore.read(workspace_id, path1)
    content2 = WorkspaceStore.read(workspace_id, path2)
    if content1 is None:
        return f"Error: virtual file '{path1}' not found."
    if content2 is None:
        return f"Error: virtual file '{path2}' not found."

    diff = list(difflib.unified_diff(
        content1.splitlines(keepends=True),
        content2.splitlines(keepends=True),
        fromfile=path1, tofile=path2,
    ))
    if not diff:
        return "Files are identical."

    limit = max(1, int(max_lines))
    if len(diff) > limit:
        return "".join(diff[:limit]) + f"\n... [diff 共 {len(diff)} 行，已截断到前 {limit} 行]"
    return "".join(diff)


# ── 代码执行工具（Docker 容器隔离执行，本地临时沙盒仅作回退）──────────

_SANDBOX_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", "node_modules", ".venv", ".idea"}
_SANDBOX_MAX_FILE_SIZE = 1_000_000   # 单文件回收上限 1MB
_SANDBOX_MAX_FILES = 50              # 单次回收文件数上限


def _seed_sandbox(workdir: str, seeded: dict) -> None:
    """把虚拟文件系统的文件预置进本地回退沙盒目录（容器路径不需要预置）。"""
    for vpath, content in seeded.items():
        dest = Path(workdir).joinpath(*vpath.split("/"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_text(content, encoding="utf-8")
        except OSError:
            continue


def _collect_sandbox(workdir: str, workspace_id: str, seeded: dict):
    """扫描本地回退沙盒目录，把新增/修改的文件回收进虚拟文件系统。"""
    entries: dict = {}
    collected: list[str] = []
    skipped: list[str] = []
    for root, dirs, names in os.walk(workdir):
        dirs[:] = [d for d in dirs if d not in _SANDBOX_SKIP_DIRS]
        for name in names:
            if name.endswith((".pyc", ".pyo")):
                continue
            full = Path(root) / name
            rel = full.relative_to(workdir).as_posix()
            try:
                if full.stat().st_size > _SANDBOX_MAX_FILE_SIZE:
                    skipped.append(f"{rel} (>1MB)")
                    continue
                content = full.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                skipped.append(f"{rel} (non-text)")
                continue
            if rel in seeded and seeded[rel] == content:
                continue  # 内容未变的预置文件，无需回收
            entries[rel] = WorkspaceStore.write(workspace_id, rel, content)
            collected.append(rel)
            if len(collected) >= _SANDBOX_MAX_FILES:
                skipped.append("... (更多文件超出单次回收上限)")
                return entries, collected, skipped
    return entries, collected, skipped


def _run_local_sandboxed(
    state: dict,
    tool_call_id: str,
    run_fn,
) -> Command:
    """本地回退沙盒执行器：隔离临时目录 + 虚拟文件预置 + 产物回收。

    仅在 Docker 不可用时使用；正常路径是容器内直接读写挂载的工作区目录。
    """
    workspace_id = workspace_of(state)
    workdir = tempfile.mkdtemp(prefix="harness_sandbox_")
    try:
        seeded = WorkspaceStore.read_all(workspace_id)
        _seed_sandbox(workdir, seeded)
        output = run_fn(workdir)
        entries, collected, skipped = _collect_sandbox(workdir, workspace_id, seeded)
        if collected:
            output += "\n\n[Collected into virtual filesystem]: " + ", ".join(collected)
        if skipped:
            output += "\n[Skipped]: " + ", ".join(skipped)
        return Command(
            update={
                "files": entries,
                "messages": [ToolMessage(output, tool_call_id=tool_call_id)],
            }
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_execution_tool(
    state: dict,
    tool_call_id: str,
    payload: str,
    is_shell: bool,
) -> Command:
    """代码/命令执行的统一入口。

    首选 Docker 容器（1 CPU / 1GB 内存，工作区挂载为 /app/workspace）：
    执行前快照工作区目录，执行后 diff 出新增/变更文件并登记元数据；
    Docker 不可用时回退到本地临时沙盒（旧的 tempfile 隔离 + 产物回收）。
    """
    workspace_id = workspace_of(state)

    if docker_runtime.is_docker_available():
        before = WorkspaceStore.scan_dir(workspace_id)
        try:
            output = docker_runtime.exec_in_workspace(
                workspace_id, payload, is_shell=is_shell, timeout=EXECUTION_TIMEOUT,
            )
        except docker_runtime.DockerError as e:
            output = None  # 容器故障：回退本地沙盒
        else:
            entries, changed = WorkspaceStore.sync_dir(workspace_id, before)
            if changed:
                output += "\n\n[Collected into workspace]: " + ", ".join(changed)
            return Command(
                update={
                    "files": entries,
                    "messages": [ToolMessage(output, tool_call_id=tool_call_id)],
                }
            )

    # ── 回退路径：本地临时沙盒 ──────────────────────────────────────
    def run(workdir: str) -> str:
        try:
            result = subprocess.run(
                command_payload,
                capture_output=True, text=True, timeout=EXECUTION_TIMEOUT, cwd=workdir,
            )
            out, err = result.stdout[-3000:], result.stderr[-2000:]
            return f"STDOUT:\n{out}\n\nSTDERR:\n{err}" if err else f"STDOUT:\n{out}"
        except subprocess.TimeoutExpired:
            return f"Error: execution timed out ({EXECUTION_TIMEOUT}s limit)."
        except Exception as e:
            return f"Error executing: {str(e)}"

    command_payload = payload if is_shell else [sys.executable, "-c", payload]
    return _run_local_sandboxed(state, tool_call_id, run)


@tool
def execute_python(
    code: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """在 Docker 隔离容器中运行一段独立的 Python 代码，并返回 stdout/stderr。
    每个项目的容器限制为 1 CPU / 1GB 内存，工作区目录挂载为容器内的 /app/workspace：
    代码可直接读取之前生成的文件；新产生的文件会自动登记进虚拟文件系统索引。
    容器不可用时自动回退到临时沙盒目录执行。常用于验证代码是否正确。
    """
    return _run_execution_tool(state, tool_call_id, code, is_shell=False)


@tool
def shell_command(
    command: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """在 Docker 隔离容器中执行 shell 命令，需要谨慎使用。
    每个项目的容器限制为 1 CPU / 1GB 内存，工作区目录挂载为容器内的 /app/workspace：
    命令产生的新文件会自动登记进虚拟文件系统索引。
    容器不可用时自动回退到临时沙盒目录执行。
    """
    return _run_execution_tool(state, tool_call_id, command, is_shell=True)


@tool
def install_package(package: str) -> str:
    """通过 pip 将某个 Python 包安装到当前环境。"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return f"Successfully installed {package}."
        else:
            return f"Failed to install {package}. Error:\n{result.stderr}"
    except Exception as e:
        return f"Error installing package: {str(e)}"


# ── 网络与搜索工具 ───────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """使用 DuckDuckGo 进行网页搜索，返回匹配结果的摘要。默认返回前 5 条，
    需要更多时最多调到 20 条，避免一次拉取过多结果占用上下文。
    """
    try:
        from duckduckgo_search import DDGS
        limit = max(1, min(int(max_results), 20))
        results_str = []
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=limit))
            for i, r in enumerate(results, 1):
                results_str.append(f"{i}. Title: {r.get('title')}\n   URL: {r.get('href')}\n   Snippet: {r.get('body')}\n")
        return "\n".join(results_str) if results_str else "No results found."
    except Exception as e:
        # 库调用失败时的兜底处理
        return f"Web search error: {str(e)}. Try using python to search or scrape."


@tool
def read_url(url: str, max_chars: int = 5000) -> str:
    """抓取某个公开 URL 的文本内容并返回（已去除 HTML 标签）。
    默认最多返回 5000 字符；内容很长时先读默认长度，需要更多再调大 max_chars。
    """
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除 script 和 style 元素
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text()
        
        # 按行拆分，并去掉每行首尾的空白字符
        lines = (line.strip() for line in text.splitlines())
        # 把多行标题拆分为逐行内容
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # 丢弃空行
        text_content = '\n'.join(chunk for chunk in chunks if chunk)
        
        limit = max(500, min(int(max_chars), 20000))
        return text_content[:limit] + (f"\n... [Truncated to {limit} chars]" if len(text_content) > limit else "")
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

# 为避免 Path 的循环导入问题，将其放在文件末尾导入
from pathlib import Path
