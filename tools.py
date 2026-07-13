"""
State-mutating and utility tools for DeepAgent Harness.
Includes virtual filesystem operations, subprocess execution, web searching,
and state management using LangGraph Command updates.
"""
import os
import sys
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

from config import EXPORT_DIR, EXECUTION_TIMEOUT, DANGEROUS_TOOLS


# ── PLANNING TOOLS ───────────────────────────────────────────────────

@tool
def write_todos(todos: list[dict], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Create or replace the task plan. Call this first for any non-trivial task.
    todos: list of {"content": str, "status": "pending"|"in_progress"|"completed"}.
    """
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Plan updated: {len(todos)} todos.", tool_call_id=tool_call_id)],
        }
    )


# ── FILESYSTEM TOOLS ─────────────────────────────────────────────────

@tool
def ls(state: Annotated[dict, InjectedState]) -> str:
    """List every file currently in the virtual filesystem."""
    files = state.get("files", {})
    return "\n".join(files.keys()) if files else "(no files yet)"


@tool
def read_file(path: str, state: Annotated[dict, InjectedState]) -> str:
    """Read the full contents of a file from the virtual filesystem."""
    files = state.get("files", {})
    if path not in files:
        return f"Error: '{path}' not found. Existing files: {list(files.keys())}"
    return files[path]


@tool
def write_file(
    path: str,
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Create a new file or fully overwrite an existing one in the virtual filesystem."""
    files = dict(state.get("files", {}))
    files[path] = content
    return Command(
        update={
            "files": files,
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
    """Replace an exact, unique substring (old_str) with new_str inside an existing virtual file.
    Fails loudly if old_str isn't found exactly once, so re-read the file if unsure.
    """
    files = dict(state.get("files", {}))
    if path not in files:
        return Command(update={"messages": [ToolMessage(f"Error: '{path}' not found in virtual filesystem.", tool_call_id=tool_call_id)]})
    content = files[path]
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
    files[path] = content.replace(old_str, new_str)
    return Command(update={"files": files, "messages": [ToolMessage(f"Edited virtual file {path}", tool_call_id=tool_call_id)]})


@tool
def save_to_disk(
    virtual_path: str,
    real_path: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Export a virtual file to the real local filesystem.
    Requires human confirmation if run in interactive environments.
    """
    files = state.get("files", {})
    if virtual_path not in files:
        return Command(update={"messages": [ToolMessage(f"Error: virtual file '{virtual_path}' not found.", tool_call_id=tool_call_id)]})
    
    content = files[virtual_path]
    try:
        # Resolve real path relative to export directory or absolute
        dest = Path(real_path)
        if not dest.is_absolute():
            dest = EXPORT_DIR / dest
        
        # Ensure parent directories exist
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        
        return Command(
            update={
                "messages": [ToolMessage(f"Successfully saved virtual file '{virtual_path}' to real path '{dest}'", tool_call_id=tool_call_id)]
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [ToolMessage(f"Error saving to disk: {str(e)}", tool_call_id=tool_call_id)]
            }
        )


@tool
def read_from_disk(
    real_path: str,
    virtual_path: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Import a real file from the local filesystem into the virtual filesystem."""
    try:
        if not os.path.exists(real_path):
            return Command(update={"messages": [ToolMessage(f"Error: real path '{real_path}' does not exist.", tool_call_id=tool_call_id)]})
        
        with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        files = dict(state.get("files", {}))
        files[virtual_path] = content
        
        return Command(
            update={
                "files": files,
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
def diff_files(path1: str, path2: str, state: Annotated[dict, InjectedState]) -> str:
    """Show unified diff between two virtual files."""
    files = state.get("files", {})
    if path1 not in files:
        return f"Error: virtual file '{path1}' not found."
    if path2 not in files:
        return f"Error: virtual file '{path2}' not found."
    
    content1 = files[path1].splitlines(keepends=True)
    content2 = files[path2].splitlines(keepends=True)
    
    diff = list(difflib.unified_diff(content1, content2, fromfile=path1, tofile=path2))
    return "".join(diff) if diff else "Files are identical."


# ── CODE EXECUTION TOOLS ─────────────────────────────────────────────

@tool
def execute_python(code: str) -> str:
    """Run a standalone Python snippet in an isolated subprocess and return stdout/stderr.
    Useful for verification.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=EXECUTION_TIMEOUT)
        out, err = result.stdout[-3000:], result.stderr[-2000:]
        return f"STDOUT:\n{out}\n\nSTDERR:\n{err}" if err else f"STDOUT:\n{out}"
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out ({EXECUTION_TIMEOUT}s limit)."
    except Exception as e:
        return f"Error executing Python: {str(e)}"
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


@tool
def shell_command(command: str) -> str:
    """Execute an arbitrary shell command in a subprocess. Requires careful use."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=EXECUTION_TIMEOUT)
        out, err = result.stdout[-3000:], result.stderr[-2000:]
        return f"STDOUT:\n{out}\n\nSTDERR:\n{err}" if err else f"STDOUT:\n{out}"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out ({EXECUTION_TIMEOUT}s limit)."
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def install_package(package: str) -> str:
    """Install a Python package via pip to the current environment."""
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


# ── WEB & SEARCH TOOLS ───────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return a summary of matches."""
    try:
        from duckduckgo_search import DDGS
        results_str = []
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            for i, r in enumerate(results, 1):
                results_str.append(f"{i}. Title: {r.get('title')}\n   URL: {r.get('href')}\n   Snippet: {r.get('body')}\n")
        return "\n".join(results_str) if results_str else "No results found."
    except Exception as e:
        # Fallback if library fails
        return f"Web search error: {str(e)}. Try using python to search or scrape."


@tool
def read_url(url: str) -> str:
    """Fetch text content from a public URL and return it (HTML stripped)."""
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read()
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text_content = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text_content[:5000] + ("\n... [Truncated to 5000 chars]" if len(text_content) > 5000 else "")
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

# To avoid circular import on Path
from pathlib import Path
