"""
工具调用的人工审核闸门。

高危工具（代码执行 / shell / 安装依赖 / 读取工作区外文件）在执行前登记一条审核请求，
执行线程阻塞等待；Streamlit 侧轮询队列渲染「允许 / 拒绝」面板，用户决定后放行。

不依赖 LangGraph 的 interrupt：主智能体、harness 流水线与并行协程都走同一套闸门，
子图没有检查点、子任务不在同一线程时同样可用。
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# 审核等待上限（秒）：超时按拒绝处理，避免无人值守时工具线程被永久卡住
APPROVAL_TIMEOUT = 300

# 需要人工审核的工具：代码执行、shell、装包、读取工作区外文件
APPROVAL_TOOLS = {"execute_python", "shell_command", "install_package", "read_from_disk"}

_IMPACT = {
    "execute_python": "将在隔离沙箱中执行 Python 代码，可能读写当前工作区文件、启动子进程（容器已断网）。",
    "shell_command": "将执行系统命令，可能修改或删除工作区文件、启动子进程（容器已断网）。",
    "install_package": "将在宿主机 Python 环境中安装依赖包，会修改当前运行环境。",
    "read_from_disk": "将读取当前工作区之外的本地文件，可能触及敏感内容。",
}


@dataclass
class PendingApproval:
    """一条待用户确认的高危工具调用。"""

    id: str
    tool_name: str
    arguments: str
    impact: str
    status: str = "pending"  # pending / approved / rejected
    created_at: float = field(default_factory=time.time)
    event: threading.Event = field(default_factory=threading.Event)


_lock = threading.Lock()
_pending: dict[str, PendingApproval] = {}
# 本会话内已勾选「不再询问」的工具名：命中后直接放行。
# 仅存在内存里，进程重启即失效；桌面单用户场景下无需按会话隔离。
_session_allowed: set[str] = set()


def needs_approval(tool_name: str) -> bool:
    """工具是否需要审核；删除类工具按名称兜底识别。"""
    name = (tool_name or "").lower()
    return name in APPROVAL_TOOLS or "delete" in name or "remove" in name


def _impact_of(tool_name: str) -> str:
    name = (tool_name or "").lower()
    if name in _IMPACT:
        return _IMPACT[name]
    if "delete" in name or "remove" in name:
        return "将删除文件或数据，删除后不可恢复。"
    return "该操作可能修改运行环境或文件。"


def _preview(arguments: Any, limit: int = 600) -> str:
    """把调用参数整理成可读预览，过滤内部注入的状态对象。"""
    if isinstance(arguments, dict):
        parts = []
        for key, value in arguments.items():
            if key in ("state", "tool_call_id"):
                continue
            text = str(value)
            if len(text) > 400:
                text = text[:400] + "…"
            parts.append(f"{key}: {text}")
        text = "\n".join(parts)
    else:
        text = str(arguments)
    return text[:limit]


def request(tool_name: str, arguments: Any) -> bool:
    """登记审核请求并阻塞等待；超时或用户拒绝返回 False。

    该工具若已被用户在本次会话中勾选「不再询问」，直接放行。
    """
    with _lock:
        if (tool_name or "").lower() in _session_allowed:
            return True
    item = PendingApproval(
        id=uuid.uuid4().hex[:12],
        tool_name=tool_name,
        arguments=_preview(arguments),
        impact=_impact_of(tool_name),
    )
    with _lock:
        _pending[item.id] = item
    try:
        if not item.event.wait(timeout=APPROVAL_TIMEOUT):
            with _lock:
                item.status = "rejected"
            return False
        return item.status == "approved"
    finally:
        with _lock:
            _pending.pop(item.id, None)


def list_pending() -> list[PendingApproval]:
    """当前还在等待用户确认的调用（供 UI 轮询渲染）。"""
    with _lock:
        return [item for item in _pending.values() if item.status == "pending"]


def resolve(approval_id: str, approved: bool, remember: bool = False) -> bool:
    """用户给出结论：放行或拒绝，并唤醒等待中的工具线程。

    remember 为真且放行时，本次会话内后续同工具的调用不再询问。
    """
    with _lock:
        item = _pending.get(approval_id)
        if item is None or item.status != "pending":
            return False
        item.status = "approved" if approved else "rejected"
        if approved and remember:
            _session_allowed.add((item.tool_name or "").lower())
    item.event.set()
    return True


def session_allowed() -> set[str]:
    """当前会话内已免审的工具名（供界面展示）。"""
    with _lock:
        return set(_session_allowed)
