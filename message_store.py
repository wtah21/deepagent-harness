"""
项目消息归档。

分层压缩会直接改写 state.messages，被压缩掉的原文只在检查点里留一份。
本模块在消息产生后把每条消息异步写入 project_messages 表，
按 workspace_id（即项目 thread_id）+ channel（执行模式）归档，作为压缩前的完整历史。
channel 把「主智能体」与「Harness 流水线」两套对话彻底分开：
两者共享同一个项目的文件与工作区，但各自的聊天记录互不混排。

写入由后台守护线程完成，调用方只做一次入队，不会被数据库 I/O 阻塞；
同一 message_id 只保留首次写入的内容（INSERT OR IGNORE），
因此重复归档同一批历史不会产生重复记录。
"""
import atexit
import json
import queue
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Sequence

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)

from config import DB_PATH, debug_log

_INSERT_SQL = (
    "INSERT OR IGNORE INTO project_messages "
    "(message_id, workspace_id, channel, role, content, tool_calls, tool_call_id, name, reasoning, created_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

# 与 persistence.init_db 中的定义保持一致；这里自建一次，
# 保证只有本模块被导入的进程（如并行工作进程）也能正常归档。
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS project_messages (
    message_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    channel TEXT DEFAULT 'main',
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls TEXT DEFAULT '',
    tool_call_id TEXT DEFAULT '',
    name TEXT DEFAULT '',
    reasoning TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_project_messages_ws ON project_messages (workspace_id);
"""

_MAIN_CHANNEL = "main"

_queue: "queue.Queue[list[tuple]]" = queue.Queue()
_worker: threading.Thread | None = None
_lock = threading.Lock()
# workspace_id -> 已入队的 message_id，避免重复转换与入队
_seen: dict[str, set[str]] = {}


def _ensure_schema() -> None:
    """确保归档表存在（幂等），并给老库补上思考过程与渠道字段。"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(project_messages)")}
        if "reasoning" not in columns:
            conn.execute("ALTER TABLE project_messages ADD COLUMN reasoning TEXT DEFAULT ''")
            conn.commit()
        if "channel" not in columns:
            # 老库里的历史没有渠道概念，统一归到主智能体这份聊天里
            conn.execute(
                f"ALTER TABLE project_messages ADD COLUMN channel TEXT DEFAULT '{_MAIN_CHANNEL}'")
            conn.commit()
    finally:
        conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_of(message: BaseMessage) -> str:
    """消息内容统一序列化成字符串（多模态内容块转 JSON）。"""
    content = message.content
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except TypeError:
        return str(content)


def _reasoning_of(message: BaseMessage) -> str:
    """取出模型返回的思考过程（ReasoningChatOpenAI 写在 additional_kwargs 里）。"""
    extra = getattr(message, "additional_kwargs", None) or {}
    text = extra.get("reasoning_content")
    return text if isinstance(text, str) else ""


def _to_record(workspace_id: str, channel: str, message: BaseMessage) -> tuple:
    message_id = getattr(message, "id", None) or f"msg_{uuid.uuid4().hex}"
    # 把生成的 id 写回消息对象：LangGraph 之后会给无 id 的消息补 UUID，
    # 若不写回，同一条消息两次归档（发送时 + 执行完成后全量归档）会
    # 各持一个不同 id 而落成两条记录，界面上表现为用户消息重复。
    setattr(message, "id", message_id)
    tool_calls = getattr(message, "tool_calls", None) or []
    return (
        str(message_id),
        str(workspace_id),
        str(channel or _MAIN_CHANNEL),
        str(getattr(message, "type", "unknown")),
        _content_of(message),
        json.dumps(tool_calls, ensure_ascii=False, default=str) if tool_calls else "",
        str(getattr(message, "tool_call_id", "") or ""),
        str(getattr(message, "name", "") or ""),
        _reasoning_of(message),
        _utc_now(),
    )


def archive_messages(workspace_id: str, messages: Sequence[BaseMessage],
                     channel: str = _MAIN_CHANNEL) -> None:
    """把一批消息异步归档到所属项目的指定渠道；非消息对象与已入队的消息会被跳过。

    `_seen` 按 workspace 记录 message_id 且跨渠道共享：一条消息只会进入它
    第一次归档的那个渠道，避免主智能体与流水线两次落盘时互相串历史。
    """
    if not workspace_id or not messages:
        return
    workspace_key = str(workspace_id)
    seen = _seen.setdefault(workspace_key, set())
    records = []
    for message in messages:
        if not isinstance(message, BaseMessage):
            continue
        message_id = getattr(message, "id", None)
        if message_id and message_id in seen:
            continue
        records.append(_to_record(workspace_key, channel, message))
        if message_id:
            seen.add(message_id)
    if not records:
        return
    _queue.put(records)
    _ensure_worker()


def _ensure_worker() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_loop, name="message-archiver", daemon=True)
        _worker.start()


def _loop() -> None:
    """后台写线程：持有独立连接，按批次落库。"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        while True:
            batch = _queue.get()
            if batch is None:
                break
            try:
                with conn:
                    conn.executemany(_INSERT_SQL, batch)
            except Exception as exc:
                debug_log(f"[message-store] 写入失败：{type(exc).__name__}: {exc}")
            finally:
                _queue.task_done()
    finally:
        conn.close()


def flush() -> None:
    """等待队列中的消息全部落库（测试或退出前使用）。"""
    _queue.join()


def list_messages(workspace_id: str, channel: str | None = None) -> list[dict]:
    """按归档顺序读取某个项目某个渠道的全部消息；channel 为空时返回全部渠道。"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        if channel is None:
            rows = conn.execute(
                "SELECT * FROM project_messages WHERE workspace_id = ? ORDER BY rowid",
                (str(workspace_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM project_messages WHERE workspace_id = ? AND channel = ? "
                "ORDER BY rowid",
                (str(workspace_id), str(channel)),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def delete_messages_by_workspace(workspace_id: str) -> int:
    """删除某个项目全部渠道的归档消息（项目删除时调用），返回删除条数。

    同步执行：只在项目删除这类一次性操作里调用，不值得走异步队列。
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        cursor = conn.execute(
            "DELETE FROM project_messages WHERE workspace_id = ?",
            (str(workspace_id),),
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def _from_record(row: dict) -> BaseMessage:
    """数据库记录 → LangChain 消息对象，供前端历史回放使用。"""
    role = row.get("role") or "human"
    content = row.get("content") or ""
    message_id = row.get("message_id") or None

    if role == "ai":
        tool_calls = []
        raw_calls = row.get("tool_calls") or ""
        if raw_calls:
            try:
                tool_calls = json.loads(raw_calls)
            except (TypeError, ValueError):
                tool_calls = []
        reasoning = row.get("reasoning") or ""
        return AIMessage(
            content=content,
            id=message_id,
            tool_calls=tool_calls,
            additional_kwargs={"reasoning_content": reasoning} if reasoning else {},
        )
    if role == "tool":
        return ToolMessage(
            content=content,
            id=message_id,
            tool_call_id=row.get("tool_call_id") or "",
            name=row.get("name") or None,
        )
    if role == "system":
        return SystemMessage(content=content, id=message_id)
    return HumanMessage(content=content, id=message_id)


def load_messages(workspace_id: str, channel: str | None = None) -> list[BaseMessage]:
    """加载某个项目某个渠道的完整会话消息（压缩丢弃前的原文都在这里）。"""
    return [_from_record(row) for row in list_messages(workspace_id, channel)]


_ensure_schema()
atexit.register(flush)
