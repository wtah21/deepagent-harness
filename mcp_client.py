"""
MCP（Model Context Protocol）客户端管理。

职责：
  - 读取 / 写入 deepagent_data/mcp_servers.json（全局 MCP 服务器配置）；
  - 支持 stdio、sse、streamable_http 三种传输方式；
  - 用后台事件循环常驻 MCP 会话，把异步 MCP 调用桥接成同步工具函数，
    这样现有同步的 LangGraph 智能体可以直接使用；
  - 把发现到的 MCP 工具以 `<服务器名>__<工具名>` 注册进工具注册表。

配置示例（mcp_servers.json）：
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\\\Users\\\\me\\\\Desktop"],
      "env": {"API_KEY": "xxx"},
      "enabled": true
    },
    "remote": {
      "transport": "sse",
      "url": "https://mcp.example.com/sse",
      "headers": {"Authorization": "Bearer xxx"},
      "enabled": true
    }
  }
}
"""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from config import MCP_CONFIG_PATH, debug_log

MCP_SERVERS_KEY = "mcpServers"
TRANSPORTS = ("stdio", "sse", "streamable_http")
CONNECT_TIMEOUT = 30.0
CALL_TIMEOUT = 120.0
# 连接后的试调用验证：只要能拿到结果就说明凭据与权限可用，不必等太久
VERIFY_TIMEOUT = 20.0

# 试调用优先挑这些只读语义的工具，避免连接校验就产生副作用
_READONLY_HINTS = (
    "list", "get", "search", "read", "info", "status", "describe",
    "fetch", "query", "ping", "help", "whoami", "count",
)

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


# ── 配置校验 ──────────────────────────────────────────────────────────

def _as_str_list(value: Any, label: str, errors: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        errors.append(f"{label} 必须是字符串数组")
        return []
    return list(value)


def _as_str_map(value: Any, label: str, errors: list[str]) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(v, (str, int, float, bool)) for v in value.values()
    ):
        errors.append(f"{label} 必须是由字符串键值对组成的对象")
        return {}
    return {str(k): str(v) for k, v in value.items()}


def validate_server_name(name: str) -> tuple[str, Optional[str]]:
    """校验 MCP 服务器名；名称会作为工具名前缀，因此限制为安全标识符。"""
    cleaned = str(name or "").strip()
    if not cleaned:
        return "", "服务器名称不能为空"
    if not _NAME_RE.match(cleaned):
        return cleaned, "服务器名称只能包含字母、数字、下划线、连字符，且不超过 40 字符"
    return cleaned, None


def validate_server_config(raw: dict) -> tuple[dict, list[str]]:
    """校验并规范化单台 MCP 服务器的配置，返回 (规范化配置, 错误列表)。"""
    if not isinstance(raw, dict):
        return {}, ["MCP 服务器配置必须是 JSON 对象"]

    errors: list[str] = []
    transport = str(raw.get("transport") or raw.get("type") or "").strip().lower()
    if transport in ("streamable-http", "streamablehttp", "http"):
        transport = "streamable_http"
    if not transport:
        # 未显式声明时按字段推断，兼容常见的 mcpServers 配置写法
        if raw.get("command"):
            transport = "stdio"
        elif raw.get("url") or raw.get("endpoint"):
            transport = "sse"
        else:
            return {}, ["缺少 transport（stdio / sse / streamable_http），也无法根据 command/url 推断"]
    if transport not in TRANSPORTS:
        return {}, [f"transport 必须是 {' / '.join(TRANSPORTS)} 之一"]

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        errors.append("enabled 必须是布尔值")

    normalized: dict[str, Any] = {"transport": transport, "enabled": enabled}

    if transport == "stdio":
        command = str(raw.get("command") or "").strip()
        if not command:
            errors.append("stdio 模式必须填写 command（如 node / python / npx / uvx / docker）")
        normalized["command"] = command
        normalized["args"] = _as_str_list(raw.get("args"), "args", errors)
        normalized["env"] = _as_str_map(raw.get("env"), "env", errors)
        cwd = str(raw.get("cwd") or "").strip()
        if cwd:
            normalized["cwd"] = cwd
    else:
        url = str(raw.get("url") or raw.get("endpoint") or "").strip()
        if not url:
            errors.append(f"{transport} 模式必须填写 url（如 https://mcp.example.com/sse）")
        elif not url.startswith(("http://", "https://")):
            errors.append("url 必须以 http:// 或 https:// 开头")
        headers = raw.get("headers", raw.get("auth"))
        if isinstance(headers, str):
            headers = {"Authorization": headers} if headers else {}
        normalized["url"] = url
        normalized["headers"] = _as_str_map(headers, "headers", errors)

    return ({}, errors) if errors else (normalized, [])


# ── 配置文件读写 ──────────────────────────────────────────────────────

def load_mcp_config() -> dict:
    """读取 MCP 配置文件；文件不存在或损坏时返回空配置。"""
    empty = {MCP_SERVERS_KEY: {}}
    if not MCP_CONFIG_PATH.exists():
        return empty
    try:
        data = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        debug_log(f"[mcp] 读取配置失败：{type(exc).__name__}: {exc}")
        return empty
    if not isinstance(data, dict):
        return empty
    servers = data.get(MCP_SERVERS_KEY)
    if not isinstance(servers, dict):
        servers = {}
    return {**data, MCP_SERVERS_KEY: servers}


def _write_config(data: dict) -> None:
    """原子写入配置文件，避免写入中断留下半个 JSON。"""
    MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MCP_CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, MCP_CONFIG_PATH)


def save_mcp_server(name: str, raw: dict) -> tuple[bool, str]:
    """校验并保存一台服务器配置（不负责连接与注册）。"""
    cleaned, name_error = validate_server_name(name)
    if name_error:
        return False, name_error
    normalized, errors = validate_server_config(raw)
    if errors:
        return False, "；".join(errors)
    data = load_mcp_config()
    data[MCP_SERVERS_KEY][cleaned] = normalized
    try:
        _write_config(data)
    except OSError as exc:
        return False, f"写入配置文件失败：{exc}"
    return True, ""


def delete_mcp_server(name: str) -> bool:
    """从配置文件中删除一台服务器，返回是否真的删除了条目。"""
    data = load_mcp_config()
    if name not in data[MCP_SERVERS_KEY]:
        return False
    data[MCP_SERVERS_KEY].pop(name)
    try:
        _write_config(data)
    except OSError as exc:
        debug_log(f"[mcp] 删除配置失败：{exc}")
        return False
    return True


# ── JSON Schema → Pydantic 参数模型 ───────────────────────────────────

def _safe_identifier(name: str) -> str:
    ident = re.sub(r"\W", "_", str(name or ""))
    if not ident or ident[0].isdigit():
        ident = f"_{ident}"
    return ident


def _json_type(schema: Any, defs: dict, model_name: str) -> Any:
    """把 MCP 工具的 inputSchema 片段转换成 Python 类型。"""
    if not isinstance(schema, dict):
        return Any

    if "$ref" in schema:
        key = str(schema["$ref"]).rsplit("/", 1)[-1]
        return _json_type(defs.get(key, {}), defs, model_name)

    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and options:
            nullable = any(isinstance(o, dict) and o.get("type") == "null" for o in options)
            types = [
                _json_type(o, defs, f"{model_name}Option")
                for o in options
                if not (isinstance(o, dict) and o.get("type") == "null")
            ]
            if not types:
                return Any
            typ = types[0] if len(types) == 1 else Union[tuple(types)]
            return Optional[typ] if nullable else typ

    enum = schema.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(v, str) for v in enum):
        return Literal[tuple(enum)]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        typ = _json_type({**schema, "type": non_null[0]}, defs, model_name) if non_null else Any
        return Optional[typ] if "null" in schema_type else typ

    if schema_type == "array":
        return list[_json_type(schema.get("items") or {}, defs, f"{model_name}Item")]
    if schema_type == "object" or "properties" in schema:
        return _object_model(schema, defs, model_name)

    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": type(None),
    }.get(schema_type, Any)


def _object_model(schema: dict, defs: dict, model_name: str) -> type[BaseModel]:
    """根据 JSON Schema 的 object 片段生成 Pydantic 模型。"""
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}

    for prop, prop_schema in props.items():
        if not isinstance(prop_schema, dict):
            prop_schema = {}
        ident = _safe_identifier(prop)
        typ = _json_type(prop_schema, defs, f"{model_name}_{ident}")
        alias = prop if ident != prop else None
        description = prop_schema.get("description")
        if prop in required:
            fields[ident] = (typ, Field(..., alias=alias, description=description))
        else:
            fields[ident] = (
                Optional[typ],
                Field(prop_schema.get("default"), alias=alias, description=description),
            )

    return create_model(
        model_name,
        __config__=ConfigDict(extra="allow", populate_by_name=True),
        **fields,
    )


def _jsonable(value: Any) -> Any:
    """把 Pydantic 模型 / 元组递归转换成可直接交给 MCP 的 JSON 值。"""
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(by_alias=True, exclude_none=True))
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


# ── 会话管理 ──────────────────────────────────────────────────────────

class _ServerConnection:
    """一台 MCP 服务器的常驻会话状态。"""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.session: Any = None
        self.tools: list[Any] = []
        self.error: Optional[str] = None
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self.stop_event: Optional[asyncio.Event] = None
        self.task: Optional[asyncio.Task] = None


class MCPManager:
    """后台事件循环 + 每台服务器一个常驻 MCP 会话。"""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop_started = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="mcp-event-loop", daemon=True)
        self._thread.start()
        self._loop_started.wait(5.0)

        self._lock = threading.RLock()
        self._conns: dict[str, _ServerConnection] = {}
        # server -> 已注册的工具名，重载时用于清理旧工具
        self._registered: dict[str, list[str]] = {}
        self._last_report: dict[str, dict] = {}
        # 服务器名 -> 最近一次试调用结果，供侧边栏展示真实可用性
        self._verify_results: dict[str, dict] = {}
        atexit.register(self.shutdown)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._loop_started.set)
        self._loop.run_forever()

    # ── 传输层 ────────────────────────────────────────────────────────

    @asynccontextmanager
    async def _transport(self, config: dict):
        """按配置打开对应传输方式的读写流。"""
        transport = config["transport"]

        if transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            env = {**os.environ}
            env.update({str(k): str(v) for k, v in (config.get("env") or {}).items()})
            params = StdioServerParameters(
                command=config["command"],
                args=list(config.get("args") or []),
                env=env,
                cwd=config.get("cwd") or None,
            )
            async with stdio_client(params) as streams:
                yield streams
            return

        if transport == "sse":
            from mcp.client.sse import sse_client

            async with sse_client(
                config["url"],
                headers=config.get("headers") or None,
                timeout=CONNECT_TIMEOUT,
            ) as streams:
                yield streams
            return

        # streamable_http
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        client = create_mcp_http_client(headers=config.get("headers") or None)
        try:
            async with streamable_http_client(config["url"], http_client=client) as streams:
                yield streams
        finally:
            await client.aclose()

    async def _serve(self, conn: _ServerConnection) -> None:
        """在事件循环里维持一条会话，直到 stop_event 被触发。"""
        from mcp import ClientSession

        conn.task = asyncio.current_task()
        try:
            async with self._transport(conn.config) as (read, write):
                async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=CALL_TIMEOUT,
                ) as session:
                    await asyncio.wait_for(session.initialize(), timeout=CONNECT_TIMEOUT)
                    listed = await asyncio.wait_for(session.list_tools(), timeout=CONNECT_TIMEOUT)
                    conn.tools = list(getattr(listed, "tools", None) or [])
                    conn.session = session
                    conn.stop_event = asyncio.Event()
                    conn.error = None
                    conn.ready.set()
                    await conn.stop_event.wait()
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # 连接失败只影响这台服务器，不拖垮整个应用
            conn.error = f"{type(exc).__name__}: {exc}"
            debug_log(f"[mcp] server={conn.name} connect failed: {conn.error}")
        finally:
            conn.session = None
            conn.stop_event = None
            conn.task = None
            conn.ready.set()
            conn.stopped.set()

    def _spawn(self, conn: _ServerConnection) -> None:
        conn.ready.clear()
        conn.stopped.clear()
        conn.error = None
        asyncio.run_coroutine_threadsafe(self._serve(conn), self._loop)

    def _stop(self, conn: _ServerConnection, timeout: float = 5.0) -> None:
        """请求会话退出并等待清理完成。"""
        if conn.stopped.is_set():
            return

        def _signal() -> None:
            if conn.stop_event is not None:
                conn.stop_event.set()
            elif conn.task is not None:
                conn.task.cancel()

        try:
            self._loop.call_soon_threadsafe(_signal)
        except RuntimeError:
            return
        if conn.stopped.wait(timeout):
            return
        try:
            self._loop.call_soon_threadsafe(
                lambda: conn.task is not None and conn.task.cancel()
            )
        except RuntimeError:
            pass
        conn.stopped.wait(2.0)

    # ── 工具转换与注册 ────────────────────────────────────────────────

    @staticmethod
    def _tool_name(server: str, tool_name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}__{tool_name}")
        return safe[:64]

    def _make_callable(self, server: str, tool: Any):
        """生成同步工具函数：把参数交给常驻会话执行。"""
        schema = getattr(tool, "input_schema", None) or {}
        props = schema.get("properties") if isinstance(schema, dict) else {}
        # Pydantic 字段名可能被清洗过，这里记录原名以便还原给 MCP
        field_names = {_safe_identifier(k): k for k in (props or {})}
        mcp_tool_name = tool.name

        def _call(**kwargs: Any) -> str:
            payload: dict[str, Any] = {}
            for key, value in kwargs.items():
                if value is None:
                    continue
                payload[field_names.get(key, key)] = _jsonable(value)
            return self.call_tool(server, mcp_tool_name, payload)

        return _call

    def _args_model(self, server: str, tool: Any) -> type[BaseModel]:
        schema = getattr(tool, "input_schema", None)
        if not isinstance(schema, dict):
            schema = {}
        defs = schema.get("$defs") or schema.get("definitions") or {}
        model_name = _safe_identifier(f"mcp_{server}_{tool.name}_args")[:60]
        return _object_model(schema, defs, model_name)

    def _register_server(self, server: str, tools: list[Any]) -> list[str]:
        from langchain_core.tools import StructuredTool

        from tool_registry import register_tool

        names: list[str] = []
        for tool in tools:
            tool_name = self._tool_name(server, tool.name)
            description = f"[MCP:{server}] {getattr(tool, 'description', None) or tool.name}"
            structured = StructuredTool(
                name=tool_name,
                description=description,
                args_schema=self._args_model(server, tool),
                func=self._make_callable(server, tool),
            )
            register_tool(structured, tool_name, description, "mcp", "moderate")
            names.append(tool_name)
        self._registered[server] = names
        return names

    def _unregister_server(self, server: str) -> None:
        from tool_registry import unregister_tool

        for name in self._registered.pop(server, []):
            unregister_tool(name)

    def _unregister_all(self) -> None:
        for server in list(self._registered):
            self._unregister_server(server)

    # ── 对外操作 ──────────────────────────────────────────────────────

    def call_tool(self, server: str, tool_name: str, arguments: dict) -> str:
        """同步调用 MCP 工具，返回可直接展示给模型的文本。

        这里刻意不抛异常：异常一旦冒泡就会终止整轮 agent 执行，
        用户只能看到一条笼统的运行时错误，模型也没有机会自救。
        把失败原因作为工具结果回传后，模型可以换用其它工具、修正参数，
        或者把错误如实转达给用户。
        """
        conn = self._conns.get(server)
        if conn is None or conn.session is None:
            reason = conn.error if conn is not None else "服务器未加载"
            return (
                f"⚠️ MCP 服务器 `{server}` 当前不可用：{reason or '尚未连接'}。\n"
                f"工具 `{server}__{tool_name}` 没有执行，请不要假设它已成功；"
                "可以改用其它工具完成该步骤，或把该错误告知用户。"
            )

        async def _call():
            return await asyncio.wait_for(
                conn.session.call_tool(tool_name, arguments or {}),
                timeout=CALL_TIMEOUT,
            )

        try:
            result = asyncio.run_coroutine_threadsafe(_call(), self._loop).result(CALL_TIMEOUT + 5)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            debug_log(f"[mcp] call failed | {server}__{tool_name} | {detail}")
            return (
                f"⚠️ MCP 工具 `{server}__{tool_name}` 调用失败：{detail}\n"
                "该调用没有产生任何结果，请不要假设它已执行或已成功。"
                "若属于参数或鉴权问题，可修正后重试，或改用其它工具；"
                "无法自行解决时把错误如实告知用户。"
            )
        return _format_result(result)

    def probe(self, name: str, raw: dict, timeout: float = CONNECT_TIMEOUT) -> dict:
        """临时连接一台服务器，拉取工具清单并试调用一次，返回验证结果。

        返回 {
            "tools": [(工具名, 描述), ...],
            "tools_error": 连接或取工具清单失败的原因,
            "verify": {"tool": 试调用的工具名, "ok": 是否成功, "detail": 结果或错误详情},
        }
        只拿到工具清单不代表服务可用：上游往往在真正调用时才校验凭据
        （例如 Browserbase 托管端点要求 URL 带 browserbaseApiKey），
        所以这里必须实际调用一次，把失败原因原样交给用户。
        """
        result = {"tools": [], "tools_error": None, "verify": None}

        normalized, errors = validate_server_config(raw)
        if errors:
            result["tools_error"] = "；".join(errors)
            return result

        conn = _ServerConnection(name, normalized)
        self._spawn(conn)
        conn.ready.wait(timeout)
        if conn.session is None:
            result["tools_error"] = conn.error or f"连接超时（{timeout:.0f} 秒）"
            self._stop(conn, 2.0)
            return result

        result["tools"] = [(t.name, getattr(t, "description", "") or "") for t in conn.tools]
        result["verify"] = self._verify_session(conn)
        self._stop(conn)
        return result

    def _verify_session(self, conn: _ServerConnection, timeout: float = VERIFY_TIMEOUT) -> dict:
        """实际调用一个无必填参数的工具，确认这台上游服务真的能返回数据。"""
        result = self._run_verify(conn, timeout)
        # 记住最近一次验证结果，侧边栏随时可以展示真实可用性
        self._verify_results[conn.name] = result
        return result

    def _run_verify(self, conn: _ServerConnection, timeout: float) -> dict:
        picked = _pick_verify_tool(conn.tools)
        if picked is None:
            return {
                "tool": None,
                "ok": None,
                "detail": "所有工具都有必填参数，无法自动试调用验证",
            }

        async def _call():
            return await asyncio.wait_for(
                conn.session.call_tool(picked.name, {}),
                timeout=timeout,
            )

        try:
            res = asyncio.run_coroutine_threadsafe(_call(), self._loop).result(timeout + 5)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            debug_log(f"[mcp] verify failed | {conn.name}__{picked.name} | {detail}")
            return {"tool": picked.name, "ok": False, "detail": detail}

        text = _format_result(res)
        ok = not (getattr(res, "is_error", False) or getattr(res, "isError", False))
        if ok and text.startswith("⚠️"):
            ok = False
        return {"tool": picked.name, "ok": ok, "detail": text[:400]}

    def reload(self, verify: bool = False) -> dict[str, dict]:
        """按配置文件重新连接全部服务器并重建 MCP 工具。返回逐服务器结果。

        verify=True 时对每台服务器额外做一次试调用：只连上并列出工具
        并不代表上游真的可用（凭据往往在调用时才校验）。
        """
        with self._lock:
            self._unregister_all()
            for conn in list(self._conns.values()):
                self._stop(conn)
            self._conns = {}

            servers = load_mcp_config().get(MCP_SERVERS_KEY, {})
            started: dict[str, _ServerConnection] = {}
            report: dict[str, dict] = {}

            for name, raw in servers.items():
                cleaned, name_error = validate_server_name(name)
                if name_error:
                    report[str(name)] = {"ok": False, "error": name_error, "tools": 0}
                    continue
                normalized, errors = validate_server_config(raw)
                if errors:
                    report[cleaned] = {"ok": False, "error": "；".join(errors), "tools": 0}
                    continue
                if not normalized.get("enabled", True):
                    report[cleaned] = {"ok": True, "error": None, "tools": 0, "disabled": True}
                    continue
                conn = _ServerConnection(cleaned, normalized)
                self._spawn(conn)
                started[cleaned] = conn

            deadline = time.time() + CONNECT_TIMEOUT
            for name, conn in started.items():
                conn.ready.wait(max(0.0, deadline - time.time()))
                if conn.session is None:
                    report[name] = {
                        "ok": False,
                        "error": conn.error or f"连接超时（{CONNECT_TIMEOUT:.0f} 秒）",
                        "tools": 0,
                    }
                    self._stop(conn, 2.0)
                    continue
                tool_names = self._register_server(name, conn.tools)
                self._conns[name] = conn
                report[name] = {
                    "ok": True,
                    "error": None,
                    "tools": len(tool_names),
                    "tool_names": tool_names,
                }
                if verify:
                    report[name]["verify"] = self._verify_session(conn)

            self._last_report = report
            debug_log(
                "[mcp] reload | "
                + ", ".join(f"{k}:{'ok' if v.get('ok') else 'fail'}({v.get('tools', 0)})" for k, v in report.items())
            )
            return report

    def remove(self, name: str) -> None:
        """断开并移除一台在内存中已加载的服务器（配置删除另由调用方负责）。"""
        with self._lock:
            conn = self._conns.pop(name, None)
            if conn is not None:
                self._stop(conn)
            self._unregister_server(name)
            self._last_report.pop(name, None)
            self._verify_results.pop(name, None)

    def status(self) -> dict:
        """返回当前配置与连接状态，供侧边栏展示。"""
        servers: dict[str, dict] = {}
        for name, raw in load_mcp_config().get(MCP_SERVERS_KEY, {}).items():
            normalized, errors = validate_server_config(raw)
            conn = self._conns.get(name)
            report = self._last_report.get(name, {})
            servers[str(name)] = {
                "transport": normalized.get("transport", raw.get("transport", "?")),
                "enabled": raw.get("enabled", True),
                "connected": bool(conn is not None and conn.session is not None),
                "tools": len(conn.tools) if conn is not None else report.get("tools", 0),
                "error": (conn.error if conn is not None else None)
                or (errors[0] if errors else report.get("error")),
                "verify": self._verify_results.get(str(name)),
            }
        return {"config_path": str(MCP_CONFIG_PATH), "servers": servers}

    def shutdown(self) -> None:
        """退出时关闭全部会话与后台事件循环。"""
        with self._lock:
            for conn in list(self._conns.values()):
                self._stop(conn, 2.0)
            self._conns.clear()
            self._unregister_all()
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass


def _pick_verify_tool(tools: list[Any]) -> Optional[Any]:
    """挑一个可用于试调用的工具：无必填参数，且优先只读语义。

    没有无参工具时返回 None —— 不能凭空编造参数去调用，
    那样只会把「参数不合法」误判成「服务不可用」。
    """
    candidates = []
    for tool in tools:
        schema = getattr(tool, "input_schema", None) or {}
        if isinstance(schema, dict) and not schema.get("required"):
            candidates.append(tool)
    if not candidates:
        return None

    readonly = [
        tool for tool in candidates
        if any(hint in str(getattr(tool, "name", "")).lower() for hint in _READONLY_HINTS)
    ]
    return (readonly or candidates)[0]


def _format_result(result: Any) -> str:
    """把 MCP 的 CallToolResult 转成模型可读的文本。"""
    parts: list[str] = []
    for item in getattr(result, "content", None) or []:
        kind = getattr(item, "type", "")
        if kind == "text":
            parts.append(str(getattr(item, "text", "")))
        elif kind == "image":
            mime = getattr(item, "mime_type", None) or getattr(item, "mimeType", "") or "image"
            parts.append(f"[图片内容，类型 {mime}]")
        elif kind == "audio":
            mime = getattr(item, "mime_type", None) or getattr(item, "mimeType", "") or "audio"
            parts.append(f"[音频内容，类型 {mime}]")
        else:
            data = item.model_dump(by_alias=True) if hasattr(item, "model_dump") else str(item)
            parts.append(json.dumps(data, ensure_ascii=False, default=str))

    text = "\n".join(p for p in parts if p)
    if not text:
        structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
        if structured is not None:
            text = json.dumps(structured, ensure_ascii=False, default=str)
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        return "⚠️ MCP 工具返回错误：" + (text or "（无内容）")
    return text or "（MCP 工具未返回文本内容）"


# ── 模块级便捷入口 ────────────────────────────────────────────────────

_manager: Optional[MCPManager] = None
_manager_lock = threading.Lock()


def get_manager() -> MCPManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = MCPManager()
        return _manager


def load_and_register(verify: bool = False) -> dict[str, dict]:
    """读取配置、连接服务器并注册 MCP 工具（应用启动与配置变更后调用）。

    verify=True 时额外做一次试调用验证；启动时保持 False，
    避免每次启动都触发上游的写操作（如创建浏览器会话）。
    """
    return get_manager().reload(verify=verify)


def probe_server(name: str, raw: dict) -> dict:
    """临时连接并试调用一次，返回 {tools, tools_error, verify}。"""
    return get_manager().probe(name, raw)


def mcp_status() -> dict:
    return get_manager().status()


def remove_mcp_server(name: str) -> None:
    get_manager().remove(name)
