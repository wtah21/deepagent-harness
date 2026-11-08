"""
A2A（Agent2Agent）远端智能体接入。

职责：
  - 读取 / 写入 deepagent_data/a2a_agents.json（全局 A2A Agent 配置）；
  - 按 `{url}/.well-known/agent-card.json` 拉取并解析 Agent Card；
  - 同时兼容 A2A v0.3.0 与 v1.0：
      v0.3.0：JSON-RPC 方法 `message/send`，Part 带 `kind` 判别字段，role 为 `user`；
      v1.0  ：JSON-RPC 方法 `SendMessage`，Part 用成员多态，role 为 `ROLE_USER`；
  - 每个远端 Agent 注册为一个工具 `a2a__<名称>`，主智能体直接拥有，
    子智能体按自身 tools 配置取用；
  - 调用时按 function calling 组装的 instructions 组装 JSON-RPC 请求，
    把远端返回的 Task / Message 解析成纯文本交回主智能体。

配置示例（a2a_agents.json）：
{
  "agents": {
    "weather": {
      "url": "https://weather-agent.example.com",
      "api_key": "xxx",
      "enabled": true,
      "protocol_version": "0.3.0",
      "card": {"name": "Weather Agent", "url": "...", "skills": [...]}
    }
  }
}
"""
from __future__ import annotations

import copy
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from config import A2A_CARD_TIMEOUT, A2A_CONFIG_PATH, A2A_TIMEOUT, debug_log

AGENTS_KEY = "agents"
AGENT_CARD_PATH = "/.well-known/agent-card.json"
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
# 注册到工具描述里的技能摘要上限，避免名片正文挤占主智能体上下文
_DESCRIPTION_LIMIT = 1200

_V1_STATES = {
    "TASK_STATE_COMPLETED": "已完成",
    "TASK_STATE_FAILED": "失败",
    "TASK_STATE_CANCELED": "已取消",
    "TASK_STATE_REJECTED": "已拒绝",
    "TASK_STATE_INPUT_REQUIRED": "需要补充输入",
    "TASK_STATE_AUTH_REQUIRED": "需要认证",
    "TASK_STATE_SUBMITTED": "已提交",
    "TASK_STATE_WORKING": "处理中",
}
_V03_STATES = {
    "completed": "已完成",
    "failed": "失败",
    "canceled": "已取消",
    "cancelled": "已取消",
    "rejected": "已拒绝",
    "input-required": "需要补充输入",
    "auth-required": "需要认证",
    "submitted": "已提交",
    "working": "处理中",
}
_INTERRUPTED_STATES = {"需要补充输入", "需要认证"}
# 同一工具因字段/参数不匹配而被模型用 custom_payload_overrides 修复的最大次数
_MAX_OVERRIDE_RETRIES = 3
_ATTEMPT_CACHE_LIMIT = 500
_HTTP_4XX_RE = re.compile(r"^HTTP 4\d\d")
# 工具名 → 已用掉的 override 重试次数；调用成功后清除
_override_attempts: dict[str, int] = {}


class A2AError(Exception):
    """A2A 配置或调用过程中的可预期错误。"""


@dataclass
class AgentProfile:
    """解析后的远端 Agent 名片。"""

    name: str
    description: str
    protocol: str
    endpoint: str
    card: dict = field(default_factory=dict)
    skills: list[dict] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    query: dict[str, str] = field(default_factory=dict)


class _A2AArgs(BaseModel):
    """远端 Agent 工具的唯一入参。"""

    instructions: str = Field(
        description=(
            "发给远端 Agent 的任务说明，必须自包含：清晰的目标、完成任务所必需的上下文，"
            "以及期望的输出格式。远端 Agent 看不到本平台的对话历史。"
        )
    )
    custom_payload_overrides: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "仅在远端返回字段/参数不匹配错误时使用：要覆盖的请求字段路径与新值，"
            '例如 {"params.configuration.blocking": null, '
            '"params.configuration.returnImmediately": false}。'
            "路径以 params 为根（可省略 params.），值为 null 表示删除该字段。"
            "系统会用覆盖后的请求体重发，同一工具最多重试 3 次。"
        ),
    )


# ── 配置校验与读写 ────────────────────────────────────────────────────

def validate_agent_name(name: str) -> tuple[str, Optional[str]]:
    """校验 A2A Agent 名称；名称会作为工具名后缀。"""
    cleaned = str(name or "").strip()
    if not cleaned:
        return "", "Agent 名称不能为空"
    if not _NAME_RE.match(cleaned):
        return cleaned, "Agent 名称只能包含字母、数字、下划线、连字符，且不超过 40 字符"
    return cleaned, None


def suggest_agent_name(card_name: str) -> str:
    """把名片里的名字转成合法工具名；无法转换时用随机后缀。"""
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", str(card_name or "")).strip("_")[:40]
    if not slug or not re.search(r"[A-Za-z0-9]", slug):
        return f"agent_{uuid.uuid4().hex[:6]}"
    if slug[0].isdigit():
        slug = f"a2a_{slug}"[:40]
    return slug


def validate_agent_config(raw: dict) -> tuple[dict, list[str]]:
    """校验并规范化一个 A2A Agent 配置，返回 (规范化配置, 错误列表)。"""
    if not isinstance(raw, dict):
        return {}, ["A2A Agent 配置必须是 JSON 对象"]

    errors: list[str] = []
    url = str(raw.get("url") or "").strip().rstrip("/")
    if not url:
        errors.append("URL 不能为空（例如 https://agent.example.com）")
    elif not url.lower().startswith(("http://", "https://")):
        errors.append("URL 必须以 http:// 或 https:// 开头")

    api_key = str(raw.get("api_key") or "").strip()
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        errors.append("enabled 必须是布尔值")

    normalized: dict[str, Any] = {
        "url": url,
        "api_key": api_key,
        "enabled": enabled,
    }
    card = raw.get("card")
    if isinstance(card, dict):
        normalized["card"] = card
    protocol = str(raw.get("protocol_version") or "").strip()
    if protocol:
        normalized["protocol_version"] = protocol

    return ({}, errors) if errors else (normalized, [])


def load_agent_config() -> dict:
    """读取 A2A 配置文件；文件不存在或损坏时返回空配置。"""
    empty = {AGENTS_KEY: {}}
    if not A2A_CONFIG_PATH.exists():
        return empty
    try:
        data = json.loads(A2A_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        debug_log(f"[a2a] 读取配置失败：{type(exc).__name__}: {exc}")
        return empty
    if not isinstance(data, dict):
        return empty
    agents = data.get(AGENTS_KEY)
    if not isinstance(agents, dict):
        agents = {}
    return {**data, AGENTS_KEY: agents}


def _write_config(data: dict) -> None:
    """原子写入配置文件，避免写入中断留下半个 JSON。"""
    A2A_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = A2A_CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, A2A_CONFIG_PATH)


def save_agent(name: str, raw: dict) -> tuple[bool, str]:
    """校验并保存一个 A2A Agent 配置（不负责注册工具）。"""
    cleaned, name_error = validate_agent_name(name)
    if name_error:
        return False, name_error
    normalized, errors = validate_agent_config(raw)
    if errors:
        return False, "；".join(errors)
    data = load_agent_config()
    data[AGENTS_KEY][cleaned] = normalized
    try:
        _write_config(data)
    except OSError as exc:
        return False, f"写入配置文件失败：{exc}"
    return True, ""


def delete_agent(name: str) -> bool:
    """从配置文件中删除一个 Agent，返回是否真的删除了条目。"""
    data = load_agent_config()
    if name not in data[AGENTS_KEY]:
        return False
    data[AGENTS_KEY].pop(name)
    try:
        _write_config(data)
    except OSError as exc:
        debug_log(f"[a2a] 删除配置失败：{exc}")
        return False
    return True


# ── Agent Card 拉取与解析 ─────────────────────────────────────────────

def fetch_agent_card(url: str, api_key: str = "", timeout: int = A2A_CARD_TIMEOUT) -> dict:
    """请求 `{url}/.well-known/agent-card.json` 并返回名片 JSON。"""
    card_url = url.rstrip("/") + AGENT_CARD_PATH
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(card_url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise A2AError(f"获取 Agent Card 失败：HTTP {exc.code}（{card_url}）") from exc
    except urllib.error.URLError as exc:
        raise A2AError(f"无法连接 {card_url}：{exc.reason}") from exc
    except TimeoutError as exc:
        raise A2AError(f"获取 Agent Card 超时（{timeout} 秒）：{card_url}") from exc

    try:
        card = json.loads(body)
    except json.JSONDecodeError as exc:
        raise A2AError("Agent Card 不是合法 JSON") from exc
    if not isinstance(card, dict):
        raise A2AError("Agent Card 必须是 JSON 对象")
    if not card.get("name"):
        raise A2AError("Agent Card 缺少 name 字段")
    return card


def _detect_protocol(card: dict) -> str:
    """有 supportedInterfaces 视为 v1.0，否则按 v0.3.0 处理。"""
    interfaces = card.get("supportedInterfaces") or card.get("supported_interfaces")
    if isinstance(interfaces, list) and interfaces:
        return "1.0"
    return "0.3.0"


def _pick_endpoint(card: dict, protocol: str) -> str:
    """挑选 JSON-RPC 端点 URL。"""
    if protocol == "1.0":
        interfaces = card.get("supportedInterfaces") or card.get("supported_interfaces") or []
        fallback = ""
        for item in interfaces:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            binding = str(
                item.get("protocolBinding") or item.get("protocol_binding") or ""
            ).strip().upper()
            if binding in ("", "JSONRPC", "JSON-RPC"):
                return url
            fallback = fallback or url
        if fallback:
            # 只有非 JSON-RPC 接口（如 gRPC / HTTP+JSON）时仍返回它，调用阶段会给出明确错误
            return fallback
        raise A2AError("Agent Card 的 supportedInterfaces 中没有可用的接口 URL")

    url = str(card.get("url") or "").strip()
    if url:
        return url
    for item in card.get("additionalInterfaces") or card.get("additional_interfaces") or []:
        if not isinstance(item, dict):
            continue
        transport = str(item.get("transport") or item.get("protocolBinding") or "").upper()
        if transport in ("", "JSONRPC", "JSON-RPC") and item.get("url"):
            return str(item["url"]).strip()
    raise A2AError("Agent Card 缺少 url 字段，无法确定调用端点")


def _auth_config(card: dict, api_key: str) -> tuple[dict[str, str], dict[str, str]]:
    """按 Agent Card 的 securitySchemes 决定 API KEY 放在 header 还是 query。"""
    if not api_key:
        return {}, {}

    schemes = card.get("securitySchemes") or card.get("security_schemes") or {}
    candidates = schemes.values() if isinstance(schemes, dict) else []
    for scheme in candidates:
        if not isinstance(scheme, dict):
            continue
        kind = str(scheme.get("type") or scheme.get("scheme_type") or "").lower()
        name = str(scheme.get("name") or "").strip()
        location = str(scheme.get("in") or scheme.get("location") or "header").lower()

        if kind in ("apikey", "api_key", "api-key") or (not kind and name):
            if location == "query":
                return {}, {name or "api_key": api_key}
            header_name = name or "Authorization"
            value = f"Bearer {api_key}" if header_name.lower() == "authorization" else api_key
            return {header_name: value}, {}

        if kind == "http" and str(scheme.get("scheme") or "").lower() == "bearer":
            return {"Authorization": f"Bearer {api_key}"}, {}

    # 名片未声明可识别的方案时按最常见的 Bearer 处理
    return {"Authorization": f"Bearer {api_key}"}, {}


def normalize_card(card: dict, api_key: str = "", name: str = "") -> AgentProfile:
    """把 Agent Card 解析成调用所需的 profile。"""
    protocol = _detect_protocol(card)
    endpoint = _pick_endpoint(card, protocol)
    headers, query = _auth_config(card, api_key)
    skills = [item for item in (card.get("skills") or []) if isinstance(item, dict)]
    return AgentProfile(
        name=name or suggest_agent_name(str(card.get("name") or "")),
        description=str(card.get("description") or ""),
        protocol=protocol,
        endpoint=endpoint,
        card=card,
        skills=skills,
        headers=headers,
        query=query,
    )


def probe_agent(url: str, api_key: str = "") -> tuple[Optional[AgentProfile], Optional[str]]:
    """拉取名片并解析，用于“添加链接”时的校验。返回 (profile, 错误)。"""
    cleaned = str(url or "").strip().rstrip("/")
    if not cleaned:
        return None, "URL 不能为空"
    if not cleaned.lower().startswith(("http://", "https://")):
        return None, "URL 必须以 http:// 或 https:// 开头"
    try:
        card = fetch_agent_card(cleaned, api_key)
        return normalize_card(card, api_key), None
    except A2AError as exc:
        return None, str(exc)
    except Exception as exc:  # 网络/解析异常统一转成可读错误
        return None, f"{type(exc).__name__}: {exc}"


# ── JSON-RPC 调用 ─────────────────────────────────────────────────────

def _build_request(profile: AgentProfile, instructions: str) -> dict:
    """按协议版本组装 message/send 或 SendMessage 请求体。"""
    message_id = str(uuid.uuid4())
    if profile.protocol == "1.0":
        method = "SendMessage"
        message = {
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [{"text": instructions}],
        }
    else:
        method = "message/send"
        message = {
            "messageId": message_id,
            "role": "user",
            "parts": [{"kind": "text", "text": instructions}],
        }
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": {
            "message": message,
            "configuration": {
                "acceptedOutputModes": ["text/plain", "application/json"],
                "blocking": True,
            },
        },
    }


def _post_jsonrpc(profile: AgentProfile, payload: dict) -> dict:
    """发送 JSON-RPC 请求并返回解析后的响应对象。"""
    url = profile.endpoint
    if profile.query:
        separator = "&" if "?" in url else "?"
        url = url + separator + urllib.parse.urlencode(profile.query)

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **profile.headers,
    }
    if profile.protocol == "1.0":
        headers["A2A-Version"] = "1.0"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=A2A_TIMEOUT) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise A2AError(f"HTTP {exc.code}：{detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise A2AError(f"无法连接远端 Agent：{exc.reason}") from exc
    except TimeoutError as exc:
        raise A2AError(f"远端 Agent 调用超时（{A2A_TIMEOUT} 秒）") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise A2AError(f"远端返回的不是合法 JSON：{body[:300]}") from exc


def _format_result(result: Any) -> str:
    """把 JSON-RPC 的 result（Task 或 Message）以原始 JSON 直接交给主智能体。

    result 本身就是结构化 JSON，不做任何拆解拼接——拆 parts 拼文本既可能
    丢字段，也会把结构化数据误当纯字符串处理。模型自己能读懂 JSON。
    """
    body = json.dumps(result, ensure_ascii=False, default=str)

    header = "[A2A] 远端 Agent 返回（原始 JSON）："
    label = ""
    if isinstance(result, dict) and "status" in result:
        status = result.get("status") if isinstance(result.get("status"), dict) else {}
        raw_state = str(status.get("state") or "")
        label = _V1_STATES.get(raw_state) or _V03_STATES.get(raw_state) or raw_state or "未知"
        header = f"[A2A] 远端 Agent 返回（原始 JSON，状态：{label}）："

    if label in _INTERRUPTED_STATES:
        body += (
            "\n\n远端 Agent 需要更多信息才能继续：请把 JSON 中的问题转达给用户，"
            "或在 instructions 中补充必要信息后再次调用。"
        )
    return f"{header}\n{body}"


def _apply_overrides(payload: dict, overrides: dict) -> dict:
    """按字段路径覆盖请求体；值为 None 时删除该字段。

    路径以 params 为根，允许省略 `params.`（如 `configuration.blocking`）；
    也允许用 `method` 修改 JSON-RPC 方法名。jsonrpc / id 不允许被覆盖。
    """
    result = copy.deepcopy(payload)
    for raw_path, value in overrides.items():
        segments = [part for part in str(raw_path or "").split(".") if part]
        if not segments:
            continue
        if segments[0] not in ("params", "method"):
            segments.insert(0, "params")
        if segments[0] == "method" and len(segments) > 1:
            continue

        cursor: Any = result
        for segment in segments[:-1]:
            child = cursor.get(segment)
            if not isinstance(child, dict):
                if value is None:
                    cursor = None
                    break
                child = {}
                cursor[segment] = child
            cursor = child
        if cursor is None:
            continue

        leaf = segments[-1]
        if value is None:
            cursor.pop(leaf, None)
        else:
            cursor[leaf] = value
    return result


def _summarize_payload(payload: dict, limit: int = 1200) -> str:
    """展示请求体结构，但隐藏 instructions 正文，避免挤占上下文。"""
    snapshot = copy.deepcopy(payload)
    message = (snapshot.get("params") or {}).get("message")
    if isinstance(message, dict):
        for part in message.get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part["text"] = f"<instructions：{len(part['text'])} 字符>"
    text = json.dumps(snapshot, ensure_ascii=False, indent=2)
    return text if len(text) <= limit else text[:limit] + "\n…（请求体已截断）"


def _invoke_once(profile: AgentProfile, payload: dict) -> tuple[bool, str, bool]:
    """发送一次请求，返回 (是否成功, 结果/错误文本, 是否值得用 override 重试)。"""
    try:
        response = _post_jsonrpc(profile, payload)
    except A2AError as exc:
        text = str(exc)
        return False, text, bool(_HTTP_4XX_RE.match(text))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", False

    error = response.get("error") if isinstance(response, dict) else None
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message") or "未知错误"
        data = error.get("data")
        detail = (
            f"，详情：{json.dumps(data, ensure_ascii=False, default=str)[:500]}"
            if data else ""
        )
        return False, f"code={code}，{message}{detail}", True
    if not isinstance(response, dict) or "result" not in response:
        return False, "响应缺少 result 字段：" + json.dumps(
            response, ensure_ascii=False
        )[:500], False
    return True, _format_result(response["result"]), False


def _retry_hint(error_text: str, payload: dict, remaining: int) -> str:
    """把请求错误包装成可操作提示，交给模型决定如何覆盖请求字段。"""
    return (
        "⚠️ A2A 调用返回错误：\n"
        f"{error_text}\n\n"
        "本次请求体（instructions 正文已省略）：\n"
        f"{_summarize_payload(payload)}\n\n"
        "如果错误与字段名/参数结构不匹配：保持 instructions 不变，再次调用本工具并用 "
        "custom_payload_overrides 覆盖请求字段。键为字段路径（以 params 为根，可省略 "
        "params.），值为新值；值为 null 表示删除该字段。示例："
        '{"params.configuration.blocking": null, '
        '"params.configuration.returnImmediately": false}。\n'
        "如果错误与请求内容无关（如鉴权失败、连接超时、任务不存在）：不要重试，"
        "直接调整 instructions 或把错误转达给用户。\n"
        f"剩余重试次数：{remaining}。"
    )


def _trim_attempt_cache() -> None:
    if len(_override_attempts) > _ATTEMPT_CACHE_LIMIT:
        _override_attempts.clear()


def invoke_agent(
    profile: AgentProfile,
    instructions: str,
    custom_payload_overrides: Optional[dict] = None,
    attempt_key: str = "",
) -> str:
    """调用远端 Agent，返回可直接交给模型的文本（失败同样以文本形式返回）。"""
    instructions = str(instructions or "").strip()
    if not instructions:
        return "⚠️ A2A 调用失败：instructions 不能为空。"

    if custom_payload_overrides is not None and not isinstance(custom_payload_overrides, dict):
        return "⚠️ A2A 调用失败：custom_payload_overrides 必须是 JSON 对象。"

    key = attempt_key or profile.name
    used = _override_attempts.get(key, 0)
    if custom_payload_overrides and used >= _MAX_OVERRIDE_RETRIES:
        return (
            f"⚠️ A2A 调用失败：custom_payload_overrides 已重试 {_MAX_OVERRIDE_RETRIES} 次仍未成功，"
            "请停止重试，把之前的错误信息如实转达给用户。"
        )

    payload = _build_request(profile, instructions)
    if custom_payload_overrides:
        payload = _apply_overrides(payload, custom_payload_overrides)

    ok, text, retryable = _invoke_once(profile, payload)
    if ok:
        _override_attempts.pop(key, None)
        return text
    if not retryable:
        return f"⚠️ A2A 调用失败（无法通过修改请求字段解决）：{text}"

    if custom_payload_overrides:
        used += 1
        _override_attempts[key] = used
        _trim_attempt_cache()

    remaining = max(0, _MAX_OVERRIDE_RETRIES - used)
    if remaining <= 0:
        return (
            f"⚠️ A2A 调用失败：custom_payload_overrides 已重试 {_MAX_OVERRIDE_RETRIES} 次仍未成功，"
            f"请停止重试并把错误信息转达给用户。\n{text}"
        )
    return _retry_hint(text, payload, remaining)


# ── 工具注册 ──────────────────────────────────────────────────────────

def _tool_description(profile: AgentProfile) -> str:
    """把名片压缩成工具描述：远端定位、调用约定与技能清单。"""
    lines = [
        f"远端 A2A Agent「{profile.name}」：{profile.description or '（未提供描述）'}",
        f"调用约定：协议 {profile.protocol}，端点 {profile.endpoint}。"
        "instructions 必须自包含（远端看不到本平台对话历史）。"
        "若远端返回字段/参数不匹配错误，保持 instructions 不变，"
        "改用 custom_payload_overrides 覆盖请求字段后重试（最多 3 次）。",
    ]
    if profile.skills:
        lines.append("可用技能：")
        for skill in profile.skills[:12]:
            name = skill.get("name") or skill.get("id") or "未命名技能"
            desc = str(skill.get("description") or "").strip()
            tags = skill.get("tags")
            tag_text = f"（标签：{', '.join(map(str, tags))}）" if isinstance(tags, list) and tags else ""
            lines.append(f"- {name}：{desc}{tag_text}")
            examples = skill.get("examples")
            if isinstance(examples, list) and examples:
                lines.append("  示例：" + "；".join(map(str, examples[:3])))
    else:
        lines.append("该 Agent 未声明细分技能，按 description 判断是否适用。")
    description = "\n".join(lines)
    return description[:_DESCRIPTION_LIMIT]


def _make_agent_tool(name: str, profile: AgentProfile):
    """为单个远端 Agent 创建 LangChain 工具。"""
    from langchain_core.tools import StructuredTool

    tool_name = f"a2a__{name}"
    description = _tool_description(profile)

    def call(instructions: str, custom_payload_overrides: Optional[dict] = None) -> str:
        return invoke_agent(
            profile, instructions, custom_payload_overrides, attempt_key=tool_name
        )

    call.__name__ = tool_name.replace("-", "_")
    call.__doc__ = description
    return StructuredTool.from_function(
        func=call,
        name=tool_name,
        description=description,
        args_schema=_A2AArgs,
    )


_registered: dict[str, list[str]] = {}
_last_report: dict[str, dict] = {}


def _unregister_all() -> None:
    from tool_registry import unregister_tool

    for names in _registered.values():
        for tool_name in names:
            unregister_tool(tool_name)
    _registered.clear()


def reload_agents() -> dict[str, dict]:
    """按配置重新注册全部远端 Agent 工具。返回逐 Agent 结果。"""
    from tool_registry import register_tool

    _unregister_all()
    report: dict[str, dict] = {}

    for raw_name, raw in load_agent_config().get(AGENTS_KEY, {}).items():
        name, name_error = validate_agent_name(str(raw_name))
        if name_error:
            report[str(raw_name)] = {"ok": False, "error": name_error, "skills": 0}
            continue

        normalized, errors = validate_agent_config(raw)
        if errors:
            report[name] = {"ok": False, "error": "；".join(errors), "skills": 0}
            continue
        if not normalized.get("enabled", True):
            report[name] = {"ok": True, "error": None, "skills": 0, "disabled": True}
            continue

        card = normalized.get("card")
        if not isinstance(card, dict):
            try:
                card = fetch_agent_card(normalized["url"], normalized["api_key"])
            except A2AError as exc:
                report[name] = {"ok": False, "error": str(exc), "skills": 0}
                continue

        try:
            profile = normalize_card(card, normalized["api_key"], name=name)
        except A2AError as exc:
            report[name] = {"ok": False, "error": str(exc), "skills": 0}
            continue

        tool = _make_agent_tool(name, profile)
        register_tool(tool, f"a2a__{name}", tool.description, "a2a", "moderate")
        _registered[name] = [f"a2a__{name}"]
        report[name] = {
            "ok": True,
            "error": None,
            "skills": len(profile.skills),
            "protocol": profile.protocol,
            "tool_names": [f"a2a__{name}"],
        }

    _last_report.clear()
    _last_report.update(report)
    debug_log(
        "[a2a] reload | "
        + ", ".join(
            f"{key}:{'ok' if value.get('ok') else 'fail'}({value.get('skills', 0)})"
            for key, value in report.items()
        )
    )
    return report


def status() -> dict:
    """返回当前配置与注册状态，供侧边栏展示。"""
    agents: dict[str, dict] = {}
    for name, raw in load_agent_config().get(AGENTS_KEY, {}).items():
        normalized, errors = validate_agent_config(raw)
        report = _last_report.get(str(name), {})
        card = normalized.get("card") if isinstance(normalized.get("card"), dict) else {}
        protocol = normalized.get("protocol_version") or ""
        if not protocol:
            protocol = _detect_protocol(card) if card else "?"
        agents[str(name)] = {
            "url": normalized.get("url") or raw.get("url", ""),
            "enabled": normalized.get("enabled", True),
            "registered": str(name) in _registered,
            "card_name": card.get("name") or "",
            "protocol": protocol,
            "skills": len(card.get("skills") or []) if isinstance(card, dict) else 0,
            "error": errors[0] if errors else report.get("error"),
        }
    return {"config_path": str(A2A_CONFIG_PATH), "agents": agents}


def remove_agent(name: str) -> None:
    """注销内存中已注册的 Agent 工具（配置删除另由调用方负责）。"""
    from tool_registry import unregister_tool

    for tool_name in _registered.pop(name, []):
        unregister_tool(tool_name)
    _override_attempts.pop(f"a2a__{name}", None)
    _last_report.pop(name, None)
