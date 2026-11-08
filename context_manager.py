"""
上下文压缩，按调用方提供两种策略（通过 make_compression_hook 的 strategy 参数选择）：

  hybrid（默认，主智能体用）：混合分层压缩。
    按「用户消息」切分轮次（每条 HumanMessage 开启新的一轮），分层处理：
      1. 系统提示词以及首轮用户核心需求：永不压缩，始终放在上下文最前面；
      2. 最近 CONTEXT_RECENT_ROUNDS（默认 5）轮：原文完整保留；
      3. 再往前的 CONTEXT_SUMMARY_ROUNDS - CONTEXT_RECENT_ROUNDS（默认 10）轮：
         调用大模型生成结构化摘要；
      4. 更早的轮次：直接丢弃。
    主智能体的会话天然多轮，按轮次分层能让近期交互保持原文细节。

  summary（子智能体用）：纯摘要压缩。
    不按轮次分层——子智能体整个执行周期只有 1 条 HumanMessage（任务指令），
    按轮次判断会因「轮次太少」而永不触发。只要上下文占用超过窗口的
    CONTEXT_TRIGGER_RATIO（默认 70%），就把「首条任务指令之后」的全部历史
    一次性压成一条结构化摘要（原始目标 / 已完成步骤 / 未完成任务 / 关键数据
    与结论 / 关键决策），首条任务指令原文保留在摘要之前，系统提示词由
    create_react_agent 的 prompt 参数携带，二者都天然不进摘要。

增量压缩（两种策略共用）：上一次压缩生成的 SystemMessage（以 SUMMARY_PREFIX 开头）
是固定的压缩起点，它之前的内容（原始目标 + 已有摘要）不再参与下一次压缩；
下次从它的下一条消息开始收集，已总结过的历史不会被重复送进摘要模型。

触发条件：估算占用（系统提示词 + 消息 + 预留输出）超过模型上下文窗口的
CONTEXT_TRIGGER_RATIO（默认 70%）时才执行压缩，否则原样返回，避免无谓的摘要调用。

接入方式：作为 ``create_react_agent`` 的 ``pre_model_hook``。该节点位于每次
「调用模型」之前（包括每一轮工具调用之后），因此每次请求模型前都会检测一次
上下文占用，等价于用户要求的「定时检测」hook。

压缩结果直接写回 ``state.messages``（用 RemoveMessage 删除被丢弃的消息），
同时把压缩前的完整原始消息异步归档到项目消息表，避免历史原文丢失。
"""
import hashlib
import json
import urllib.request
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.config import get_config
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from config import (
    CONTEXT_RECENT_ROUNDS, CONTEXT_SUMMARY_ROUNDS, CONTEXT_TRIGGER_RATIO,
    CONTEXT_WINDOW_TOKENS, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE,
    debug_log,
)
from message_store import archive_messages

# 摘要消息的固定前缀，便于在模型输入里识别压缩边界
SUMMARY_PREFIX = "【更早对话的压缩摘要】"

# 单条消息写入摘要请求时的截断长度
_SUMMARY_MESSAGE_CHARS = 2000
# 整段摘要请求的输入上限（字符），从最近的消息往前收集
_SUMMARY_INPUT_CHARS = 24000
# 摘要请求自身的输出上限
_SUMMARY_MAX_TOKENS = 1024

# 接口探测不到时使用的兜底窗口大小
_FALLBACK_WINDOW = 128000

# 已知模型的上下文窗口，接口未返回窗口字段时使用
_KNOWN_WINDOWS = {
    "deepseek-chat": 128000,
    "deepseek-reasoner": 128000,
    "deepseek-v3": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4.1": 1047576,
    "claude-3-5-sonnet": 200000,
    "qwen-plus": 131072,
    "qwen-max": 32768,
}

# 探测结果缓存：key 是模型名
_window_cache: dict[str, int] = {}
# 摘要缓存：key 是 (thread_id, 摘要区间指纹)
_summary_cache: dict[tuple[str, str], str] = {}
_summary_model: ChatOpenAI | None = None
_encoder: Any = None


# ── token 估算 ────────────────────────────────────────────────────────

def _get_encoder():
    """取 tiktoken 编码器；不可用时返回 None，退化为字符估算。"""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = False
    return _encoder or None


def count_text_tokens(text: str) -> int:
    """估算一段文本的 token 数。"""
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    # 粗略估算：中文约 1 字 1 token，英文约 4 字符 1 token
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return max(1, (len(text) - ascii_chars) + ascii_chars // 4)


def message_text(message: BaseMessage) -> str:
    """把消息内容统一转成文本（兼容多模态内容块）。"""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "image_url":
                    parts.append("[图片]")
        return "".join(parts)
    return str(content)


def count_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    """估算一批消息的 token 数（含角色开销与工具调用参数）。"""
    total = 0
    for message in messages:
        total += 4
        total += count_text_tokens(message_text(message))
        for call in getattr(message, "tool_calls", None) or []:
            total += count_text_tokens(json.dumps(call, ensure_ascii=False, default=str))
    return total


# ── 上下文窗口探测 ────────────────────────────────────────────────────

def _extract_window(entry: dict) -> int:
    """从接口返回的单个模型条目里找出上下文窗口字段。"""
    keys = (
        "context_window", "context_length", "max_context_length",
        "max_model_len", "max_input_tokens", "input_token_limit",
        "max_tokens",
    )
    for key in keys:
        value = entry.get(key)
        if isinstance(value, int) and value > 0:
            return value
    for value in entry.values():
        if isinstance(value, dict):
            found = _extract_window(value)
            if found:
                return found
    return 0


def _probe_context_window(model_name: str) -> int:
    """调用 OpenAI 兼容的 /models 接口，尝试读取模型上下文窗口。"""
    import os

    base_url = (os.getenv("BASE_URL") or "").rstrip("/")
    if not base_url:
        return 0
    url = base_url if base_url.endswith("/models") else f"{base_url}/models"
    headers = {"Accept": "application/json"}
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        debug_log(f"[context] 窗口探测失败 {url}: {type(exc).__name__}: {exc}")
        return 0

    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or entry.get("model") or "")
        if model_id == model_name or model_id.split("/")[-1] == model_name:
            return _extract_window(entry)
    return 0


def resolve_context_window(model_name: str) -> int:
    """取模型上下文窗口：显式配置 > 探测接口 > 已知模型表 > 兜底值。"""
    if CONTEXT_WINDOW_TOKENS > 0:
        return CONTEXT_WINDOW_TOKENS
    if model_name in _window_cache:
        return _window_cache[model_name]

    window = _probe_context_window(model_name)
    source = "接口探测"
    if not window:
        window = _KNOWN_WINDOWS.get(model_name) or _KNOWN_WINDOWS.get(model_name.split("/")[-1]) or 0
        source = "内置映射"
    if not window:
        window = _FALLBACK_WINDOW
        source = "兜底默认"

    _window_cache[model_name] = window
    debug_log(f"[context] 模型 {model_name} 上下文窗口={window}（来源：{source}）")
    return window


# ── 工具调用配对修复 ──────────────────────────────────────────────────

_INTERRUPTED_TOOL_NOTICE = (
    "⚠️ 工具 `{name}` 的执行在上一轮被中断，没有产生任何结果。"
    "请不要假设该调用已执行或已成功，如需其结果请重新发起调用。"
)


def repair_tool_call_pairs(
    messages: Sequence[BaseMessage],
) -> tuple[list[BaseMessage], list[str]]:
    """修复 tool_calls 与 ToolMessage 的配对，返回（修复后的消息, 补齐的工具名）。

    OpenAI 要求带 tool_calls 的 assistant 消息后面紧跟响应每个 tool_call_id 的 tool 消息。
    执行中断后补写的 ToolMessage 若被追加到会话末尾（中间隔着新的用户消息），这段历史
    就再也无法通过接口校验，后续每次调用都会 400，表现为整个会话卡死。
    这里按原顺序重建：缺失的补一条中断占位结果、重复或错位的丢掉、孤立的 ToolMessage 移除。
    """
    repaired: list[BaseMessage] = []
    filled: list[str] = []
    index = 0
    total = len(messages)

    while index < total:
        message = messages[index]
        index += 1

        # 没有前置 tool_calls 的孤立工具结果：接口会直接拒绝，丢弃
        if isinstance(message, ToolMessage):
            continue
        repaired.append(message)

        calls = getattr(message, "tool_calls", None) or []
        if not isinstance(message, AIMessage) or not calls:
            continue

        call_ids = [str(call.get("id") or "") for call in calls]
        # 只认紧跟在 assistant 消息之后的工具结果，其余位置的都属于错位
        pending: list[ToolMessage] = []
        while index < total and isinstance(messages[index], ToolMessage):
            pending.append(messages[index])
            index += 1

        answered: dict[str, ToolMessage] = {}
        for tool_message in pending:
            call_id = str(getattr(tool_message, "tool_call_id", "") or "")
            if call_id in call_ids and call_id not in answered:
                answered[call_id] = tool_message

        for call, call_id in zip(calls, call_ids):
            if call_id and call_id in answered:
                repaired.append(answered[call_id])
                continue
            name = str(call.get("name") or "tool")
            repaired.append(
                ToolMessage(
                    content=_INTERRUPTED_TOOL_NOTICE.format(name=name),
                    tool_call_id=call_id,
                    name=name,
                )
            )
            filled.append(name)

    return repaired, filled


# ── 轮次切分 ──────────────────────────────────────────────────────────

def split_rounds(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    """按用户消息切分轮次：每条 HumanMessage 开启新的一轮。"""
    rounds: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage) and current:
            rounds.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        rounds.append(current)
    return rounds


def _render_message(message: BaseMessage) -> str:
    """把单条消息渲染成摘要请求里的一行文本。"""
    if isinstance(message, HumanMessage):
        role = "用户"
    elif isinstance(message, AIMessage):
        role = "助手"
    elif isinstance(message, ToolMessage):
        role = f"工具结果({getattr(message, 'name', '') or 'tool'})"
    elif isinstance(message, SystemMessage):
        role = "系统"
    else:
        role = getattr(message, "type", "消息")

    text = message_text(message).strip()
    if len(text) > _SUMMARY_MESSAGE_CHARS:
        text = text[:_SUMMARY_MESSAGE_CHARS] + "…（已截断）"
    line = f"[{role}] {text}"

    calls = getattr(message, "tool_calls", None) or []
    if calls:
        rendered = []
        for call in calls:
            name = call.get("name", "tool") if isinstance(call, dict) else "tool"
            args = call.get("args", {}) if isinstance(call, dict) else {}
            rendered.append(f"{name}({json.dumps(args, ensure_ascii=False, default=str)[:200]})")
        line += "\n  → 调用工具: " + "; ".join(rendered)
    return line


def _render_zone(messages: Sequence[BaseMessage], input_chars: int = _SUMMARY_INPUT_CHARS) -> str:
    """把摘要区间的消息渲染成文本；超长时优先保留更近的消息。"""
    lines: list[str] = []
    used = 0
    for message in reversed(messages):
        line = _render_message(message)
        if used + len(line) > input_chars:
            lines.append("（更早的内容因长度限制已省略）")
            break
        lines.append(line)
        used += len(line)
    return "\n".join(reversed(lines))


SUMMARY_SYSTEM_PROMPT = (
    "你是对话上下文压缩器。请把给定的历史对话压缩成结构化摘要，"
    "供后续继续完成同一任务时使用。只输出下面五个小节，内容必须来自对话，禁止编造："
)

SUMMARY_TEMPLATE = """请按以下模板压缩这段历史对话：

## 用户的原始目标
（用户最初想达成什么，保持原始意图不变）

## 已完成的步骤
（按顺序列出已经做完的关键动作，含涉及的文件、工具、子任务）

## 未完成的任务
（还没做完、待验证或待补充的事项）

## 关键数据与结论
（重要的变量、路径、接口、数值、错误信息、验证结果）

## 关键决策
（已经确定的技术选型、方案取舍和原因）

历史对话：
{conversation}
"""


def _get_summary_model() -> ChatOpenAI:
    """摘要用的大模型：低温、限制输出长度。"""
    global _summary_model
    if _summary_model is None:
        import os

        _summary_model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", DEFAULT_MODEL),
            temperature=0,
            max_tokens=min(_SUMMARY_MAX_TOKENS, max(256, DEFAULT_MAX_TOKENS)),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("BASE_URL", ""),
        )
    return _summary_model


def _rule_based_summary(zone: Sequence[BaseMessage]) -> str:
    """摘要模型不可用时的兜底：抽取每条消息的首句，保证压缩流程不中断。"""
    goals: list[str] = []
    steps: list[str] = []
    results: list[str] = []
    for message in zone:
        text = message_text(message).strip().replace("\n", " ")
        if not text:
            continue
        excerpt = text[:160]
        if isinstance(message, HumanMessage) and not goals:
            goals.append(excerpt)
        elif isinstance(message, ToolMessage):
            results.append(f"- 工具 {getattr(message, 'name', 'tool')}: {excerpt}")
        elif isinstance(message, AIMessage):
            steps.append(f"- {excerpt}")
    return "\n".join([
        SUMMARY_PREFIX + "（规则兜底生成）",
        "## 用户的原始目标",
        goals[0] if goals else "（未识别）",
        "## 已完成的步骤",
        "\n".join(steps[-10:]) or "（无）",
        "## 未完成的任务",
        "（请参考最近对话继续推进）",
        "## 关键数据与结论",
        "\n".join(results[-10:]) or "（无）",
        "## 关键决策",
        "（无）",
    ])


def _zone_fingerprint(zone: Sequence[BaseMessage]) -> str:
    """摘要区间的缓存指纹：区间长度 + 全部消息内容的哈希。"""
    if not zone:
        return ""
    digest = hashlib.sha256()
    for message in zone:
        digest.update((getattr(message, "id", "") or "").encode("utf-8", errors="replace"))
        digest.update(message_text(message).encode("utf-8", errors="replace"))
        digest.update(str(getattr(message, "tool_calls", "")).encode("utf-8", errors="replace"))
    return f"{len(zone)}:{digest.hexdigest()[:16]}"


def _current_thread_id() -> str:
    """取当前运行的线程 id，用于隔离不同会话的摘要缓存。"""
    try:
        return str(get_config().get("configurable", {}).get("thread_id") or "default")
    except Exception:
        return "default"


def summarize_zone(
    zone: Sequence[BaseMessage],
    thread_id: str,
    input_chars: int = _SUMMARY_INPUT_CHARS,
) -> str:
    """对摘要区间生成结构化摘要，带进程内缓存。

    input_chars 控制送进摘要模型的最大字符数：混合分层的摘要区间只有约 10 轮，
    默认值足够；纯摘要策略的区间是「自上次摘要以来的全部历史」，需要调用方放宽。
    """
    fingerprint = _zone_fingerprint(zone)
    key = (thread_id, fingerprint)
    if not fingerprint:
        return ""
    cached = _summary_cache.get(key)
    if cached:
        return cached

    conversation = _render_zone(zone, input_chars)
    summary = ""
    try:
        response = _get_summary_model().invoke([
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=SUMMARY_TEMPLATE.format(conversation=conversation)),
        ])
        summary = message_text(response).strip()
    except Exception as exc:
        debug_log(f"[context] 摘要模型调用失败：{type(exc).__name__}: {exc}")
        summary = ""

    if not summary:
        summary = _rule_based_summary(zone)
    else:
        summary = f"{SUMMARY_PREFIX}\n{summary}"

    _summary_cache[key] = summary
    # 简单控制缓存规模，避免长时间运行的进程无限增长
    if len(_summary_cache) > 256:
        for stale in list(_summary_cache)[:128]:
            _summary_cache.pop(stale, None)
    return summary


# ── 压缩主流程 ────────────────────────────────────────────────────────

def _find_summary_index(messages: Sequence[BaseMessage]) -> int:
    """定位上一次压缩生成的摘要消息（没有则返回 -1）。"""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, SystemMessage) and message_text(message).startswith(SUMMARY_PREFIX):
            return index
    return -1


def _merge_summary(previous: str, new: str) -> str:
    """把上一轮摘要与新摘要合并成一条；旧摘要文本原样保留，不重新送模型总结。"""
    if not previous:
        return new
    if not new:
        return previous
    body = new[len(SUMMARY_PREFIX):].lstrip("\n") if new.startswith(SUMMARY_PREFIX) else new
    return f"{previous.rstrip()}\n\n{SUMMARY_PREFIX}（续）\n{body}"


def compress_messages(
    messages: Sequence[BaseMessage],
    system_prompt: str,
    model_name: str,
    max_output_tokens: int,
    thread_id: str,
) -> tuple[list[BaseMessage], dict]:
    """按分层策略压缩消息，返回（压缩后的完整消息列表, 诊断信息）。

    未触发压缩时返回的列表与入参一致，调用方据此判断是否需要更新 state。
    """
    rounds = split_rounds(messages)
    window = resolve_context_window(model_name)
    used = (
        count_text_tokens(system_prompt)
        + count_messages_tokens(messages)
        + max(0, max_output_tokens)
    )
    stats = {
        "rounds": len(rounds),
        "tokens": used,
        "window": window,
        "ratio": round(used / window, 3) if window else 0.0,
        "triggered": False,
        "summarized_rounds": 0,
        "dropped_rounds": 0,
    }

    # 未超阈值或轮次不足以压缩时，原样返回完整历史
    if window <= 0 or used <= window * CONTEXT_TRIGGER_RATIO:
        return list(messages), stats
    if len(rounds) <= CONTEXT_RECENT_ROUNDS:
        return list(messages), stats

    # 增量起点：上一次摘要及其之前的内容永不重复压缩
    summary_index = _find_summary_index(messages)
    if summary_index >= 0:
        prefix = list(messages[:summary_index])
        previous_summary = message_text(messages[summary_index])
        previous_summary_id = getattr(messages[summary_index], "id", None)
        body = list(messages[summary_index + 1:])
    else:
        # 首次压缩：第 1 轮是「用户的核心需求」，固定保留在摘要之前
        prefix = list(rounds[0])
        previous_summary = ""
        previous_summary_id = None
        body = [message for round_messages in rounds[1:] for message in round_messages]

    body_rounds = split_rounds(body)
    if len(body_rounds) <= CONTEXT_RECENT_ROUNDS:
        # 剩下的全在保留窗口内，没有可压缩的轮次
        return list(messages), stats

    # 最近 5 轮完整保留，其余按 5-15 轮摘要、15 轮前丢弃
    summary_span = max(0, CONTEXT_SUMMARY_ROUNDS - CONTEXT_RECENT_ROUNDS)
    recent_rounds = body_rounds[-CONTEXT_RECENT_ROUNDS:]
    older_rounds = body_rounds[:-CONTEXT_RECENT_ROUNDS]
    summary_rounds = older_rounds[-summary_span:] if summary_span else []
    drop_rounds = older_rounds[: len(older_rounds) - len(summary_rounds)] if summary_rounds else older_rounds

    zone = [message for round_messages in summary_rounds for message in round_messages]
    new_summary = summarize_zone(zone, thread_id) if zone else ""
    merged_summary = _merge_summary(previous_summary, new_summary)

    if not merged_summary and not drop_rounds:
        return list(messages), stats

    compressed: list[BaseMessage] = list(prefix)
    if merged_summary:
        compressed.append(SystemMessage(content=merged_summary, id=previous_summary_id))
    for round_messages in recent_rounds:
        compressed.extend(round_messages)

    stats["triggered"] = True
    stats["summarized_rounds"] = len(summary_rounds)
    stats["dropped_rounds"] = len(drop_rounds)
    stats["after_tokens"] = (
        count_text_tokens(system_prompt)
        + count_messages_tokens(compressed)
        + max(0, max_output_tokens)
    )

    debug_log(
        "[context] 触发压缩 | "
        f"rounds={stats['rounds']} tokens={used}->{stats['after_tokens']}/{window} "
        f"kept={len(recent_rounds)}+core summarized={len(summary_rounds)} "
        f"dropped={len(drop_rounds)} incremental={summary_index >= 0}"
    )
    return compressed, stats


# 纯摘要策略（子智能体）送摘要模型的输入预算：区间是「自上次摘要以来的全部历史」，
# 远大于混合分层的约 10 轮，按主智能体的默认预算会丢掉大量早期步骤
_SUMMARY_ZONE_INPUT_CHARS = 60000


def compress_messages_summary(
    messages: Sequence[BaseMessage],
    system_prompt: str,
    model_name: str,
    max_output_tokens: int,
    thread_id: str,
) -> tuple[list[BaseMessage], dict]:
    """纯摘要压缩（子智能体专用）：占用超阈值时把历史整体压成一条结构化摘要。

    与混合分层的差别：不按轮次分层，也没有「最近 N 轮保留原文」「更早直接丢弃」
    两层——子智能体整个执行周期只有 1 条 HumanMessage，按轮次判断会永不触发。

    压缩后保留：
      - 首条任务指令（用户原始目标）原文，始终放在上下文最前面；
      - 一条结构化摘要（SUMMARY_TEMPLATE 五个小节，覆盖原始目标、已完成步骤、
        未完成任务、关键数据与结论、关键决策）；
      - 系统提示词由 create_react_agent 的 prompt 参数携带，天然不占消息历史。
    """
    window = resolve_context_window(model_name)
    used = (
        count_text_tokens(system_prompt)
        + count_messages_tokens(messages)
        + max(0, max_output_tokens)
    )
    stats = {
        "tokens": used,
        "window": window,
        "ratio": round(used / window, 3) if window else 0.0,
        "triggered": False,
        "summarized_rounds": 0,
        "dropped_rounds": 0,
    }

    # 未超阈值时原样返回完整历史
    if window <= 0 or used <= window * CONTEXT_TRIGGER_RATIO:
        return list(messages), stats

    # 增量起点：上一次摘要及其之前的内容（含原始目标）永不重复压缩
    summary_index = _find_summary_index(messages)
    if summary_index >= 0:
        prefix = list(messages[:summary_index])
        previous_summary = message_text(messages[summary_index])
        previous_summary_id = getattr(messages[summary_index], "id", None)
        zone = list(messages[summary_index + 1:])
    else:
        # 首次压缩：首条任务指令原文保留，其余全部送摘要
        first_user = next(
            (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), 0
        )
        prefix = list(messages[: first_user + 1])
        previous_summary = ""
        previous_summary_id = None
        zone = list(messages[first_user + 1:])

    if not zone:
        return list(messages), stats

    new_summary = summarize_zone(zone, thread_id, input_chars=_SUMMARY_ZONE_INPUT_CHARS)
    merged_summary = _merge_summary(previous_summary, new_summary)
    if not merged_summary:
        return list(messages), stats

    compressed: list[BaseMessage] = list(prefix)
    compressed.append(SystemMessage(content=merged_summary, id=previous_summary_id))

    stats["triggered"] = True
    stats["summarized_rounds"] = len(zone)
    stats["after_tokens"] = (
        count_text_tokens(system_prompt)
        + count_messages_tokens(compressed)
        + max(0, max_output_tokens)
    )

    debug_log(
        "[context] 触发纯摘要压缩 | "
        f"tokens={used}->{stats['after_tokens']}/{window} "
        f"zone={len(zone)}条 incremental={summary_index >= 0}"
    )
    return compressed, stats


def make_compression_hook(
    system_prompt: str,
    model_name: str = "",
    max_output_tokens: int = DEFAULT_MAX_TOKENS,
    archive: bool = True,
    strategy: str = "hybrid",
):
    """生成 pre_model_hook：每次调用模型前归档消息并检测上下文占用。

    压缩结果直接写回 ``state.messages``；``llm_input_messages`` 显式置空，
    让模型使用压缩后的 ``messages``（同时清掉旧版本可能残留的该字段）。

    archive=False 时不写项目会话表：子智能体收到的任务指令与中间过程
    只服务于它自己的推理，归档后会在对话界面上多出成串的用户/助手气泡。

    strategy="hybrid"（默认）用混合分层压缩（主智能体：多轮会话按轮次分层），
    strategy="summary" 用纯摘要压缩（子智能体：单轮任务指令，超阈值整体摘要）。
    """
    import os

    resolved_model = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL)
    compress = compress_messages if strategy == "hybrid" else compress_messages_summary

    def _hook(state: dict) -> dict:
        messages = list(state.get("messages") or [])
        workspace_id = str(state.get("workspace_id") or _current_thread_id())

        # 压缩会删掉旧消息，先在这一刻把完整原文异步归档到项目数据库
        if archive:
            archive_messages(workspace_id, messages)

        # 兜底修复：任何来源（中断、补写、压缩边界）造成的 tool_calls 未配对，
        # 都会让模型接口返回 400 并让会话永久卡死，这里统一校正后再送模型
        repaired, filled = repair_tool_call_pairs(messages)
        update: dict = {"llm_input_messages": []}
        if filled:
            messages = repaired
            update["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + messages
            debug_log(f"[context] 补齐中断的工具调用：{filled}")

        compressed, stats = compress(
            messages,
            system_prompt=system_prompt,
            model_name=resolved_model,
            max_output_tokens=max_output_tokens,
            thread_id=workspace_id,
        )

        if not stats["triggered"] or len(compressed) == len(messages):
            return update

        # REMOVE_ALL_MESSAGES 让 add_messages 用压缩结果整体替换历史，
        # 同时保证摘要消息落在核心需求之后、最近轮次之前
        update["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + compressed
        debug_log(
            f"[context] state.messages {len(messages)} -> {len(compressed)} 条"
        )
        return update

    return _hook


def clear_caches() -> None:
    """清空窗口探测与摘要缓存（测试或切换模型时使用）。"""
    _window_cache.clear()
    _summary_cache.clear()
