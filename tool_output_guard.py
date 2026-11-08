"""
工具输出治理：超长工具结果在进入模型上下文之前先压缩。

预算：单个工具结果超过 TOOL_OUTPUT_TOKEN_BUDGET 时触发。分页类工具应在
参数层先控制返回量（ls / read_file / web_search 等已支持），这里只做兜底。

路由：把工具名、调用参数、用户当前目标与结果前 TOOL_OUTPUT_JEV_SAMPLE_CHARS
个字符交给 JEV（TypeSafe System One，choice 问题）判断压缩方式：

  - structured_truncate：结构化截断。保留元数据（总记录数、字段、统计信息）、
    前几条完整记录，让模型既理解数据结构又知道数据规模；适用于列表、目录、
    评论、商品结果这类结构对称的数据。
  - programmatic_filter：程序化过滤。用主模型生成过滤代码，在临时沙箱目录中
    读取原始数据并筛选/聚合，只把精简结果交回主模型；适用于巨量杂乱的原始
    数据、需要条件组合或统计运算、以及关键信息散落各处的大海捞针场景。

JEV 置信度不足或调用失败时，用本地启发式在两种策略之间兜底。
无论走哪条路径，最终结果都会再做一次预算校验，确保不会二次超长。
"""
import csv
import dataclasses
import functools
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from config import (
    DEFAULT_MAX_TOKENS, DEFAULT_MODEL, JEV_API_KEY,
    JEV_BASE_URL, JEV_MODEL, TOOL_OUTPUT_FILTER_TIMEOUT, TOOL_OUTPUT_HEAD_RECORDS,
    TOOL_OUTPUT_JEV_MIN_CONFIDENCE, TOOL_OUTPUT_JEV_SAMPLE_CHARS,
    TOOL_OUTPUT_JEV_TIMEOUT, TOOL_OUTPUT_TOKEN_BUDGET, debug_log,
)
from context_manager import count_text_tokens, message_text

STRUCTURED = "structured_truncate"
PROGRAMMATIC = "programmatic_filter"

# 结构化截断最多扫描的记录数（统计信息基于这段数据）
_STAT_SCAN_LIMIT = 2000
# 单条记录在截断输出中的最大字符数
_RECORD_CHARS = 800
# 过滤代码黑名单：明显越权的模块/调用直接拒绝执行
_BLOCKED_SNIPPETS = (
    "subprocess", "os.system", "os.popen", "socket", "requests",
    "urllib", "shutil", "eval(", "exec(", "__import__",
)

_filter_cache: dict[str, str] = {}
_shared_model: ChatOpenAI | None = None

# JEV 熔断：连续失败（SSL 超时、451 等）后直接走本地启发式，
# 否则每一次超长工具输出都要把超时等满，整条流水线会被拖成龟速。
_JEV_FAILURES_TO_TRIP = 2
_JEV_COOLDOWN_SECONDS = 300
_jev_failures = 0
_jev_disabled_until = 0.0


def _jev_available() -> bool:
    return time.monotonic() >= _jev_disabled_until


def _note_jev_failure(exc: Exception) -> None:
    """记录一次 JEV 失败；连续失败达阈值时熔断一段时间。"""
    global _jev_failures, _jev_disabled_until
    _jev_failures += 1
    if _jev_failures >= _JEV_FAILURES_TO_TRIP:
        _jev_disabled_until = time.monotonic() + _JEV_COOLDOWN_SECONDS
        _jev_failures = 0
        debug_log(
            f"[tool-guard] JEV 连续失败，熔断 {_JEV_COOLDOWN_SECONDS}s，"
            f"期间改用本地启发式（最后一次：{type(exc).__name__}: {exc}）"
        )
        return
    debug_log(f"[tool-guard] JEV 调用失败：{type(exc).__name__}: {exc}")


def _note_jev_success() -> None:
    global _jev_failures
    _jev_failures = 0


def _get_model() -> ChatOpenAI:
    """程序化过滤与语义压缩共用的模型实例（懒加载）。"""
    global _shared_model
    if _shared_model is None:
        _shared_model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", DEFAULT_MODEL),
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("BASE_URL", ""),
        )
    return _shared_model


# ── 入口 ──────────────────────────────────────────────────────────────

def guard_tool_output(
    tool_name: str,
    tool_args: dict,
    output: str,
    state: dict | None = None,
) -> str:
    """把超长工具结果压缩到预算内；未超预算时原样返回。"""
    if not isinstance(output, str) or not output:
        return output

    budget = TOOL_OUTPUT_TOKEN_BUDGET
    if budget <= 0 or count_text_tokens(output) <= budget:
        return output

    user_goal = _extract_user_goal(state)
    strategy = choose_strategy(tool_name, tool_args, output, user_goal)
    debug_log(
        f"[tool-guard] {tool_name} 输出 {len(output)} 字符 / "
        f"≈{count_text_tokens(output)} tokens，策略={strategy}"
    )

    if strategy == PROGRAMMATIC:
        filtered = _programmatic_filter(tool_name, tool_args, output, user_goal)
        if filtered:
            header = (
                f"[程序化过滤] 原始输出 {len(output)} 字符，"
                f"已由沙箱代码按工具调用上下文筛选；结果如下：\n"
            )
            return _finalize(
                header + filtered, tool_name, tool_args, user_goal, budget
            )
        debug_log("[tool-guard] 程序化过滤无结果，回退结构化截断")

    return _finalize(
        structured_truncate(output), tool_name, tool_args, user_goal, budget
    )


def wrap_tool(tool: BaseTool) -> BaseTool:
    """给工具加一层输出守卫，保留原函数签名（InjectedState 注入依赖签名）。"""
    original = getattr(tool, "func", None)
    if original is None or getattr(original, "__tool_output_guarded__", False):
        return tool

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        result = original(*args, **kwargs)
        return _guard_result(tool.name, kwargs, result)

    setattr(guarded, "__tool_output_guarded__", True)
    return tool.model_copy(update={"func": guarded})


def _guard_result(tool_name: str, call_kwargs: dict, result: Any) -> Any:
    """压缩 str 返回值或 Command 中 ToolMessage 的正文。"""
    state = call_kwargs.get("state") if isinstance(call_kwargs, dict) else None
    tool_args = {
        key: value
        for key, value in (call_kwargs or {}).items()
        if key not in ("state", "tool_call_id")
    }

    if isinstance(result, str):
        return guard_tool_output(tool_name, tool_args, result, state)

    if isinstance(result, Command):
        update = result.update if isinstance(result.update, dict) else {}
        messages = update.get("messages") if isinstance(update, dict) else None
        if messages:
            guarded_messages = []
            for message in messages:
                if isinstance(message, ToolMessage) and isinstance(message.content, str):
                    guarded_messages.append(message.model_copy(update={
                        "content": guard_tool_output(tool_name, tool_args, message.content, state),
                    }))
                else:
                    guarded_messages.append(message)
            update = {**update, "messages": guarded_messages}
            return dataclasses.replace(result, update=update)
    return result


def _extract_user_goal(state: dict | None) -> str:
    """取最近一条用户消息作为过滤代码的上下文。"""
    if not isinstance(state, dict):
        return ""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            return message_text(message)[:500]
    return ""


# ── 策略选择：JEV 决策 + 启发式兜底 ──────────────────────────────────

def choose_strategy(
    tool_name: str,
    tool_args: dict,
    output: str,
    user_goal: str = "",
) -> str:
    """先用 JEV 判断；置信度不足时退回本地启发式。"""
    choice, confidence = _ask_jev(tool_name, tool_args, output, user_goal)
    if choice in (STRUCTURED, PROGRAMMATIC) and confidence >= TOOL_OUTPUT_JEV_MIN_CONFIDENCE:
        debug_log(f"[tool-guard] JEV 决策 {choice}（confidence={confidence:.2f}）")
        return choice

    fallback = _heuristic_strategy(output)
    debug_log(
        f"[tool-guard] JEV choice={choice or 'N/A'} confidence={confidence:.2f}，"
        f"改用启发式 {fallback}"
    )
    return fallback


def _ask_jev(
    tool_name: str,
    tool_args: dict,
    output: str,
    user_goal: str,
) -> tuple[str, float]:
    """调用 JEV /v1/systemone 的 choice 问题，返回（策略, 置信度）。"""
    if not JEV_API_KEY or not _jev_available():
        return "", 0.0

    args_text = json.dumps(tool_args, ensure_ascii=False, default=str)[:500]
    state_text = (
        f"Tool name: {tool_name}\n"
        f"Call arguments: {args_text}\n"
        f"User goal: {user_goal or '(unknown)'}\n"
        f"Output size: {len(output)} characters, about {count_text_tokens(output)} tokens\n"
        f"Output head ({TOOL_OUTPUT_JEV_SAMPLE_CHARS} characters):\n"
        f"{output[:TOOL_OUTPUT_JEV_SAMPLE_CHARS]}"
    )
    payload = {
        "model": JEV_MODEL,
        "state": state_text,
        "questions": {
            "strategy": {
                "type": "choice",
                "instructions": (
                    "Decide how to compress this oversized tool output. First look at the "
                    "SHAPE of the data in the sample. If every record has the same fields "
                    "and the records form a uniform list, choose structured_truncate. "
                    "Only choose programmatic_filter when the data is unstructured or "
                    "mixed, or the answer requires scanning/filtering every record."
                ),
                "criteria": {
                    STRUCTURED: (
                        "Uniform record list: JSON array of objects with identical keys, "
                        "CSV/TSV table with a header row, directory listing, search or "
                        "comment list. Truncation keeps record count, fields, aggregate "
                        "numbers and a few full examples, which is enough for the main model."
                    ),
                    PROGRAMMATIC: (
                        "Unstructured or heterogeneous raw text such as server logs, "
                        "dumped files, or nested ad-hoc JSON. The answer requires "
                        "row-by-row filtering, grouping, or locating items scattered "
                        "across the whole payload, so head/tail truncation would lose it."
                    ),
                },
            }
        },
    }

    url = f"{JEV_BASE_URL.rstrip('/')}/systemone"
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {JEV_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=TOOL_OUTPUT_JEV_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        _note_jev_failure(exc)
        return "", 0.0
    _note_jev_success()

    answers = data.get("answers") if isinstance(data, dict) else None
    answer = (answers or data or {}).get("strategy") or {}
    if not isinstance(answer, dict):
        return "", 0.0
    return str(answer.get("choice") or ""), float(answer.get("confidence") or 0.0)


def _heuristic_strategy(text: str) -> str:
    """JEV 不可用时的本地判断。"""
    try:
        json.loads(text)
        return STRUCTURED
    except (ValueError, TypeError):
        pass

    lines = text.splitlines()
    if len(lines) >= 2:
        for delimiter in (",", "\t", "|"):
            if lines[0].count(delimiter) >= 1 and lines[1].count(delimiter) == lines[0].count(delimiter):
                return STRUCTURED

    # 结构化程度低且行数很多：认为需要代码过滤
    if len(lines) > 500:
        return PROGRAMMATIC
    return STRUCTURED


# ── 结构化截断 ────────────────────────────────────────────────────────

def structured_truncate(text: str) -> str:
    """保留元数据、统计信息与前几条完整记录的结构化截断。"""
    stripped = text.lstrip()
    if stripped[:1] in ("{", "["):
        try:
            return _truncate_json(json.loads(text), text)
        except (ValueError, TypeError):
            pass

    delimiter = _detect_delimiter(text)
    if delimiter:
        return _truncate_delimited(text, delimiter)

    return _truncate_lines(text)


def _truncate_json(data: Any, raw: str) -> str:
    lines = [
        f"[结构化截断] 原始输出为 JSON，{len(raw)} 字符。",
    ]

    if isinstance(data, list):
        total = len(data)
        lines.append(f"记录总数：{total}")
        records = data[:_STAT_SCAN_LIMIT]

        if records and all(isinstance(item, dict) for item in records):
            fields = _field_types(records)
            lines.append("字段：" + ", ".join(f"{name}({kind})" for name, kind in fields.items()))
            stats = _record_stats(records)
            if stats:
                lines.append("统计：" + "；".join(stats))
            lines.append(f"前 {min(TOOL_OUTPUT_HEAD_RECORDS, total)} 条完整记录：")
            for record in records[:TOOL_OUTPUT_HEAD_RECORDS]:
                lines.append(_json_line(record))
        elif records:
            stats = _scalar_stats(records)
            if stats:
                lines.append("统计：" + "；".join(stats))
            lines.append(f"前 {min(TOOL_OUTPUT_HEAD_RECORDS, total)} 条：")
            for item in records[:TOOL_OUTPUT_HEAD_RECORDS]:
                lines.append(_json_line(item))

        if total > TOOL_OUTPUT_HEAD_RECORDS:
            lines.append(f"（其余 {total - TOOL_OUTPUT_HEAD_RECORDS} 条已省略）")
    elif isinstance(data, dict):
        lines.append("顶层键：" + ", ".join(str(key) for key in list(data.keys())[:40]))
        for key, value in list(data.items())[:20]:
            if isinstance(value, list):
                lines.append(f"- {key}: 列表，共 {len(value)} 项")
                for item in value[:min(3, TOOL_OUTPUT_HEAD_RECORDS)]:
                    lines.append("  " + _json_line(item))
            elif isinstance(value, dict):
                lines.append(f"- {key}: 对象，键 {list(value.keys())[:15]}")
            else:
                text_value = str(value)
                lines.append(f"- {key}: {text_value[:120]}")
    else:
        lines.append(f"内容：{str(data)[:500]}")

    return "\n".join(lines)


def _field_types(records: list[dict]) -> dict[str, str]:
    """汇总记录中出现的字段及其类型（取首次出现的类型）。"""
    fields: dict[str, str] = {}
    for record in records[:200]:
        if not isinstance(record, dict):
            continue
        for name, value in record.items():
            if name not in fields:
                fields[name] = type(value).__name__
    return fields


def _record_stats(records: list[dict]) -> list[str]:
    """数值字段的最值与均值、低基数字段的取值分布。"""
    stats: list[str] = []
    if not records:
        return stats
    field_names = list(_field_types(records).keys())
    for name in field_names[:15]:
        values = [
            record.get(name) for record in records
            if isinstance(record, dict) and name in record
        ]
        numeric = [
            value for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numeric and len(numeric) >= max(3, len(values) // 2):
            stats.append(
                f"{name}: n={len(numeric)}, min={min(numeric):.4g}, "
                f"max={max(numeric):.4g}, mean={sum(numeric) / len(numeric):.4g}"
            )
            continue
        text_values = [str(value) for value in values if value is not None]
        if text_values and len(set(text_values)) <= 10:
            counter = Counter(text_values).most_common(5)
            stats.append(f"{name} 分布: " + ", ".join(f"{k}={v}" for k, v in counter))
    return stats


def _scalar_stats(values: list) -> list[str]:
    """标量列表的数值统计或取值分布。"""
    numeric = [
        value for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if numeric and len(numeric) == len(values):
        return [
            f"n={len(numeric)}, min={min(numeric):.4g}, max={max(numeric):.4g}, "
            f"mean={sum(numeric) / len(numeric):.4g}"
        ]
    text_values = [str(value) for value in values]
    if len(set(text_values)) <= 10:
        counter = Counter(text_values).most_common(5)
        return ["取值分布: " + ", ".join(f"{k}={v}" for k, v in counter)]
    return []


def _json_line(value: Any) -> str:
    """序列化单条记录，过长时截断。"""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > _RECORD_CHARS:
        text = text[:_RECORD_CHARS] + "…（该条记录过长，已截断）"
    return text


def _detect_delimiter(text: str) -> str:
    """首两行分隔符一致时认为是分隔符文本（CSV/TSV）。"""
    lines = text.splitlines()[:2]
    if len(lines) < 2:
        return ""
    for delimiter in (",", "\t", "|"):
        if lines[0].count(delimiter) >= 1 and lines[1].count(delimiter) == lines[0].count(delimiter):
            return delimiter
    return ""


def _truncate_delimited(text: str, delimiter: str) -> str:
    lines = text.splitlines()
    total = len(lines)
    sample = lines[:_STAT_SCAN_LIMIT]
    rows = list(csv.reader(io.StringIO("\n".join(sample)), delimiter=delimiter))
    header = rows[0] if rows else []
    records = [dict(zip(header, row)) for row in rows[1:] if row]

    output = [f"[结构化截断] 分隔符文本，共 {total} 行，{len(header)} 列，{len(text)} 字符。"]
    output.append("列名：" + ", ".join(header[:30]))
    if records:
        stats = _record_stats(records)
        if stats:
            output.append("统计：" + "；".join(stats))
        output.append(f"前 {min(TOOL_OUTPUT_HEAD_RECORDS, len(records))} 行完整记录：")
        for row in rows[1:1 + TOOL_OUTPUT_HEAD_RECORDS]:
            output.append(delimiter.join(row)[:_RECORD_CHARS])
    if total > TOOL_OUTPUT_HEAD_RECORDS + 1:
        output.append(f"（其余 {total - TOOL_OUTPUT_HEAD_RECORDS - 1} 行已省略）")
    return "\n".join(output)


def _truncate_lines(text: str) -> str:
    """纯文本（如日志）的头部 + 尾部截断，附带规模与级别统计。"""
    lines = text.splitlines()
    total = len(lines)
    head_count = 25
    tail_count = 15
    head = lines[:head_count]
    tail = lines[-tail_count:] if total > head_count + tail_count else []

    output = [
        f"[结构化截断] 纯文本输出，共 {total} 行 / {len(text)} 字符。",
    ]
    level_counts = []
    for level in ("ERROR", "FATAL", "WARN", "WARNING"):
        count = sum(1 for line in lines if level in line)
        if count:
            level_counts.append(f"{level}={count}")
    if level_counts:
        output.append("级别统计（全文）：" + ", ".join(level_counts))

    output.append(f"前 {len(head)} 行：")
    output.extend(head)
    if tail:
        output.append(f"（中间 {total - len(head) - len(tail)} 行已省略）")
        output.append(f"后 {len(tail)} 行：")
        output.extend(tail)
    return "\n".join(output)


def _enforce_budget(text: str, budget_tokens: int) -> str:
    """最终兜底：二分裁剪，确保结果严格不超预算。"""
    if budget_tokens <= 0:
        return text
    if count_text_tokens(text) <= budget_tokens:
        return text

    notice = "\n...（内容已截断以适配上下文预算）"
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_text_tokens(text[:mid] + notice) <= budget_tokens:
            low = mid
        else:
            high = mid - 1

    result = _trim_to_sentence(text[:low]) + notice
    # 预算极小、连提示都放不下时按纯预算硬切
    return result if count_text_tokens(result) <= budget_tokens else text[:low]


def _trim_to_sentence(text: str) -> str:
    """尽量在句末标点处收尾，避免留下半句话。"""
    cut = max((text.rfind(char) for char in "。；！？;.!?\n"), default=-1)
    return text[: cut + 1] if cut >= len(text) // 2 else text


# ── 语义压缩（第三层兜底）─────────────────────────────────────────────

_SEMANTIC_PROMPT = """你是工具结果语义压缩器。主模型即将读取一段工具调用结果，\
但它超出了上下文预算，需要你压缩。压缩后必须仍让主模型清楚：数据从哪来、规模多大、\
哪些内容与当前任务相关。

必须保留：
1. 工具名与调用参数（数据来源与查询范围）；
2. 数据规模与结构（总条数/行数、字段名及类型、层级关系）；
3. 与用户目标直接相关的记录、数值、结论与报错信息；
4. 异常值、失败信号与边界情况；
5. 若内容被省略，写明省略了什么，以及用哪个参数（offset / limit / start_line / page 等）可继续获取。

禁止：
- 编造原始内容中不存在的信息；
- 保留重复记录、无关的样板字段、大段原文；
- 输出开场白，或解释你做了什么。

直接输出压缩结果，总长度不超过 {budget} tokens。"""

_SEMANTIC_USER_TEMPLATE = """工具名：{tool_name}
调用参数：{tool_args}
用户目标：{user_goal}
原始规模：{size} 字符 / 约 {tokens} tokens
内容：
{content}
"""


def _finalize(
    text: str,
    tool_name: str,
    tool_args: dict,
    user_goal: str,
    budget_tokens: int,
) -> str:
    """第三层：前两层压缩后仍超预算时，交给大模型做语义压缩。"""
    if budget_tokens <= 0 or count_text_tokens(text) <= budget_tokens:
        return text

    compressed = _semantic_compress(tool_name, tool_args, user_goal, text, budget_tokens)
    if compressed:
        return _enforce_budget(compressed, budget_tokens)

    debug_log("[tool-guard] 语义压缩不可用，回退硬裁剪")
    return _enforce_budget(text, budget_tokens)


def _semantic_compress(
    tool_name: str,
    tool_args: dict,
    user_goal: str,
    text: str,
    budget_tokens: int,
) -> str:
    prompt = _SEMANTIC_USER_TEMPLATE.format(
        tool_name=tool_name,
        tool_args=json.dumps(tool_args, ensure_ascii=False, default=str)[:500],
        user_goal=user_goal or "（未知，按工具名与参数推断）",
        size=len(text),
        tokens=count_text_tokens(text),
        content=_clip_for_llm(text, budget_tokens * 2),
    )
    try:
        response = _get_model().bind(
            max_tokens=min(4000, max(512, budget_tokens))
        ).invoke([
            ("system", _SEMANTIC_PROMPT.format(budget=budget_tokens)),
            ("human", prompt),
        ])
        result = message_text(response).strip()
    except Exception as exc:
        debug_log(f"[tool-guard] 语义压缩失败：{type(exc).__name__}: {exc}")
        return ""

    if not result:
        return ""
    debug_log(
        f"[tool-guard] 语义压缩 {count_text_tokens(text)} tokens -> "
        f"{count_text_tokens(result)} tokens"
    )
    return result


def _clip_for_llm(text: str, max_tokens: int) -> str:
    """限制喂给压缩模型的输入规模，避免压缩器自己撑爆上下文。"""
    if count_text_tokens(text) <= max_tokens:
        return text
    return text[: max_tokens * 4] + "\n…（输入过长，已预先截断一部分）"


# ── 程序化过滤（programmatic tool calling）────────────────────────────

_FILTER_SYSTEM_PROMPT = """你是数据过滤代码生成器。主模型调用了一个工具，返回结果过长，\
需要你写一段 Python 代码把与任务相关的信息筛出来。

约束：
1. 原始数据在当前目录的 data.txt 中（UTF-8），用 `open("data.txt", encoding="utf-8", errors="replace")` 读取。
2. 只允许使用标准库（json / csv / re / statistics / collections 等），不要联网、不要安装依赖、不要访问其它文件。
3. 根据「用户当前目标」和工具调用参数做筛选、聚合或检索，不要无差别输出全部数据。
4. 只能通过 print 输出最终结果，输出总长度控制在 4000 字符以内。
5. 只输出代码本身，不要 markdown 代码块，不要解释。
"""

_FILTER_USER_TEMPLATE = """工具名称：{tool_name}
调用参数：{tool_args}
用户当前目标：{user_goal}
原始输出规模：{size} 字符
数据样本（开头部分，仅用于判断结构）：
{sample}
"""


def _programmatic_filter(
    tool_name: str,
    tool_args: dict,
    output: str,
    user_goal: str,
) -> str:
    key = hashlib.sha256(
        (tool_name + "\x00" + user_goal + "\x00" + output).encode("utf-8", "replace")
    ).hexdigest()
    if key in _filter_cache:
        return _filter_cache[key]

    sample = output[: max(2000, TOOL_OUTPUT_JEV_SAMPLE_CHARS * 2)]
    code = _generate_filter_code(tool_name, tool_args, sample, user_goal, len(output))
    if not code:
        return ""
    filtered = _run_filter_code(code, output)
    if filtered:
        _filter_cache[key] = filtered
        if len(_filter_cache) > 64:
            for stale in list(_filter_cache)[:32]:
                _filter_cache.pop(stale, None)
    return filtered


def _generate_filter_code(
        tool_name: str,
        tool_args: dict,
        sample: str,
        user_goal: str,
        size: int,
    ) -> str:
    prompt = _FILTER_USER_TEMPLATE.format(
        tool_name=tool_name,
        tool_args=json.dumps(tool_args, ensure_ascii=False, default=str)[:500],
        user_goal=user_goal or "（未知，按工具名与参数推断）",
        size=size,
        sample=sample,
    )
    try:
        response = _get_model().bind(
            max_tokens=min(1600, max(512, DEFAULT_MAX_TOKENS))
        ).invoke([
            ("system", _FILTER_SYSTEM_PROMPT),
            ("human", prompt),
        ])
        code = message_text(response).strip()
    except Exception as exc:
        debug_log(f"[tool-guard] 过滤代码生成失败：{type(exc).__name__}: {exc}")
        return ""

    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code.strip(), flags=re.IGNORECASE)
    if not code:
        return ""
    if any(snippet in code for snippet in _BLOCKED_SNIPPETS):
        debug_log("[tool-guard] 过滤代码包含被禁止的调用，放弃程序化过滤")
        return ""
    return code


def _run_filter_code(code: str, data: str) -> str:
    """在临时目录中执行过滤代码，数据写入 data.txt。"""
    workdir = tempfile.mkdtemp(prefix="tool_guard_")
    try:
        with open(os.path.join(workdir, "data.txt"), "w", encoding="utf-8") as handle:
            handle.write(data)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=workdir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TOOL_OUTPUT_FILTER_TIMEOUT,
        )
        output = (completed.stdout or "").strip()
        if not output:
            debug_log(
                f"[tool-guard] 过滤代码无输出：returncode={completed.returncode} "
                f"stderr={(completed.stderr or '')[:300]}"
            )
            return ""
        return output
    except subprocess.TimeoutExpired:
        debug_log(f"[tool-guard] 过滤代码执行超时（{TOOL_OUTPUT_FILTER_TIMEOUT}s）")
        return ""
    except Exception as exc:
        debug_log(f"[tool-guard] 过滤代码执行失败：{type(exc).__name__}: {exc}")
        return ""
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
