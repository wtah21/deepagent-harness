"""
图装配：主智能体 = 「任务编排者」。

主智能体不自己产出 agent / harness 配置，只负责任务拆解、分发与结果整合：
  - 拆解：用 write_todos 规划子任务；
  - 分发：每个已注册子智能体都是一个独立工具，主智能体按工具描述选型调用，
    并把最小必要上下文、任务目标与期望输出格式传给子智能体；
  - 配置：config_designer 作为仅绑定在主智能体上的工具，生成并入库 agent / harness 配置；
  - 执行：run_harness 让主智能体可以直接执行某条已入库的流水线（也可由用户在界面侧试运行）；
  - 兜底：同一子任务重试 2 次仍失败时，主智能体用自己的通用工具直接完成。

工具注入策略（与 langgraph-bigtool 的「检索 + 动态挂载」一致）：
  - 工具总数 ≤ TOOL_SEARCH_THRESHOLD 时全量绑定；
  - 超过阈值时只绑定常驻工具（核心文件/执行/规划 + 编排工具）与 search_tool，
    检索结果写入 selected_tool_ids，下一轮再动态绑定。
  因为主图需要自定义状态、提示词、上下文压缩与审核闸门，这里手写 StateGraph，
  只复用 bigtool 的检索-挂载模式，而不是直接调用它的 create_agent。
"""
import os
import sqlite3
import threading
from typing import Annotated, Any, Sequence

from langchain_core.messages import AIMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config import (
    CHECKPOINT_DB_PATH, CORE_TOOLS, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    RERANK_WARMUP, TOOL_SEARCH_THRESHOLD, debug_log,
)
from reasoning import ReasoningChatOpenAI
from context_manager import make_compression_hook
from state import DeepAgentState
from prompts import ORCHESTRATOR_PROMPT
from subagents import build_subagent_tools
from config_designer import config_designer
from harness import run_harness
from hitl import ask_user
from tool_registry import get_tools_by_names, get_tools_by_category
from tool_search import (
    SEARCH_TOOL_NAME, build_candidates, build_search_tool, run_search,
    sync_tool_index, warmup_reranker, warmup_search,
)

CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")
checkpointer = SqliteSaver(conn)

model = ReasoningChatOpenAI(
    model=os.getenv("MODEL_NAME", DEFAULT_MODEL),
    temperature=float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE)),
    max_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
    api_key=os.getenv("DEEPSEEK_API_KEY", ""),
    base_url=os.getenv("BASE_URL", ""),
    # 只有模型真正走流式，界面才能实时拿到 reasoning_content 的增量
    streaming=True,
)

# 编排类工具：只在主智能体上，且始终常驻
ORCHESTRATOR_TOOLS = [config_designer, ask_user, run_harness]
ORCHESTRATOR_TOOL_NAMES = {tool.name for tool in ORCHESTRATOR_TOOLS}

_main_agent: Any = None
_main_agent_signature: tuple = ()

# 编译期算好的工具快照，运行期只读，避免每个模型轮次都重建工具对象
_tool_candidates: list[dict[str, str]] = []
_search_tool: Any = None
_retrieval_mode: bool = False
_all_tools_cache: list[Any] = []
_resident_cache: list[Any] = []
_tool_map: dict[str, Any] = {}


def _add_new(left: list[str], right: list[str]) -> list[str]:
    """累加去重的 reducer：检索到的工具名在整个会话内保持可选。"""
    if not left:
        left = []
    if not right:
        return left
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


class MainState(DeepAgentState):
    """主智能体状态：在共享状态上增加工具检索的两个通道。"""
    # 已检索并挂载的工具名（只增不减，避免多轮后工具消失）
    selected_tool_ids: Annotated[list[str], _add_new]
    # 已经检索过的工具名：第二次检索会自动排除，防止重复返回同一批
    tool_search_history: Annotated[list[str], _add_new]


def _all_tools() -> list[Any]:
    """主智能体可用的全部工具（不含 search_tool）。"""
    tools = list(ORCHESTRATOR_TOOLS)
    tools.extend(build_subagent_tools())
    tools.extend(get_tools_by_names(CORE_TOOLS))
    # 用户通过 MCP 接入的外部工具：主智能体直接拥有全部 MCP 工具，
    # 子智能体则按自身配置里的 tools 名单从注册表中取用。
    tools.extend(meta.func for meta in get_tools_by_category("mcp"))
    # 用户通过 A2A 接入的远端 Agent：每个 Agent 一个工具，主智能体直接拥有
    tools.extend(meta.func for meta in get_tools_by_category("a2a"))
    # 同名工具只保留一个（MCP 工具名可能和内置工具撞名）
    unique: dict[str, Any] = {}
    for tool in tools:
        name = getattr(tool, "name", "") or getattr(tool, "__name__", "")
        if name:
            unique.setdefault(name, tool)
    return list(unique.values())


def _resident_tools(all_tools: list[Any]) -> list[Any]:
    """始终注入的工具：核心工具 + 编排工具。"""
    resident_names = ORCHESTRATOR_TOOL_NAMES | set(CORE_TOOLS)
    return [t for t in all_tools if getattr(t, "name", "") in resident_names]


def _compress(state: dict) -> dict:
    """执行上下文压缩钩子，返回它建议的状态更新。"""
    hook = _compression_hook
    try:
        update = hook(state) or {}
    except Exception as exc:  # noqa: BLE001 压缩失败不应阻塞对话
        debug_log(f"[main-agent] 压缩失败：{type(exc).__name__}: {exc}")
        return {}
    # llm_input_messages 只在 create_react_agent 内部有意义，手写图里丢弃
    update.pop("llm_input_messages", None)
    return update


def _messages_for_llm(state: dict, update: dict) -> list:
    """取实际送给模型的消息：优先用压缩结果，否则用现有历史。"""
    raw = list(state.get("messages") or [])
    replacement = update.get("messages")
    if not replacement:
        return raw
    if replacement and isinstance(replacement[0], RemoveMessage):
        return list(replacement[1:])
    return list(replacement)


def _agent_node(state: dict) -> dict:
    """调用模型：绑定常驻工具 + 已检索工具（检索模式下再加 search_tool）。"""
    update = _compress(state)
    messages = _messages_for_llm(state, update)

    if _retrieval_mode:
        selected = set(state.get("selected_tool_ids") or [])
        dynamic = list(_resident_cache)
        dynamic.extend(
            tool for name, tool in _tool_map.items() if name in selected
        )
        tools_for_model = [*_dedupe(dynamic), _search_tool]
    else:
        tools_for_model = _all_tools_cache

    bound = model.bind_tools(tools_for_model)
    response = bound.invoke([SystemMessage(content=ORCHESTRATOR_PROMPT), *messages])

    previous = list(update.get("messages") or [])
    return {"messages": [*previous, response]}


def _dedupe(tools: list[Any]) -> list[Any]:
    seen: dict[str, Any] = {}
    for tool in tools:
        name = getattr(tool, "name", "")
        if name and name not in seen:
            seen[name] = tool
    return list(seen.values())


def _select_tools_node(state: dict) -> dict:
    """执行 search_tool：检索结果写入 selected_tool_ids，供下一轮绑定。"""
    last = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    found_ids: list[str] = []

    for call in getattr(last, "tool_calls", []) or []:
        call_id = str(call.get("id") or "")
        if call.get("name") != SEARCH_TOOL_NAME:
            tool_messages.append(ToolMessage(
                "检索工具需要单独调用：请先只调用 search_tool，拿到结果后再调用具体工具。",
                tool_call_id=call_id,
            ))
            continue

        args = call.get("args") or {}
        query = str(args.get("query") or "").strip()
        exclude = args.get("exclude") or []
        if not isinstance(exclude, list):
            exclude = []
        names, text = run_search(query, exclude, state, model, _tool_candidates)
        found_ids.extend(names)
        tool_messages.append(ToolMessage(text, tool_call_id=call_id))

    return {
        "selected_tool_ids": found_ids,
        "tool_search_history": found_ids,
        "messages": tool_messages,
    }


def _route(state: dict):
    """模型要求检索则走 select_tools，否则执行工具或结束。"""
    last = state["messages"][-1] if state.get("messages") else None
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return END
    if any(call.get("name") == SEARCH_TOOL_NAME for call in last.tool_calls):
        return "select_tools"
    return "tools"


def _compile_main_agent() -> Any:
    """编译主图：agent → (select_tools | tools) → agent。"""
    tool_node = ToolNode(_all_tools_cache, name="tools")

    builder = StateGraph(MainState)
    builder.add_node("agent", _agent_node)
    builder.add_node("select_tools", _select_tools_node)
    builder.add_node("tools", tool_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent", _route, {"select_tools": "select_tools", "tools": "tools", END: END}
    )
    builder.add_edge("select_tools", "agent")
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


# 每次编译时绑定的压缩钩子
_compression_hook = make_compression_hook(
    ORCHESTRATOR_PROMPT,
    model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL),
    max_output_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
)


def _prepare_retrieval(index_tools: list[Any], candidates: list[dict[str, str]]) -> None:
    """后台准备检索能力：模型加载与索引同步并行，最后预热整条检索链路。"""
    if RERANK_WARMUP:
        # 模型加载约 15-20s，和索引同步并行，互不等待
        threading.Thread(target=warmup_reranker, daemon=True).start()
    sync_tool_index(index_tools)
    warmup_search(candidates)


def get_main_agent() -> Any:
    """取主智能体（任务编排者）；工具清单变化时自动重新编译。"""
    global _main_agent, _main_agent_signature
    global _tool_candidates, _search_tool, _retrieval_mode
    global _all_tools_cache, _resident_cache, _tool_map

    all_tools = _all_tools()
    signature = tuple(sorted(f"{t.name}\x00{t.description}" for t in all_tools))
    if _main_agent is not None and signature == _main_agent_signature:
        return _main_agent

    _all_tools_cache = all_tools
    _resident_cache = _resident_tools(all_tools)
    _tool_map = {t.name: t for t in all_tools}
    _retrieval_mode = len(all_tools) > TOOL_SEARCH_THRESHOLD
    resident_names = {t.name for t in _resident_cache}
    _tool_candidates = build_candidates(all_tools, resident_names)
    _search_tool = build_search_tool(all_tools, lambda: model)
    _main_agent_signature = signature

    if _retrieval_mode:
        # 预热与索引同步都放后台：首次启动不因模型加载 / 网络请求卡住界面
        threading.Thread(
            target=_prepare_retrieval,
            args=(
                [t for t in all_tools if t.name not in resident_names],
                list(_tool_candidates),
            ),
            daemon=True,
        ).start()
        debug_log(
            f"[main-agent] compile | retrieval=on tools={len(all_tools)} "
            f"resident={len(_resident_tools(all_tools))}"
        )
    else:
        debug_log(f"[main-agent] compile | retrieval=off tools={len(all_tools)}")

    _main_agent = _compile_main_agent()
    return _main_agent


# 模块加载时先编译一次，保证首次 get_state / update_state 即可用
get_main_agent()
