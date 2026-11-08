"""
图装配：主智能体 = 「任务编排者」。

主智能体不自己产出 agent / harness 配置，只负责任务拆解、分发与结果整合：
  - 拆解：用 write_todos 规划子任务；
  - 分发：每个已注册子智能体都是一个独立工具，主智能体按工具描述选型调用，
    并把最小必要上下文、任务目标与期望输出格式传给子智能体；
  - 配置：config_designer 作为仅绑定在主智能体上的工具，生成并入库 agent / harness 配置；
  - 执行：run_harness 让主智能体可以直接执行某条已入库的流水线（也可由用户在界面侧试运行）；
  - 兜底：同一子任务重试 2 次仍失败时，主智能体用自己的通用工具直接完成。
"""
import os
import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

from config import (
    CHECKPOINT_DB_PATH, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    DEFAULT_TOOLS, debug_log,
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

_main_agent: Any = None
_main_agent_signature: tuple = ()


def _build_tools() -> list:
    """主智能体的工具集：配置设计师 + 流水线执行 + 全部子智能体工具 + 兜底通用工具。

    主智能体既可以直接跑已入库的 harness（run_harness），也可以把任务拆给单个子智能体；
    用户仍然可以在界面侧的「Harness Pipeline Workflow」模式里自行试运行。
    """
    tools = [config_designer, ask_user, run_harness]
    tools.extend(build_subagent_tools())
    tools.extend(get_tools_by_names(DEFAULT_TOOLS))
    # 用户通过 MCP 接入的外部工具：主智能体直接拥有全部 MCP 工具，
    # 子智能体则按自身配置里的 tools 名单从注册表中取用。
    tools.extend(meta.func for meta in get_tools_by_category("mcp"))
    # 用户通过 A2A 接入的远端 Agent：每个 Agent 一个工具，主智能体直接拥有
    tools.extend(meta.func for meta in get_tools_by_category("a2a"))
    return tools


def get_main_agent() -> Any:
    """取主智能体（任务编排者）；子智能体清单或职能描述变化时自动重新编译。"""
    global _main_agent, _main_agent_signature
    tools = _build_tools()
    signature = tuple(sorted(f"{t.name}\x00{t.description}" for t in tools))
    if _main_agent is None or signature != _main_agent_signature:
        debug_log(f"[main-agent] compile | tools={sorted(t.name for t in tools)}")
        _main_agent = create_react_agent(
            model,
            tools=tools,
            prompt=ORCHESTRATOR_PROMPT,
            state_schema=DeepAgentState,
            checkpointer=checkpointer,
            # 每次调用模型前检测上下文占用，超阈值时按分层策略压缩模型输入
            pre_model_hook=make_compression_hook(
                ORCHESTRATOR_PROMPT,
                model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL),
                max_output_tokens=int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)),
            ),
        )
        _main_agent_signature = signature
    return _main_agent


# 模块加载时先编译一次，保证首次 get_state / update_state 即可用
get_main_agent()
