"""
DeepAgent Harness 的动态智能体工厂与子智能体工具。

AgentFactory 按配置（数据库或内置模板）即时编译出 LangGraph ReAct 智能体；
每个已注册子智能体再被包装成一个独立工具暴露给主智能体，工具描述采用
Google 风格 docstring（固定句式 + 职能介绍 + Args + Returns），便于模型选型。
"""
import os
import datetime
from typing import Annotated, Any, Dict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command

from state import DeepAgentState, workspace_of
from context_manager import make_compression_hook
from config import SUBAGENT_RECURSION_LIMIT
from persistence import AgentStore
from prompts import BUILTIN_TEMPLATES
from tool_registry import get_tools_by_names

load_dotenv()


class AgentFactory:
    """带缓存的工厂，用于按配置编译 LangGraph 智能体。"""

    _cache: Dict[str, Any] = {}

    @classmethod
    def get_agent(cls, agent_name: str) -> Any:
        """按名称获取或编译智能体：依次查询缓存、数据库配置，最后回退到内置模板。"""
        # 检查缓存
        if agent_name in cls._cache:
            return cls._cache[agent_name]

        # 获取配置
        config = AgentStore.get(agent_name)
        if not config:
            config = BUILTIN_TEMPLATES.get(agent_name)

        if not config:
            raise ValueError(f"Agent '{agent_name}' is not registered in the database or built-ins.")

        # 解析工具函数
        tools_list = get_tools_by_names(config.get("tools", []))

        # 构建 LLM
        model_name = config.get("model") or os.getenv("MODEL_NAME", "deepseek-v4-flash")
        temperature = config.get("temperature", 0.1)
        max_tokens = config.get("max_tokens", 8000)
        api_key = os.getenv("DEEPSEEK_API_KEY", "")

        url = os.getenv("BASE_URL", "")
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=url,
            # 走流式：执行线程才能在模型逐字生成的过程中检查「用户是否点了停止」，
            # 否则要等整段生成结束（实测可达一分钟）才轮到下一个事件边界
            streaming=True,
        )

        # 创建智能体
        agent = create_react_agent(
            llm,
            tools=tools_list,
            prompt=config.get("system_prompt", ""),
            state_schema=DeepAgentState,
            # 子智能体用纯摘要压缩：整个执行周期只有 1 条 HumanMessage（任务指令），
            # 混合分层按轮次判断会因「轮次太少」而永不触发；主智能体保持混合分层
            pre_model_hook=make_compression_hook(
                config.get("system_prompt", ""),
                model_name=model_name,
                max_output_tokens=max_tokens,
                archive=False,
                strategy="summary",
            ),
        )

        cls._cache[agent_name] = agent
        return agent

    @classmethod
    def clear_cache(cls):
        """清空已缓存的智能体，以便加载新的配置。"""
        cls._cache.clear()


# ── 子智能体工具：每个已注册智能体一个独立工具 ────────────────────────

def _registered_agents() -> Dict[str, dict]:
    """合并数据库中的智能体与内置模板（同名时以已有配置为准）。"""
    seen = {cfg["name"]: cfg for cfg in AgentStore.list_all()}
    for name, cfg in BUILTIN_TEMPLATES.items():
        seen.setdefault(name, cfg)
    return seen


def _tool_description(agent_name: str, role: str) -> str:
    """按 Google 风格 docstring 组织工具描述：固定句式 + 职能 + Args + Returns。"""
    return (
        f"把请求交给 {agent_name} 处理：{role}\n\n"
        "Args:\n"
        f"    instructions: 交给 {agent_name} 的完整自包含任务描述，必须包含三部分——"
        "完成该任务所必需的最小上下文、清晰的任务目标，以及期望的输出格式。"
        "子智能体看不到本次对话历史，缺失的上下文必须在 instructions 中补齐。\n\n"
        "Returns:\n"
        f"    {agent_name} 完成任务后的结论摘要（纯文本），它产出的文件已写回共享工作区。"
    )


def _make_subagent_tool(agent_name: str, role: str) -> BaseTool:
    """把单个子智能体包装成工具：入参是自包含的任务描述，输出是子智能体的结论。"""

    def _runner(
            instructions: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict, InjectedState],
        ) -> Command:
        """调用子智能体并把它产出的文件、结论写回共享状态。"""
        try:
            agent = AgentFactory.get_agent(agent_name)
        except ValueError as e:
            return Command(update={"messages": [ToolMessage(
                content=f"⚠️ 无法解析子智能体 {agent_name}：{e}",
                tool_call_id=tool_call_id,
            )]})

        try:
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=instructions)],
                    "files": state.get("files", {}),
                    "workspace_id": workspace_of(state),
                    "todos": state.get("todos", []),
                    "execution_log": state.get("execution_log", []),
                },
                # 不传 config 的话 ReAct 循环只有 LangGraph 默认的 25 步
                # （约 12 轮工具调用），稍复杂的子任务就会步数耗尽
                config={"recursion_limit": SUBAGENT_RECURSION_LIMIT},
            )
        except Exception as e:
            return Command(update={"messages": [ToolMessage(
                content=(
                    f"⚠️ 子智能体 {agent_name} 执行失败：{type(e).__name__}: {e}。\n"
                    "本次调用没有产出任何结果，请不要假设它已完成；可修正 instructions 后重试"
                    "（同一子任务最多重试 2 次），仍失败则由你自己用通用工具兜底完成这部分工作。"
                ),
                tool_call_id=tool_call_id,
            )]})

        final_reply = result["messages"][-1].content
        # 子智能体的内部消息（收到的任务指令、中间的思考文本）一律不进项目会话表：
        # 任务指令本来就在主智能体的 tool_calls 参数里，结论也由下面的 ToolMessage 带回，
        # 把它们归档会让 UI 多出成串的「用户气泡 + 助手思考气泡」。

        # 审计日志条目
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "agent_name": agent_name,
            "action": "subagent_tool",
            "summary": f"Executed task: {instructions[:100]}...",
            "status": "success",
        }
        current_logs = list(state.get("execution_log", []))
        current_logs.append(log_entry)

        return Command(update={
            "files": result.get("files", state.get("files", {})),
            "todos": result.get("todos", state.get("todos", [])),
            "execution_log": current_logs,
            "messages": [ToolMessage(f"[{agent_name}] {final_reply}", tool_call_id=tool_call_id)],
        })

    return StructuredTool.from_function(
        func=_runner,
        name=agent_name,
        description=_tool_description(agent_name, role),
    )


def build_subagent_tools() -> list[BaseTool]:
    """为每个已注册智能体（内置 + 自定义）生成一个独立工具，供主智能体调用。"""
    return [
        _make_subagent_tool(cfg["name"], cfg.get("description", ""))
        for cfg in _registered_agents().values()
    ]


# ── 初始化 ────────────────────────────────────────────────────────────

def initialize_builtin_agents():
    """把内置智能体模板同步进数据库。

    模板里的 system_prompt / tools 一改动就需要在已有环境生效，因此同名记录只要是
    内置配置（is_builtin 为真）就整体覆盖；用户自定义的智能体不受影响。
    """
    for name, config in BUILTIN_TEMPLATES.items():
        existing = AgentStore.get(name)
        if existing is None or existing.get("is_builtin"):
            if existing != config:
                AgentStore.save(config)


# 执行数据库填充
initialize_builtin_agents()
