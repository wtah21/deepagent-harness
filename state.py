"""
DeepAgent Harness 系统的共享状态定义。
每个节点（主智能体 + 所有子智能体）都读写同一套状态结构。
文件、待办事项和执行日志在所有智能体之间共享。
使用自定义 reducer 以支持在单个步骤中进行并发/多次更新。
"""
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph.message import add_messages
from langgraph.config import get_config
from langchain_core.messages import BaseMessage
from langgraph.managed.is_last_step import RemainingSteps


class Todo(TypedDict):
    """单个计划事项，包含状态跟踪。"""
    content: str
    status: Literal["pending", "in_progress", "completed"]


class AgentConfig(TypedDict):
    """自定义智能体的配置。"""
    name: str
    description: str
    system_prompt: str
    tools: list[str]
    icon: str
    color: str
    model: str
    max_tokens: int
    temperature: float
    is_builtin: bool


class HarnessStage(TypedDict):
    """多智能体 harness 流水线中的单个阶段。"""
    agent_name: str
    instructions: str
    pass_files: bool
    description: str


class HarnessConfig(TypedDict):
    """多智能体工作流 harness 的配置。"""
    name: str
    description: str
    purpose: str
    strategy: Literal["sequential", "parallel", "review_loop", "brainstorming"]
    stages: list[HarnessStage]
    max_review_rounds: int
    icon: str
    color: str


class ExecutionLogEntry(TypedDict):
    """智能体操作的审计追踪条目。"""
    timestamp: str
    agent_name: str
    action: str
    summary: str
    status: Literal["success", "error", "pending"]


class DispatchTask(TypedDict):
    """分发器拆解出的单个子任务。

    每个子智能体节点只读取 agent_name 与自身一致的 instructions，
    因此 instructions 必须是自包含的（该子智能体看不到其它子任务与对话历史）。
    """
    agent_name: str
    instructions: str


class Proposal(TypedDict):
    """单个子智能体在隔离沙盒中的产出。

    子智能体不直接写 files，而是把新增/修改的文件内容放进这里，
    由汇总节点统一做冲突检测、合并，再写回真正的工作区。
    """
    agent_name: str
    instructions: str
    summary: str
    files: dict[str, str]


class ModeratorDecision(TypedDict):
    """头脑风暴主持人的本轮决策。

    主持人节点写入、路由函数读取：next_agent 决定下一位发言人，
    should_end 决定是否终止会议，progress_summary 用于展示会议进程。
    """
    progress_summary: str
    core_disagreement: str
    next_agent: str
    instruction: str
    should_end: bool
    consensus: str
    open_issues: str


# ── 状态通道的 Reducer 函数 ──────────────────────────────────────────

def merge_files(left: dict[str, dict], right: dict[str, dict]) -> dict[str, dict]:
    """Reducer：合并虚拟文件系统的索引更新，避免抛出 InvalidUpdateError。

    files 里存的是索引（大小 / 哈希 / 更新时间），文件内容存在工作区表，
    因此这里的合并开销与文件体积无关，也不会撑大检查点。
    """
    if left is None:
        left = {}
    if right is None:
        return left
    merged = dict(left)
    for key, value in right.items():
        # 值为 None 表示删除该文件索引：普通 dict.update 无法删除键，
        # 侧边栏删除文件后索引会残留，因此这里显式处理删除语义
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def merge_todos(left: list[Todo], right: list[Todo]) -> list[Todo]:
    """待办列表的 Reducer，允许在单个步骤中被多次写入。"""
    if right is not None:
        return right
    return left if left is not None else []


def merge_execution_log(left: list[ExecutionLogEntry], right: list[ExecutionLogEntry]) -> list[ExecutionLogEntry]:
    """审计执行日志条目的 Reducer，允许进行多次更新。"""
    if right is not None:
        return right
    return left if left is not None else []


def merge_tasks_to_dispatch(left: list[DispatchTask], right: list[DispatchTask]) -> list[DispatchTask]:
    """分发任务的 Reducer：分发器整体覆盖上一轮的任务清单。"""
    if right is not None:
        return right
    return left if left is not None else []


def merge_proposals(left: dict[str, Proposal], right: dict[str, Proposal]) -> dict[str, Proposal]:
    """提案的 Reducer：按 agent_name 合并，同一智能体多次提交时合并其文件变更。

    并行分支在同一个 superstep 内各自写 proposals，必须用 reducer 合并而不是互相覆盖。
    """
    if left is None:
        left = {}
    if right is None:
        return left
    merged = dict(left)
    for agent_name, proposal in right.items():
        existing = merged.get(agent_name)
        if not existing:
            merged[agent_name] = proposal
            continue
        files = dict(existing.get("files") or {})
        files.update(proposal.get("files") or {})
        merged[agent_name] = {
            "agent_name": agent_name,
            "instructions": proposal.get("instructions") or existing.get("instructions", ""),
            "summary": proposal.get("summary") or existing.get("summary", ""),
            "files": files,
        }
    return merged


def merge_moderator_decision(left: ModeratorDecision, right: ModeratorDecision) -> ModeratorDecision:
    """主持人决策的 Reducer：每轮决策整体覆盖上一轮。"""
    if right is not None:
        return right
    return left if left is not None else {}


# ── 状态结构定义 ──────────────────────────────────────────────────────

class DeepAgentState(TypedDict):
    """整个 DeepAgent 系统的共享状态。"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    todos: Annotated[list[Todo], merge_todos]
    # 虚拟文件系统的轻量索引：{path: {size, hash, updated_at}}，内容存在工作区表
    files: Annotated[dict[str, dict], merge_files]
    # 当前 project 的工作区标识（即 thread_id），同一工作区内的所有智能体共享文件
    workspace_id: str
    remaining_steps: RemainingSteps
    execution_log: Annotated[list[ExecutionLogEntry], merge_execution_log]
    # parallel 策略：分发器拆解出的子任务清单，各子智能体只读取属于自己的那一条
    tasks_to_dispatch: Annotated[list[DispatchTask], merge_tasks_to_dispatch]
    # parallel 策略：各子智能体在隔离沙盒中的产出，由汇总节点统一裁决后写回 files
    proposals: Annotated[dict[str, Proposal], merge_proposals]
    # brainstorming 策略：主持人最近一次决策（决定下一位发言人 / 是否终止会议）
    moderator_decision: Annotated[ModeratorDecision, merge_moderator_decision]


def workspace_of(state: dict) -> str:
    """取本次运行所属的工作区标识（即项目线程 id）。

    正常情况下由入口直接写入 state；这里再按线程 id 兜一层，
    避免某个入口忘记注入时所有项目退化成共用同一个工作区。
    """
    workspace_id = (state or {}).get("workspace_id")
    if workspace_id:
        return workspace_id
    try:
        return str(get_config().get("configurable", {}).get("thread_id") or "default")
    except Exception:
        return "default"
