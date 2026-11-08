"""
DeepAgent 的 Harness 引擎。
通过动态创建 LangGraph StateGraph 来编译并执行多智能体工作流
（顺序流水线、并行扇出、评审循环、群组头脑风暴）。
"""
import asyncio
import copy
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

try:  # LangGraph 未公开该常量，缺失时用同值字符串兜底
    from langgraph._internal._constants import CONFIG_KEY_CHECKPOINTER
except ImportError:  # pragma: no cover
    CONFIG_KEY_CHECKPOINTER = "__pregel_checkpointer"

from config import (
    DEFAULT_RECURSION_LIMIT, SUBAGENT_RECURSION_LIMIT,
    DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    PROJECT_ROOT, debug_log,
)
from state import DeepAgentState, workspace_of
from subagents import AgentFactory
from persistence import AgentStore, HarnessStore, WorkspaceStore
from prompts import BUILTIN_TEMPLATES

# harness 内部嵌套图（harness 子图及其子智能体）统一使用的执行配置。
# 子智能体是 ReAct 循环，步数需求远超 LangGraph 默认的 25 步上限。
#
# 同时显式声明「不继承父图的 checkpointer」：主图用的是同步 SqliteSaver，
# 而 parallel 策略要并发执行必须在异步上下文里跑，同步 saver 没有异步实现，
# 一旦被继承就会抛 NotImplementedError。harness 子图本身不需要跨轮恢复，
# 禁用继承后它与主线程的检查点完全隔离，互不覆盖。
_NESTED_GRAPH_CONFIG = {
    "recursion_limit": DEFAULT_RECURSION_LIMIT,
    "configurable": {CONFIG_KEY_CHECKPOINTER: None},
}

# 子智能体专用配置：比外层图更高的递归上限。
# 子图与外层图各自独立计数，外层 200 步足够；而子智能体是重量级 ReAct 循环
# （每轮 2 个超步，还要按纪律"小步写文件 + 逐一验证"），200 步只够约 100 次
# 工具调用，实测单 stage 写 21 个文件就会撞上限。保留禁用 checkpointer 继承。
_SUBAGENT_GRAPH_CONFIG = {
    **_NESTED_GRAPH_CONFIG,
    "recursion_limit": SUBAGENT_RECURSION_LIMIT,
}


def _emit_stream_event(event: Dict[str, Any]) -> None:
    """向上层的自定义事件流推送一条 harness 进度事件；无流式上下文时静默忽略。"""
    try:
        get_stream_writer()(event)
    except Exception:
        pass


# ── 后台运行注册表 ────────────────────────────────────────────────────
# Streamlit 每次交互都会重跑整个脚本，同步执行的长任务会被 rerun 直接中断
# （日志里只留下 start、没有 end/FAILED）。因此流水线放到后台线程里跑，
# 进度写进这张进程级注册表：模块级变量在 rerun 之间不会重建，界面只负责读取。
# 线程只写注册表与数据库，绝不调用 st.* —— Streamlit 的 API 只能在脚本线程里用。
_runs: Dict[str, Dict[str, Any]] = {}
_runs_lock = threading.RLock()


class RunCancelled(Exception):
    """后台运行被用户终止。

    由界面的 stop 按钮置位取消标记，执行线程在下个事件边界抛出本异常收尾。
    """


def create_run(workspace_id: str, harness_name: str, strategy: str, progress: dict,
               channel: str = "harness") -> dict:
    """登记一次后台执行，返回运行记录（含短 id）。

    channel 区分「主智能体」与「Harness 流水线」两种执行方式：
    它们共用同一个项目的工作区，但进度与聊天记录互不混排。
    """
    run = {
        "id": uuid.uuid4().hex[:8],
        "workspace_id": workspace_id,
        "channel": channel,
        "harness_name": harness_name,
        "strategy": strategy,
        "status": "running",
        "started_at": time.time(),
        "error": "",
        "cancel_requested": False,
        "progress": progress,
    }
    with _runs_lock:
        _runs[run["id"]] = run
    return run


def update_run(run_id: str, mutate: Callable[[dict], None]) -> None:
    """在锁内修改运行记录（进度对象是可变 dict，读方必须拿快照）。"""
    with _runs_lock:
        run = _runs.get(run_id)
        if run is not None:
            mutate(run)


def finish_run(run_id: str, status: str, error: str = "") -> None:
    """标记运行结束；记录保留，供界面在收尾这一次渲染里读到最终状态。"""
    with _runs_lock:
        run = _runs.get(run_id)
        if run is not None:
            run["status"] = status
            run["error"] = error
            run["finished_at"] = time.time()


def cancel_run(run_id: str) -> None:
    """请求终止一次运行：只置标记，由执行线程在下个事件边界自行收尾。"""
    with _runs_lock:
        run = _runs.get(run_id)
        if run is not None and run["status"] == "running":
            run["cancel_requested"] = True


def is_cancelled(run_id: Optional[str]) -> bool:
    """该运行是否已被请求终止。"""
    if not run_id:
        return False
    with _runs_lock:
        run = _runs.get(run_id)
        return bool(run and run.get("cancel_requested"))


_run_local = threading.local()


def bind_run(run_id: str) -> None:
    """把当前线程与一次运行绑定：工具内部（如 run_harness）据此检查取消标记。"""
    _run_local.run_id = run_id


def current_run_id() -> Optional[str]:
    return getattr(_run_local, "run_id", None)


def check_cancelled() -> None:
    """在事件边界检查一次取消标记，被终止时抛出 RunCancelled 交由执行线程收尾。"""
    if is_cancelled(current_run_id()):
        raise RunCancelled("用户终止了执行")


def snapshot_run(run_id: str) -> Optional[dict]:
    """取运行记录的深拷贝；进度在别的线程里被改写，直接读会读到半截状态。"""
    with _runs_lock:
        run = _runs.get(run_id)
        return copy.deepcopy(run) if run is not None else None


def active_run(workspace_id: Optional[str] = None,
               channel: Optional[str] = None) -> Optional[dict]:
    """取当前仍在执行的那一条运行（同工作区同时只允许一条）。"""
    with _runs_lock:
        for run in _runs.values():
            if run["status"] != "running":
                continue
            if workspace_id is not None and run["workspace_id"] != workspace_id:
                continue
            if channel is not None and run.get("channel") != channel:
                continue
            return copy.deepcopy(run)
    return None


def discard_run(run_id: str) -> None:
    with _runs_lock:
        _runs.pop(run_id, None)


def _initial_task(state: DeepAgentState) -> str:
    """取出本次 harness 运行的初始任务，即用户在 Harness Pipeline Workflow 模式里输入的需求。"""
    for msg in state.get("messages", []) or []:
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


def _stage_prompt(instructions: str, task: str) -> str:
    """把真实的用户任务拼进阶段指令，避免子智能体只能靠虚拟文件猜任务目标。"""
    if not task:
        return instructions
    return f"本次任务（来自用户）：\n{task}\n\n你的具体职责：\n{instructions}"


def _round_of(state: DeepAgentState, agent_name: str) -> int:
    """按 state 中已产出的该智能体消息数推导轮次，避免使用编译期闭包计数器。"""
    produced = sum(
        1 for msg in state.get("messages", []) or []
        if isinstance(msg, AIMessage) and str(msg.content).startswith(f"[{agent_name}]")
    )
    return produced + 1


# 严格评审通过判定：只认可“独立成行的 APPROVED 结论”。
# 宽松的子串匹配会把“我未输出 APPROVED”“结论：不通过”这类表述误判为已通过。
# 说明：结论前缀只允许"结论"类的固定词，不能放开通配，否则"我未输出 APPROVED"会被命中。
_APPROVED_LINE = re.compile(
    r"(?im)^[ \t>*#\-•]*(?:\**\s*)?"
    r"(?:最终结论|评审结论|审查结论|审核结论|结论|verdict|result)?"
    r"\s*[:：]?\s*\**\s*APPROVED\s*\**\s*[。.!]?\s*$"
)


def _is_approved(text: str) -> bool:
    """判断评审文本是否给出了通过结论。"""
    return bool(_APPROVED_LINE.search(_strip_stage_prefix(text)))


def _strip_stage_prefix(text: str) -> str:
    """去掉节点写回 state 时添加的 `[agent_name] ` 前缀。

    该前缀会让“结论行”不再位于文本行首，行首锚定的严格判定会因此失效。
    """
    return re.sub(r"^\s*\[[^\]]*\]\s*", "", str(text or ""))


def _clip(text: str, limit: int) -> str:
    """把多行文本压成单行并截断，避免事件流过长。"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _truncate(text: Any, limit: int) -> str:
    """按长度截断但保留原有换行，供 UI 以代码块形式原样展示。"""
    raw = str(text)
    return raw if len(raw) <= limit else raw[:limit] + "\n…（已截断）"


def _brief_args(args: Any) -> str:
    """把工具参数压成一行摘要。"""
    if not isinstance(args, dict):
        return _clip(args, 80)
    return ", ".join(f"{k}={_clip(v, 60)}" for k, v in list(args.items())[:3])


def _emit_subagent_message(msg: Any, stage: str, round_no: int, sink: Callable[[dict], None]) -> None:
    """把子智能体的一条消息拆成工具调用 / 工具结果 / 文本三类事件。

    工具调用与结果都带上 call_id，UI 侧据此把结果并回对应的调用条目，
    从而渲染成一个「工具名 + 参数 + 输出」的折叠块。
    """
    tool_calls = getattr(msg, "tool_calls", None) or []
    for call in tool_calls:
        sink({
            "type": "tool",
            "node": stage,
            "round": round_no,
            "call_id": call.get("id") or "",
            "tool": call.get("name") or "unknown",
            "detail": _brief_args(call.get("args", {})),
        })

    if isinstance(msg, ToolMessage):
        sink({
            "type": "result",
            "node": stage,
            "round": round_no,
            "call_id": getattr(msg, "tool_call_id", "") or "",
            "tool": msg.name or "tool",
            "detail": _truncate(msg.content, 2000),
        })
        return

    content = getattr(msg, "content", "")
    if content and not tool_calls:
        sink({
            "type": "message",
            "node": stage,
            "round": round_no,
            "detail": _truncate(content, 4000),
        })


# LangGraph ReAct 循环步数耗尽时，create_react_agent 会丢弃模型的真实响应，
# 用这句内置文案充当最终答复（chat_agent_executor._are_more_steps_needed）。
# 它不代表任务完成，只代表这一轮的步数预算用完了。
_STEPS_EXHAUSTED_TEXT = "Sorry, need more steps to process this request."

# 步数耗尽后允许续跑的次数：每次续跑都会重新获得一整轮步数预算
_STEPS_EXHAUSTED_RESUMES = 2


class HarnessStepExhausted(Exception):
    """子智能体在（含续跑的）步数预算内没有收敛。"""


def _steps_exhausted(final_state: dict) -> bool:
    """子智能体的最终答复是不是 LangGraph 的步数耗尽占位文案。"""
    msgs = (final_state or {}).get("messages") or []
    if not msgs or not isinstance(msgs[-1], AIMessage):
        return False
    return str(getattr(msgs[-1], "content", "") or "").strip() == _STEPS_EXHAUSTED_TEXT


def _stream_subagent_once(
        agent: Any,
        init_state: dict,
        stage: str,
        round_no: int,
        sink: Callable[[dict], None],
    ) -> dict:
    """流式执行子智能体一轮（一条完整的 recursion_limit 预算）。"""
    final_state: dict = {}
    for mode, payload in agent.stream(
        init_state,
        config=_SUBAGENT_GRAPH_CONFIG,
        stream_mode=["messages", "updates", "values"],
    ):
        # 每个 token 与每个 superstep 都是一次事件边界：用户点了停止就立即中断。
        # 只挂 messages 模式而不渲染它，是为了让长时间的模型生成也能随时停下——
        # 否则要等这一整段生成与工具调用跑完（实测可达一分钟以上）才轮得到检查。
        check_cancelled()
        if mode == "messages":
            continue
        if mode == "values":
            final_state = payload
            continue

        for node_payload in (payload or {}).values():
            produced = (node_payload or {}).get("messages") or []
            if not isinstance(produced, list):
                produced = [produced]
            for msg in produced:
                _emit_subagent_message(msg, stage, round_no, sink)

    return final_state


def stream_subagent(
        agent: Any,
        init_state: dict,
        stage: str,
        round_no: int,
        sink: Callable[[dict], None],
    ) -> dict:
    """流式执行子智能体，逐条回调其工具调用与消息，最后返回最终状态。

    直达模式（UI 直接调用子智能体）与 harness 内部节点共用这一条链路，
    因此两种模式下渲染出的执行过程格式完全一致。

    步数耗尽兜底：ReAct 循环用满 recursion_limit 不代表任务失败，只是这轮
    预算不够（长任务经多轮摘要压缩后尤其常见——压缩会丢掉"已写过哪些文件"
    的细节，模型容易重复劳动）。此时去掉占位文案，把已有状态传回去续跑，
    重新拿到一整轮预算；续跑仍耗尽则抛 HarnessStepExhausted 让流水线明确
    失败，而不是把占位文案当成阶段结论假装成功。
    """
    final_state = _stream_subagent_once(agent, init_state, stage, round_no, sink)
    for resume_no in range(1, _STEPS_EXHAUSTED_RESUMES + 1):
        if not _steps_exhausted(final_state):
            return final_state
        debug_log(
            f"[harness.subagent] {stage} 第 {round_no} 轮步数耗尽，"
            f"执行第 {resume_no}/{_STEPS_EXHAUSTED_RESUMES} 次续跑"
        )
        msgs = list(final_state.get("messages") or [])
        if _steps_exhausted({"messages": msgs}):
            # 占位文案不是真实结论，去掉后模型才能接着干而不是接着道歉
            msgs = msgs[:-1]
        final_state = _stream_subagent_once(
            agent, {**final_state, "messages": msgs}, stage, round_no, sink)
    if _steps_exhausted(final_state):
        raise HarnessStepExhausted(
            f"子智能体 {stage} 在 {_STEPS_EXHAUSTED_RESUMES + 1} 轮步数预算内仍未完成任务。"
            "任务可能过于庞大，请把目标拆小，或调大 SUBAGENT_RECURSION_LIMIT 后重试。")
    return final_state


async def _astream_graph(compiled_graph: Any, init_state: dict, sink: Callable[[dict], None]) -> dict:
    """以异步方式流式执行 harness 图（parallel 策略专用，保证同 superstep 内的节点并发）。"""
    final_state = init_state
    async for mode, payload in compiled_graph.astream(
        init_state,
        config=_NESTED_GRAPH_CONFIG,
        stream_mode=["custom", "values", "messages"],
    ):
        # messages 只是 token 级的取消检查点，事件仍只由 custom 驱动
        check_cancelled()
        if mode == "custom":
            sink(payload)
        elif mode == "values":
            final_state = payload
    return final_state


# ── parallel 策略：分发器 / 子任务沙盒 / 汇总器 ──────────────────────

class DispatchTaskDraft(BaseModel):
    """分发器输出的单个子任务。"""
    agent_name: str = Field(description="执行该子任务的智能体名，必须来自给定的可分发智能体清单")
    instructions: str = Field(
        description="只属于该智能体的自包含任务描述：完成该子任务所必需的上下文 + 清晰的任务目标 + 预期输出格式")


class DispatchPlan(BaseModel):
    """分发器的结构化输出：一份子任务清单。"""
    tasks: list[DispatchTaskDraft] = Field(description="子任务清单，每个智能体最多一条")


_DISPATCH_SCHEMA_DOC = json.dumps(DispatchPlan.model_json_schema(), ensure_ascii=False, indent=2)
_DISPATCH_PROMPT = """你是多智能体并行流水线的「任务分发器」。

根据用户任务，把工作拆解成若干可并行的子任务，分发给下列智能体。
{roster}

拆解规则（必须遵守）：
1. agent_name 只能从上面的清单中选择，每个智能体最多一条子任务；
2. instructions 必须自包含：子智能体看不到对话历史，也看不到其它子任务，
   因此要写清完成该子任务所必需的最小上下文、清晰的任务目标与预期输出格式；
3. 这些智能体会**并行**执行，严禁让它们修改同一个文件：
   必须在 instructions 中为每个智能体划清可修改的文件范围；
4. 某个智能体不需要参与时，不要为它生成任务。

用户任务：
{task}

输出格式（严格输出符合以下 JSON Schema 的单个 JSON 对象，不要输出任何其他内容）：
{schema}
"""

_harness_llm: Optional[ChatOpenAI] = None
_dispatch_chain: Any = None


def _llm_client() -> ChatOpenAI:
    """延迟创建 harness 内部使用的模型客户端。"""
    global _harness_llm
    if _harness_llm is None:
        _harness_llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", DEFAULT_MODEL),
            temperature=0.0,
            max_tokens=max(8192, DEFAULT_MAX_TOKENS),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("BASE_URL", ""),
        )
    return _harness_llm


def _safe_name(name: str) -> str:
    """把智能体名清洗成可用于目录名/文件名的片段。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name or "agent"))[:32] or "agent"


def _agent_role(agent_name: str) -> str:
    """取智能体的职能描述，用于让分发器理解每个智能体擅长什么。"""
    config = AgentStore.get(agent_name) or BUILTIN_TEMPLATES.get(agent_name) or {}
    return str(config.get("description") or "").strip()


def _parse_plan(raw: Any) -> list[dict]:
    """把模型输出解析成任务列表；json_mode 下模型可能直接给 dict 或 JSON 字符串。"""
    if isinstance(raw, DispatchPlan):
        return [task.model_dump() for task in raw.tasks]
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        tasks = raw.get("tasks") or []
        return [task for task in tasks if isinstance(task, dict)]
    return []


def _dispatch_tasks(stages: list[dict], task: str) -> list[dict]:
    """让分发器拆解任务；模型失败或漏发阶段时按阶段原始指令兜底。

    兜底的意义：harness 的每个阶段都是用户显式配置的，
    分发器漏掉某个智能体不应该导致该阶段被静默跳过。
    """
    global _dispatch_chain
    roster = "\n".join(
        f"- {stage['agent_name']}：{_agent_role(stage['agent_name']) or str(stage.get('instructions', ''))[:200]}"
        for stage in stages
    )
    prompt = _DISPATCH_PROMPT.format(
        roster=f"## 本次可用的子智能体\n{roster}",
        task=task or "（未提供任务描述）",
        schema=_DISPATCH_SCHEMA_DOC,
    )

    drafts: list[dict] = []
    try:
        if _dispatch_chain is None:
            _dispatch_chain = _llm_client().with_structured_output(DispatchPlan, method="json_mode")
        drafts = _parse_plan(_dispatch_chain.invoke(prompt))
    except Exception as e:
        debug_log(f"[harness.dispatcher] 拆解失败，回退到阶段原始指令：{type(e).__name__}: {e}")

    allowed = {stage["agent_name"] for stage in stages}
    tasks: list[dict] = []
    seen: set[str] = set()
    for draft in drafts:
        name = str(draft.get("agent_name") or "").strip()
        instructions = str(draft.get("instructions") or "").strip()
        if not name or not instructions or name not in allowed or name in seen:
            continue
        tasks.append({"agent_name": name, "instructions": instructions})
        seen.add(name)

    # 兜底：未被分配任务的阶段，按它在 harness 中的原始指令执行
    for stage in stages:
        name = stage["agent_name"]
        if name not in seen:
            tasks.append({
                "agent_name": name,
                "instructions": _stage_prompt(str(stage.get("instructions", "")), task),
            })
    return tasks


def _seed_sandbox(workspace_id: str, sandbox_dir: Path, files: dict) -> None:
    """把主工作区中该阶段可见的文件复制进子任务沙盒，实现写时隔离。"""
    if not files:
        return
    source_root = WorkspaceStore.workspace_dir(workspace_id)
    for rel in files:
        source = source_root / rel
        if not source.is_file():
            continue
        dest = sandbox_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, dest)
        except OSError:
            continue


async def _run_parallel_subagent(stage: dict, task: dict, state: DeepAgentState) -> dict:
    """在独立进程 + 独立沙盒目录中执行一个子智能体，返回它的文件提案。

    子进程内部读写自己那份文件树，主进程只在结束时接收「变更文件内容」，
    因此多个子智能体并行修改同一文件不会互相覆盖。
    """
    agent_name = stage["agent_name"]
    workspace_id = workspace_of(state)
    sandbox_dir = Path(tempfile.mkdtemp(prefix=f"harness_{_safe_name(agent_name)}_"))
    request_path = sandbox_dir.parent / f"{sandbox_dir.name}.json"

    passed_files = dict(state.get("files") or {}) if stage.get("pass_files", True) else {}
    _seed_sandbox(workspace_id, sandbox_dir, passed_files)
    request_path.write_text(
        json.dumps({
            "agent_name": agent_name,
            "instructions": task["instructions"],
            "workspace_id": workspace_id,
            "sandbox_dir": str(sandbox_dir),
            "files": passed_files,
            "recursion_limit": DEFAULT_RECURSION_LIMIT,
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    final_payload: Optional[dict] = None
    stderr_chunks: list[bytes] = []
    proc: Optional[asyncio.subprocess.Process] = None

    async def _drain_stderr() -> None:
        while True:
            chunk = await proc.stderr.read(8192)
            if not chunk:
                return
            stderr_chunks.append(chunk)

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "parallel_worker", str(request_path),
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=4 * 1024 * 1024,
        )
        drain = asyncio.create_task(_drain_stderr())

        async for raw in proc.stdout:
            # 每个子进程事件都是一次边界：用户终止时立刻停，由 finally 收掉子进程
            check_cancelled()
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # 忽略非协议输出
            if event.get("type") == "final":
                final_payload = event
            elif isinstance(event, dict):
                _emit_stream_event(event)

        returncode = await proc.wait()
        await drain
        stderr_text = b"".join(stderr_chunks).decode("utf-8", "replace").strip()

        if final_payload is None:
            return {
                "status": "error",
                "summary": (
                    f"⚠️ 子进程异常退出（exit={returncode}），没有产出任何文件。"
                    + (f"\n{stderr_text[-800:]}" if stderr_text else "")
                ),
                "files": {},
            }
        return final_payload
    finally:
        # 被终止时子进程还在跑，必须先杀掉再清理沙盒，否则进程会留在后台继续写临时目录
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        try:
            request_path.unlink(missing_ok=True)
        except OSError:
            pass


def _conflict_alias(path: str, agent_name: str) -> str:
    """为冲突文件生成带智能体后缀的别名路径，便于用户人工比对。"""
    target = Path(path)
    name = f"{target.stem}.conflict-{_safe_name(agent_name)}{target.suffix}"
    return str(target.with_name(name)).replace("\\", "/")


def _merge_conflict(path: str, versions: dict[str, str]) -> tuple[Optional[str], str]:
    """让模型合并多个智能体对同一文件的不同版本。

    返回 (合并后的内容, 无法合并的原因)；成功时原因为空串。
    """
    blocks = "\n\n".join(
        f"## 版本 {agent_name}\n```\n{content}\n```"
        for agent_name, content in versions.items()
    )
    prompt = f"""你是并行子任务的「冲突裁决器」。多个子智能体并行修改了同一个文件，需要合并成一个最终版本。

    文件：{path}

    {blocks}

    要求：
    - 保留各版本中互不冲突的改动；语义冲突处按「更完整、更正确」的原则裁决；
    - 只输出合并后的完整文件内容，不要解释、不要 Markdown 代码围栏；
    - 如果确实无法合并（两版实现互斥且无法取舍），只输出一行：UNMERGEABLE:<一句话原因>"""

    try:
        reply = _llm_client().invoke(prompt)
        text = str(reply.content).strip()
    except Exception as e:
        return None, f"合并模型调用失败：{type(e).__name__}: {e}"

    if not text:
        return None, "合并模型返回了空内容"
    stripped = text.strip()
    if stripped.startswith("```"):
        # 模型偶尔无视要求，仍用代码围栏包裹
        stripped = re.sub(r"^```[^\n]*\n", "", stripped)
        stripped = re.sub(r"\n```\s*$", "", stripped).strip()
    if stripped.upper().startswith("UNMERGEABLE"):
        reason = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
        return None, reason or "无法合并"
    return stripped, ""


def _final_summary(
            state: DeepAgentState,
            proposals: dict,
            final_files: dict[str, str],
            unresolved: dict[str, str],
    ) -> str:
    """根据各子任务结论与最终文件产出生成汇总回复（模型失败时用规则拼接兜底）。"""
    task = _initial_task(state)
    blocks = []
    for name, proposal in proposals.items():
        # 只把各分支的结论交给汇总模型：分发器下发的子任务指令不进入最终答复
        blocks.append(f"### {name}\n{_truncate(proposal.get('summary', ''), 1200)}")
    written = "\n".join(f"- {path}" for path in sorted(final_files)) or "（无）"
    conflict_note = ""
    if unresolved:
        conflict_note = "\n\n无法自动合并的文件：\n" + "\n".join(
            f"- {path}：{reason}（各版本已按 .conflict-<智能体> 后缀另存）"
            for path, reason in unresolved.items()
        )

    prompt = f"""你是多智能体并行流水线的「汇总器」。请根据下面的执行结果，给用户一份连贯的最终答复。

    ## 用户任务
    {task}

    ## 各子智能体的执行结果
    {chr(10).join(blocks) if blocks else '（没有子智能体产出）'}

    ## 最终写入工作区的文件
    {written}
    {conflict_note}

    要求：
    - 说明整体完成了什么、关键产出在哪些文件、分别由哪个智能体完成；
    - 如有冲突或失败，如实说明影响与遗留问题；
    - 不要复述工具调用过程，不要输出完整代码；
    - 使用简体中文，控制在 400 字以内。"""

    try:
        reply = _llm_client().invoke(prompt)
        text = str(reply.content).strip()
        if text:
            return text
    except Exception as e:
        debug_log(f"[harness.aggregator] 汇总回复失败，使用规则拼接：{type(e).__name__}: {e}")

    parts = ["并行流水线已完成。"]
    if final_files:
        parts.append("产出文件：" + "、".join(sorted(final_files)) + "。")
    for name, proposal in proposals.items():
        parts.append(f"- {name}：{_clip(proposal.get('summary', ''), 200)}")
    if unresolved:
        parts.append("以下文件存在无法自动合并的冲突，已另存为 .conflict-<智能体> 文件："
                     + "、".join(unresolved) + "。")
    return "\n".join(parts)


def _aggregator_node(state: DeepAgentState) -> dict:
    """汇总节点：冲突检测 → 合并 → 写回 files → 生成最终回复。"""
    _emit_stream_event({"type": "node", "node": "aggregator", "round": 1, "phase": "start"})
    proposals = state.get("proposals") or {}
    workspace_id = workspace_of(state)

    # path -> {agent_name: content}
    versions: dict[str, dict[str, str]] = {}
    for key, proposal in proposals.items():
        name = proposal.get("agent_name") or key
        for path, content in (proposal.get("files") or {}).items():
            versions.setdefault(path, {})[name] = content

    entries: dict[str, dict] = {}
    final_files: dict[str, str] = {}
    unresolved: dict[str, str] = {}

    for path, candidates in versions.items():
        if len(set(candidates.values())) == 1:
            # 只有一个版本，或多个智能体写出了完全相同的内容：直接采用
            final_files[path] = next(iter(candidates.values()))
            continue

        merged, reason = _merge_conflict(path, candidates)
        if merged is not None:
            final_files[path] = merged
            continue

        # 无法合并：各版本改名落盘供人工比对，原路径保持运行前的内容不被覆盖
        unresolved[path] = reason
        for agent_name, content in candidates.items():
            final_files[_conflict_alias(path, agent_name)] = content

    for path, content in final_files.items():
        try:
            # 索引键也归一化：子智能体可能用容器内绝对路径，避免侧边栏出现 /app/workspace 前缀
            entries[WorkspaceStore.normalize_path(path) or path] = WorkspaceStore.write(
                workspace_id, path, content
            )
        except (OSError, ValueError) as e:
            unresolved.setdefault(path, f"写入失败：{type(e).__name__}: {e}")

    summary = _final_summary(state, proposals, final_files, unresolved)
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "agent_name": "harness_engine",
        "action": "parallel_aggregate",
        "summary": f"汇总 {len(proposals)} 个子智能体的提案，写入 {len(entries)} 个文件"
                   + (f"，{len(unresolved)} 个文件存在冲突" if unresolved else "。"),
        "status": "error" if unresolved else "success",
    }
    _emit_stream_event({"type": "node", "node": "aggregator", "round": 1, "phase": "done"})
    return {
        "files": entries,
        "messages": [AIMessage(content=summary)],
        "execution_log": list(state.get("execution_log") or []) + [log_entry],
    }


# ── brainstorming 策略：中立主持人 / 参会角色 ────────────────────────

MODERATOR_NODE = "moderator"
_MODERATOR_TAG = f"[{MODERATOR_NODE}]"


class ModeratorDecisionDraft(BaseModel):
    """主持人每一轮的结构化决策。"""
    progress_summary: str = Field(
        description="【当前会议进展】正文：提炼上一位发言者的要点并与此前进展合并，2-4 句话")
    core_disagreement: str = Field(default="", description="当前尚未解决的核心分歧；没有则留空")
    next_agent: str = Field(
        default="", description="下一位发言的智能体名，必须来自参会角色清单；终止会议时留空")
    instruction: str = Field(default="", description="给下一位发言者的客观中立指示，不预设结论")
    should_end: bool = Field(
        default=False, description="讨论是否已收敛：各方核心分歧得到回应且形成了可执行方案")
    consensus: str = Field(default="", description="终止时填写各方已认可的方案；未达成共识写“未达成共识”")
    open_issues: str = Field(default="", description="终止时填写仍未解决的分歧；没有则留空")


_MODERATOR_SCHEMA_DOC = json.dumps(
    ModeratorDecisionDraft.model_json_schema(), ensure_ascii=False, indent=2)

_MODERATOR_PROMPT = """你是多智能体头脑风暴会议的「中立主持人」。

你的职责不是提出方案，也不是评判谁对谁错，而是推进一场高质量的多角色辩论：
1. 把上一位发言者的输出提炼成简洁的会议进展，并与此前的进展合并成【当前会议进展】；
2. 决定下一位发言人，并给出客观中立的发言指示（例如“请站在你的专业角度，对此发表看法或提出你的对策”）；
3. 一轮讨论后总结当前的核心分歧；
4. 当各方的核心分歧都得到回应、并收敛出一个可执行方案时，终止会议。

规则（必须遵守）：
- 会议刚开始（尚无发言记录）时，next_agent 必须指向一位参会者，instruction 要求其提出初始方案，以启动头脑风暴；
- 之后优先让尚未充分表达、且能回应现有分歧的角色发言；不要连续让同一角色发言；
- 指示必须中立客观，不得替任何一方站台，不得预设结论或暗示倾向；
- 至少每个参会角色都发过一次言、且核心分歧已被明确回应后，才允许 should_end=true；
- 达到最大发言次数 {max_speeches} 时必须终止，并如实说明未解决的分歧；
- progress_summary 用简体中文，只陈述事实与观点分歧，不做价值判断；
- 终止时 consensus 写明各方已认可的方案（无共识则写“未达成共识”），open_issues 列出遗留分歧。

参会角色（立场互相冲突）：
{roster}

讨论记录（按时间顺序）：
{transcript}

已发言次数：{speeches} / {max_speeches}

输出格式（严格输出符合以下 JSON Schema 的单个 JSON 对象，不要输出任何其他内容）：
{schema}
"""

_moderator_chain: Any = None


def _tagged_body(content: Any) -> tuple[str, str]:
    """把 `[name] 正文` 拆成 (name, 正文)；没有标签时 name 为空串。"""
    text = str(content or "")
    match = re.match(r"^\s*\[([^\]]+)\]\s*", text)
    if not match:
        return "", text
    return match.group(1), text[match.end():]


def _speech_counts(state: DeepAgentState, participants: list[str]) -> dict[str, int]:
    """统计每位参会角色已经发言的次数。"""
    counts = {name: 0 for name in participants}
    for msg in state.get("messages") or []:
        if not isinstance(msg, AIMessage):
            continue
        name, _ = _tagged_body(msg.content)
        if name in counts:
            counts[name] += 1
    return counts


def _last_speaker(state: DeepAgentState, participants: list[str]) -> str:
    """取最近一位发过言的参会角色。"""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage):
            name, _ = _tagged_body(msg.content)
            if name in participants:
                return name
    return ""


def _latest_moderator_note(state: DeepAgentState) -> str:
    """取主持人最近一次发言（进展总结 + 指示），作为本轮发言者的输入。"""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage):
            name, body = _tagged_body(msg.content)
            if name == MODERATOR_NODE:
                return body
    return ""


def _discussion_transcript(
            state: DeepAgentState, participants: list[str],
            limit: int = 12, per_message: int = 2000,
    ) -> str:
    """把讨论记录压成主持人可读的摘要文本。"""
    blocks = []
    for msg in state.get("messages") or []:
        if not isinstance(msg, AIMessage):
            continue
        name, body = _tagged_body(msg.content)
        if name in participants:
            blocks.append(f"### {name} 的发言\n{_truncate(body, per_message)}")
        elif name == MODERATOR_NODE:
            blocks.append(f"### 主持人的进展总结\n{_truncate(body, 600)}")
    return "\n\n".join(blocks[-limit:]) or "（讨论尚未开始）"


def _pick_next_speaker(
            participants: list[str], counts: dict[str, int],
            last_name: str, proposed: str,
    ) -> str:
    """确定下一位发言人：尊重模型选择，但保证轮换、避免同一角色连轴发言。"""
    proposed = str(proposed or "").strip()
    everyone_spoke = all(count >= 1 for count in counts.values())
    if proposed in counts and (proposed != last_name or everyone_spoke):
        return proposed
    candidates = [name for name in participants if name != last_name] or list(participants)
    return min(candidates, key=lambda name: (counts.get(name, 0), participants.index(name)))


def _fallback_decision(
            state: DeepAgentState, participants: list[str], counts: dict[str, int],
            max_speeches: int, reason: str,
    ) -> ModeratorDecisionDraft:
    """模型不可用时的规则兜底：轮询选人，保证会议能继续或安全终止。"""
    speeches = sum(counts.values())
    last_name, last_body = "", ""
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, AIMessage):
            name, body = _tagged_body(msg.content)
            if name in participants:
                last_name, last_body = name, body
                break

    if last_name:
        summary = f"{last_name} 已完成第 {speeches} 次发言，要点：{_clip(last_body, 180)}"
    else:
        summary = "会议刚开始，尚未有角色发言。"

    if speeches >= max_speeches:
        return ModeratorDecisionDraft(
            progress_summary=summary,
            should_end=True,
            consensus="未达成共识（主持人模型不可用，未能提炼最终方案）",
            open_issues=reason,
        )

    return ModeratorDecisionDraft(
        progress_summary=summary,
        next_agent=_pick_next_speaker(participants, counts, last_name, ""),
        instruction=(
            "请先提出你的初始方案，说明核心思路与关键取舍，以启动本次讨论。"
            if speeches == 0
            else "请站在你的专业角度，对此发表看法或提出你的对策，并直接回应此前的争议点。"
        ),
    )


def _moderate(
            stages: list[dict], state: DeepAgentState, participants: list[str],
            counts: dict[str, int], max_speeches: int,
    ) -> ModeratorDecisionDraft:
    """调用中立主持人模型，产出本轮的会议进展、下一位发言人与指示。"""
    global _moderator_chain
    speeches = sum(counts.values())
    roster = "\n".join(
        f"- {stage['agent_name']}：{_truncate(str(stage.get('instructions') or _agent_role(stage['agent_name'])), 300)}"
        for stage in stages
    )
    prompt = _MODERATOR_PROMPT.format(
        max_speeches=max_speeches,
        roster=roster,
        transcript=_discussion_transcript(state, participants),
        speeches=speeches,
        schema=_MODERATOR_SCHEMA_DOC,
    )
    if speeches >= max_speeches:
        prompt += "\n\n注意：已达到最大发言次数，本轮必须终止会议（should_end=true，next_agent 留空）。"

    try:
        if _moderator_chain is None:
            _moderator_chain = _llm_client().with_structured_output(
                ModeratorDecisionDraft, method="json_mode")
        raw = _moderator_chain.invoke(prompt)
        if isinstance(raw, ModeratorDecisionDraft):
            return raw
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict):
            return ModeratorDecisionDraft(**raw)
    except Exception as e:
        reason = f"主持人模型调用失败：{type(e).__name__}: {e}"
        debug_log(f"[harness.moderator] {reason}")
        return _fallback_decision(state, participants, counts, max_speeches, reason)

    return _fallback_decision(
        state, participants, counts, max_speeches, "主持人模型返回了无法解析的结果")


def _format_progress(decision: ModeratorDecisionDraft, ending: bool, next_agent: str) -> str:
    """把主持人的决策渲染成会议进程消息。"""
    parts = ["【当前会议进展】", decision.progress_summary.strip() or "（暂无新的进展）"]
    if decision.core_disagreement.strip():
        parts += ["", f"当前核心分歧：{decision.core_disagreement.strip()}"]

    if ending:
        parts += [
            "",
            "【会议结束】",
            f"最终方案：{decision.consensus.strip() or '未达成共识'}",
        ]
        if decision.open_issues.strip():
            parts.append(f"遗留分歧：{decision.open_issues.strip()}")
    else:
        instruction = decision.instruction.strip() or "请站在你的专业角度，对此发表看法或提出你的对策。"
        parts += ["", f"对下一位发言者（{next_agent}）的指示：{instruction}"]
    return "\n".join(parts)


def _speaker_prompt(
            stage: dict, state: DeepAgentState, participants: list[str], round_no: int,
    ) -> str:
    """拼装参会角色这一轮的发言提示词：任务 + 角色立场 + 主持人指示 + 讨论记录。"""
    task = _initial_task(state)
    note = _latest_moderator_note(state)
    transcript = _discussion_transcript(state, participants)
    return f"""你是多智能体头脑风暴会议的参会角色「{stage['agent_name']}」，这是你的第 {round_no} 次发言。

本次要讨论的问题：
{task or '（未提供任务描述）'}

你的角色立场（必须坚持，不得与其他角色趋同）：
{stage.get('instructions') or _agent_role(stage['agent_name'])}

主持人本轮的进展总结与指示：
{note or '（主持人尚未给出指示）'}

此前的讨论记录：
{transcript}

发言要求：
- 站在你的专业角度独立判断，直接回应主持人指示与讨论中的争议点；
- 若你认为现有方案有问题，必须指出具体缺陷，并给出可执行的替代方案或改进措施；
- 禁止无原则附和、禁止只做礼貌性肯定；你的职责本就与其他参会者冲突，请为你的立场辩护，\
用事实与权衡（成本、质量、风险、工期）说服对方；
- 需要产出代码或文档时，直接写入虚拟文件系统；
- 结尾用 2-3 句话给出你本轮的核心主张，便于主持人提炼。"""


def _completion_reason(config: dict, final_state: dict) -> str:
    """推断 harness 的结束原因，供 UI 直接展示。"""
    if config.get("strategy") == "brainstorming":
        last = ""
        for msg in reversed((final_state or {}).get("messages", [])):
            if isinstance(msg, AIMessage):
                last = str(msg.content)
                break
        return "consensus" if "【会议结束】" in last else "completed"
    if config.get("strategy") != "review_loop":
        return "completed"

    stages = config.get("stages", [])
    reviewer_name = stages[-1]["agent_name"] if stages else ""
    reviews = [
        str(msg.content)
        for msg in (final_state or {}).get("messages", [])
        if isinstance(msg, AIMessage) and str(msg.content).startswith(f"[{reviewer_name}]")
    ]
    if reviews and _is_approved(reviews[-1]):
        return "approved"
    if len(reviews) >= config.get("max_review_rounds", 3):
        return "max_rounds"
    return "completed"


# ── 内置 Harness 模板 ────────────────────────────────────────────────

BUILTIN_HARNESSES = [
    {
        "name": "Code & Test",
        "description": "顺序循环：由 Coder 编写代码，Tester 编写并运行测试，按需循环修改。",
        "purpose": (
            "适合「写完代码还要跑通测试」的质量闭环任务：新功能开发、修 bug 后回归、"
            "给已有模块补齐测试。不适合纯方案设计或没有可执行代码的文档任务。"
        ),
        "strategy": "review_loop",
        "stages": [
            {"agent_name": "coder", "instructions": "针对用户的需求编写一份完整的 Python 实现。", "pass_files": True,
             "description": "按需求写出完整实现；评审未通过时接收评审意见并迭代修复。"},
            {"agent_name": "tester", "instructions": (
                "为虚拟文件系统中的实现编写并运行完整测试：覆盖正常路径、边界条件与错误处理。"
                "发现失败时先修好测试本身（只有当实现确实有缺陷时才算实现的问题）。"
                "输出要求：先列出测试结果与失败原因（每条标明文件、位置与建议），"
                "然后**单独一行**给出结论——只要还存在失败的用例或缺失的关键覆盖，就输出 "
                "'## 结论：NEEDS_FIX'；只有当全部用例通过且关键路径都已覆盖时才输出 "
                "'## 结论：APPROVED'。"
            ), "pass_files": True,
             "description": "编写并运行测试，输出失败清单与 APPROVED / NEEDS_FIX 结论，决定是否回到编码阶段。"}
        ],
        "max_review_rounds": 3,
        "icon": "🔄",
        "color": "#ef4444"
    },
    {
        "name": "Design & Build",
        "description": "由 Architect 设计整体架构与接口规范，再由 Coder 实现。",
        "purpose": (
            "适合从零搭建或整体改造的场景：需要先定模块划分、依赖方向与核心接口，"
            "再按设计落代码，避免边写边改结构。"
        ),
        "strategy": "sequential",
        "stages": [
            {"agent_name": "architect", "instructions": "设计系统的架构与核心接口，并把设计写入 design.py", "pass_files": True,
             "description": "产出架构设计、模块划分与核心接口（design.py），为后续实现定边界。"},
            {"agent_name": "coder", "instructions": "阅读 design.py，并按照设计实现完整方案。", "pass_files": True,
             "description": "依据 design.py 实现完整方案，把架构设计落成可运行代码。"}
        ],
        "max_review_rounds": 1,
        "icon": "🏗️",
        "color": "#8b5cf6"
    },
    {
        "name": "Research & Write",
        "description": "由 Researcher 收集技术文档资料，Coder 依据研究结果实现代码。",
        "purpose": (
            "适合需要先查资料再动手的场景：技术选型调研、接入陌生的第三方库或 API、"
            "需要带出处依据的实现与文档撰写。"
        ),
        "strategy": "sequential",
        "stages": [
            {"agent_name": "researcher", "instructions": "搜索网络，为该任务收集技术参考资料，并写入 research_notes.md。", "pass_files": True,
             "description": "检索并筛选资料，把关键结论与出处整理进 research_notes.md。"},
            {"agent_name": "coder", "instructions": "阅读 research_notes.md，并实现所要求的应用代码。", "pass_files": True,
             "description": "依据研究笔记实现代码，把资料结论落到具体实现。"}
        ],
        "max_review_rounds": 1,
        "icon": "📚",
        "color": "#14b8a6"
    },
    {
        "name": "Spec & Build",
        "description": "由 Product Manager 拆解需求与验收标准，Architect 定架构，Coder 实现。",
        "purpose": (
            "适合需求含糊、需要先澄清再交付的场景：把一句话需求变成可验收的 PRD，"
            "再定架构并实现，适合完整功能从需求到代码的一次性交付。"
        ),
        "strategy": "sequential",
        "stages": [
            {"agent_name": "product-manager", "instructions": "把用户需求拆成 PRD 与验收标准，写入 requirements.md。", "pass_files": True,
             "description": "把需求拆成 PRD、用户故事与可验收标准（requirements.md）。"},
            {"agent_name": "architect", "instructions": "阅读 requirements.md，给出模块划分与核心接口，写入 design.py。", "pass_files": True,
             "description": "依据 PRD 给出模块划分、依赖方向与核心接口（design.py）。"},
            {"agent_name": "coder", "instructions": "按 requirements.md 与 design.py 实现完整方案。", "pass_files": True,
             "description": "按 PRD 与设计实现完整方案，逐条对齐验收标准。"}
        ],
        "max_review_rounds": 1,
        "icon": "📋",
        "color": "#d97706"
    }
]


def sync_builtin_harnesses() -> None:
    """把内置 harness 模板同步进数据库，使模板升级能生效到已有记录。

    仅按名称覆盖同名记录；UI 目前不提供 harness 编辑入口，因此不存在覆盖用户自定义的风险。
    """
    for template in BUILTIN_HARNESSES:
        if HarnessStore.get(template["name"]) != template:
            HarnessStore.save(template)


class HarnessEngine:
    """负责动态编译并运行多智能体 harness 的编排器。"""

    @staticmethod
    def compile_graph(config: dict) -> Any:
        """根据 harness 配置动态编译出一个 LangGraph StateGraph。"""
        builder = StateGraph(DeepAgentState)
        strategy = config.get("strategy", "sequential")
        stages = config.get("stages", [])

        if not stages:
            raise ValueError("Harness 至少需要配置一个阶段。")

        if strategy == "sequential":
            prev_node = START
            for idx, stage in enumerate(stages):
                node_name = f"stage_{idx}_{stage['agent_name']}"

                def make_node(stg, name):
                    def node_fn(state: DeepAgentState) -> dict:
                        _emit_stream_event({
                            "type": "node", "node": stg["agent_name"], "round": 1, "phase": "start"})
                        agent = AgentFactory.get_agent(stg["agent_name"])
                        prompt = _stage_prompt(stg["instructions"], _initial_task(state))

                        # 判断是否向该阶段传递状态中的文件
                        passed_files = state.get("files", {}) if stg.get("pass_files", True) else {}

                        # 调用子智能体（流式执行，执行过程实时透出给 UI）
                        result = stream_subagent(
                            agent,
                            {
                                "messages": [HumanMessage(content=prompt)],
                                "files": passed_files,
                                "workspace_id": workspace_of(state),
                                "todos": state.get("todos", []),
                                "execution_log": state.get("execution_log", []),
                            },
                            stg["agent_name"],
                            1,
                            _emit_stream_event,
                        )

                        # 记录执行日志：只留阶段与状态，不把阶段指令写进日志，
                        # 否则执行时间线会把分发器给各智能体的任务指令暴露到界面上
                        log_entry = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "agent_name": stg["agent_name"],
                            "action": f"harness_{name}",
                            "summary": f"阶段 {stg['agent_name']} 执行完成",
                            "status": "success"
                        }
                        logs = list(state.get("execution_log", []))
                        logs.append(log_entry)
                        _emit_stream_event({
                            "type": "node", "node": stg["agent_name"], "round": 1, "phase": "done"})

                        # 输出状态更新
                        return {
                            "files": result.get("files", state.get("files", {})),
                            "todos": result.get("todos", state.get("todos", [])),
                            "execution_log": logs,
                            "messages": [AIMessage(content=f"[{stg['agent_name']}] {result['messages'][-1].content}")]
                        }
                    return node_fn

                builder.add_node(node_name, make_node(stage, node_name))
                builder.add_edge(prev_node, node_name)
                prev_node = node_name

            builder.add_edge(prev_node, END)

        elif strategy == "parallel":
            # 图结构：dispatcher 先拆解任务 → 各子智能体在独立进程/独立沙盒中并行执行
            # → aggregator 做冲突检测、合并、写回 files 并生成最终回复。
            def dispatcher_fn(state: DeepAgentState) -> dict:
                _emit_stream_event({
                    "type": "node", "node": "dispatcher", "round": 1, "phase": "start"})
                task = _initial_task(state)
                tasks = _dispatch_tasks(stages, task)
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "agent_name": "harness_engine",
                    "action": "parallel_dispatch",
                    # 只记录拆解结果数量，不把子任务指令写进日志
                    "summary": f"已拆解出 {len(tasks)} 个并行子任务",
                    "status": "success",
                }
                _emit_stream_event({
                    "type": "node", "node": "dispatcher", "round": 1, "phase": "done"})
                return {
                    "tasks_to_dispatch": tasks,
                    "execution_log": list(state.get("execution_log") or []) + [log_entry],
                }

            builder.add_node("dispatcher", dispatcher_fn)
            builder.add_node("aggregator", _aggregator_node)
            builder.add_edge(START, "dispatcher")

            for idx, stage in enumerate(stages):
                node_name = f"stage_{idx}_{stage['agent_name']}"

                def make_parallel_node(stg):
                    async def node_fn(state: DeepAgentState) -> dict:
                        _emit_stream_event({
                            "type": "node", "node": stg["agent_name"], "round": 1, "phase": "start"})
                        # 只执行属于自己这个智能体的那条指令
                        task = next(
                            (t for t in (state.get("tasks_to_dispatch") or [])
                             if t.get("agent_name") == stg["agent_name"]),
                            None,
                        )
                        logs = list(state.get("execution_log") or [])
                        if task is None:
                            logs.append({
                                "timestamp": datetime.datetime.now().isoformat(),
                                "agent_name": stg["agent_name"],
                                "action": "parallel_skip",
                                "summary": "分发器没有为该智能体分配子任务，本阶段跳过。",
                                "status": "success",
                            })
                            _emit_stream_event({
                                "type": "node", "node": stg["agent_name"], "round": 1, "phase": "done"})
                            return {"execution_log": logs}

                        try:
                            result = await _run_parallel_subagent(stg, task, state)
                        finally:
                            # 每个分支真正结束时才发 done：并行节点同时起步，
                            # 完成时刻各不相同，前端据此单独点亮各自的 ✓
                            _emit_stream_event({
                                "type": "node", "node": stg["agent_name"],
                                "round": 1, "phase": "done"})
                        summary = str(result.get("summary") or "")
                        status = "success" if result.get("status") == "success" else "error"
                        logs.append({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "agent_name": stg["agent_name"],
                            "action": "parallel_subagent",
                            # 只记状态，不把分发器下发的子任务指令写进日志
                            "summary": "并行子任务执行成功" if status == "success" else "并行子任务执行失败",
                            "status": status,
                        })
                        return {
                            "proposals": {stg["agent_name"]: {
                                "agent_name": stg["agent_name"],
                                "instructions": task["instructions"],
                                "summary": summary,
                                "files": result.get("files") or {},
                            }},
                            "messages": [AIMessage(content=f"[{stg['agent_name']}] {summary}")],
                            "execution_log": logs,
                        }
                    return node_fn

                builder.add_node(node_name, make_parallel_node(stage))
                builder.add_edge("dispatcher", node_name)
                builder.add_edge(node_name, "aggregator")

            builder.add_edge("aggregator", END)

        elif strategy == "review_loop":
            if len(stages) < 2:
                raise ValueError("评审循环策略至少需要 2 个阶段：创建阶段与评审阶段（最后一个阶段）。")

            creators = stages[:-1]   # 创建链：前面的所有阶段按顺序执行
            reviewer = stages[-1]    # 最后一个阶段作为评审者
            max_rounds = config.get("max_review_rounds", 3)

            def make_creator_node(stg):
                def creator_fn(state: DeepAgentState) -> dict:
                    round_no = _round_of(state, stg["agent_name"])
                    _emit_stream_event({
                        "type": "node", "node": stg["agent_name"], "round": round_no,
                        "phase": "start"})
                    feedback = ""
                    # 在历史消息中查找最近一条评审者消息
                    for msg in reversed(state.get("messages", [])):
                        if isinstance(msg, AIMessage) and msg.content.startswith(f"[{reviewer['agent_name']}]"):
                            feedback = msg.content
                            break

                    prompt = _stage_prompt(stg["instructions"], _initial_task(state))
                    if feedback:
                        prompt += f"\n\n上一次评审者的反馈：\n{feedback}\n请修复上述问题。"

                    agent = AgentFactory.get_agent(stg["agent_name"])
                    result = stream_subagent(
                        agent,
                        {
                            "messages": [HumanMessage(content=prompt)],
                            "files": state.get("files", {}),
                            "workspace_id": workspace_of(state),
                            "todos": state.get("todos", []),
                            "execution_log": state.get("execution_log", []),
                        },
                        stg["agent_name"],
                        round_no,
                        _emit_stream_event,
                    )

                    _emit_stream_event({
                        "type": "node", "node": stg["agent_name"], "round": round_no,
                        "phase": "done"})
                    return {
                        "files": result.get("files", state.get("files", {})),
                        "todos": result.get("todos", state.get("todos", [])),
                        "messages": [AIMessage(content=f"[{stg['agent_name']}] {result['messages'][-1].content}")]
                    }
                return creator_fn

            # 创建链节点：START → creator_0 → ... → creator_n → reviewer
            creator_node_names = []
            for idx, stg in enumerate(creators):
                node_name = f"creator_{idx}_{stg['agent_name']}"
                builder.add_node(node_name, make_creator_node(stg))
                creator_node_names.append(node_name)

            # 评审者（Reviewer）节点
            def reviewer_fn(state: DeepAgentState) -> dict:
                round_no = _round_of(state, reviewer["agent_name"])
                _emit_stream_event({
                    "type": "node", "node": reviewer["agent_name"], "round": round_no,
                    "phase": "start"})
                agent = AgentFactory.get_agent(reviewer["agent_name"])
                result = stream_subagent(
                    agent,
                    {
                        "messages": [HumanMessage(content=_stage_prompt(reviewer["instructions"], _initial_task(state)))],
                        "files": state.get("files", {}),
                        "workspace_id": workspace_of(state),
                        "todos": state.get("todos", []),
                        "execution_log": state.get("execution_log", []),
                    },
                    reviewer["agent_name"],
                    round_no,
                    _emit_stream_event,
                )
                _emit_stream_event({
                    "type": "node", "node": reviewer["agent_name"], "round": round_no,
                    "phase": "done"})
                return {
                    "messages": [AIMessage(content=f"[{reviewer['agent_name']}] {result['messages'][-1].content}")]
                }

            builder.add_node("reviewer", reviewer_fn)

            builder.add_edge(START, creator_node_names[0])
            for prev_node, next_node in zip(creator_node_names, creator_node_names[1:]):
                builder.add_edge(prev_node, next_node)
            builder.add_edge(creator_node_names[-1], "reviewer")

            # 路由决策
            def check_loop(state: DeepAgentState) -> Literal["creator", END]:
                msgs = state.get("messages", [])
                review_count = sum(1 for m in msgs if isinstance(m, AIMessage) and str(m.content).startswith(f"[{reviewer['agent_name']}]"))

                if review_count >= max_rounds:
                    return END

                # 严格判定：只认可评审者“独立成行”的 APPROVED 结论。
                # 子串匹配会把“我未输出 APPROVED”“结论：不通过”等表述误判为已通过。
                if _is_approved(str(msgs[-1].content) if msgs else ""):
                    return END

                return "creator"

            # 循环回到创建链的起点
            builder.add_conditional_edges("reviewer", check_loop, {"creator": creator_node_names[0], END: END})

        elif strategy == "brainstorming":
            # 图结构：moderator 每轮总结进展并指定下一位发言人 → 该角色发言 → 回到 moderator，
            # 直到主持人判定收敛（或达到最大发言次数）后终止。
            if len(stages) < 2:
                raise ValueError("头脑风暴策略至少需要 2 个参会角色（立场互相冲突的智能体）。")

            participants = list(dict.fromkeys(stage["agent_name"] for stage in stages))
            stage_of: dict[str, dict] = {}
            for stage in stages:
                stage_of.setdefault(stage["agent_name"], stage)
            node_of = {name: f"speaker_{idx}_{name}" for idx, name in enumerate(participants)}
            max_speeches = int(config.get("max_review_rounds", 6) or 6)

            def moderator_fn(state: DeepAgentState) -> dict:
                counts = _speech_counts(state, participants)
                speeches = sum(counts.values())
                round_no = min(max(1, speeches + 1), max_speeches)
                _emit_stream_event({
                    "type": "node", "node": MODERATOR_NODE, "round": round_no, "phase": "start"})

                decision = _moderate(stages, state, participants, counts, max_speeches)
                last_name = _last_speaker(state, participants)

                # 终止条件：达到发言上限；或主持人判定收敛、且每个角色都已发过言
                min_speeches = min(max_speeches, len(participants))
                ending = speeches >= max_speeches or (
                    decision.should_end and speeches >= min_speeches)

                if ending:
                    next_agent = ""
                    instruction = ""
                    if not decision.consensus.strip():
                        decision.consensus = "未达成共识"
                else:
                    next_agent = _pick_next_speaker(
                        participants, counts, last_name, decision.next_agent)
                    instruction = decision.instruction.strip() or (
                        "请站在你的专业角度，对此发表看法或提出你的对策。")
                    if speeches == 0:
                        instruction = (
                            "请先提出你的初始方案，说明核心思路与关键取舍，以启动本次讨论。"
                            + instruction)

                body = _format_progress(decision, ending, next_agent)
                _emit_stream_event({
                    "type": "message", "node": MODERATOR_NODE,
                    "round": round_no, "detail": _truncate(body, 4000),
                })
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "agent_name": MODERATOR_NODE,
                    "action": "brainstorm_marshal",
                    # 只记录进程状态；发言正文与收敛结论不写进审计日志
                    "summary": "会议已结束" if ending else f"指定下一位发言者：{next_agent}",
                    "status": "success",
                }
                _emit_stream_event({
                    "type": "node", "node": MODERATOR_NODE, "round": round_no, "phase": "done"})
                return {
                    "messages": [AIMessage(content=f"{_MODERATOR_TAG} {body}")],
                    "moderator_decision": {
                        "progress_summary": decision.progress_summary,
                        "core_disagreement": decision.core_disagreement,
                        "next_agent": next_agent,
                        "instruction": instruction,
                        "should_end": ending,
                        "consensus": decision.consensus,
                        "open_issues": decision.open_issues,
                    },
                    "execution_log": list(state.get("execution_log") or []) + [log_entry],
                }

            def make_speaker_node(stg):
                def speaker_fn(state: DeepAgentState) -> dict:
                    counts = _speech_counts(state, participants)
                    round_no = sum(counts.values()) + 1
                    _emit_stream_event({
                        "type": "node", "node": stg["agent_name"], "round": round_no,
                        "phase": "start"})

                    result = stream_subagent(
                        AgentFactory.get_agent(stg["agent_name"]),
                        {
                            "messages": [HumanMessage(
                                content=_speaker_prompt(stg, state, participants, round_no))],
                            "files": state.get("files", {}),
                            "workspace_id": workspace_of(state),
                            "todos": state.get("todos", []),
                            "execution_log": state.get("execution_log", []),
                        },
                        stg["agent_name"],
                        round_no,
                        _emit_stream_event,
                    )
                    reply = (
                        str(result["messages"][-1].content)
                        if result.get("messages") else "（本轮没有产出）"
                    )
                    log_entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "agent_name": stg["agent_name"],
                        "action": "brainstorm_speak",
                        "summary": f"第 {round_no} 次发言完成",
                        "status": "success",
                    }
                    _emit_stream_event({
                        "type": "node", "node": stg["agent_name"], "round": round_no,
                        "phase": "done"})
                    return {
                        # 发言是串行的，直接写共享工作区：后一位发言者能立即看到前一位的修改
                        "files": result.get("files", state.get("files", {})),
                        "todos": result.get("todos", state.get("todos", [])),
                        "messages": [AIMessage(content=f"[{stg['agent_name']}] {reply}")],
                        "execution_log": list(state.get("execution_log") or []) + [log_entry],
                    }
                return speaker_fn

            builder.add_node(MODERATOR_NODE, moderator_fn)
            for name in participants:
                builder.add_node(node_of[name], make_speaker_node(stage_of[name]))
                builder.add_edge(node_of[name], MODERATOR_NODE)

            builder.add_edge(START, MODERATOR_NODE)

            def route_after_moderator(state: DeepAgentState) -> str:
                decision = state.get("moderator_decision") or {}
                if decision.get("should_end"):
                    return END
                return str(decision.get("next_agent") or "")

            builder.add_conditional_edges(
                MODERATOR_NODE,
                route_after_moderator,
                {**node_of, END: END},
            )

        return builder.compile()

    @staticmethod
    def describe(config: dict) -> Dict[str, Any]:
        """返回用于 UI 展示的流水线节点清单与循环轮次。"""
        strategy = config.get("strategy", "sequential")
        stages = config.get("stages", [])

        if strategy == "review_loop" and len(stages) >= 2:
            # 前 N-1 个阶段均为创建者（按顺序组成创建链），最后一个阶段是评审者，
            # 因此 UI 流程条需要展示全部阶段。
            return {
                "strategy": strategy,
                "nodes": [stage["agent_name"] for stage in stages],
                "total_rounds": config.get("max_review_rounds", 3),
            }

        if strategy == "brainstorming":
            # 头脑风暴：主持人与参会角色交替发言，UI 流程条展示主持人和全部角色
            participants = list(dict.fromkeys(stage["agent_name"] for stage in stages))
            return {
                "strategy": strategy,
                "nodes": [MODERATOR_NODE, *participants],
                "total_rounds": config.get("max_review_rounds", 6),
            }

        nodes = [stage["agent_name"] for stage in stages]
        if strategy == "parallel":
            # 并行流水线：先分发，再并行执行各子智能体，最后汇总
            nodes = ["dispatcher", *nodes, "aggregator"]
        return {"strategy": strategy, "nodes": nodes, "total_rounds": 1}

    @staticmethod
    def run(config: dict, compiled_graph: Any, init_state: dict, sink: Callable[[dict], None] = None) -> dict:
        """流式执行 harness 图，并把节点进度实时透出给 sink（默认是上层的自定义事件流）。

        直达模式下由 UI 传入自己的 sink，图内产生的自定义事件会原样转发过去。
        """
        sink = sink or _emit_stream_event
        harness_name = config.get("name", "Harness")
        describe = HarnessEngine.describe(config)
        debug_log(
            f"[harness.run] name={harness_name} "
            f"config_stages={[s.get('agent_name') for s in config.get('stages', [])]} "
            f"describe_nodes={describe.get('nodes')} sink={'ui' if sink is not _emit_stream_event else 'default'}"
        )

        # 每次进度事件就是一次「事件边界」：用户点了停止就在这里立刻中断。
        # 这样不必等整条流水线跑完，也不会在执行到一半时把线程硬杀掉。
        raw_sink = sink

        def sink(event: dict) -> None:  # noqa: F811 - 有意包一层做取消检查
            check_cancelled()
            raw_sink(event)

        sink({"type": "start", "harness": harness_name, **describe})

        final_state = init_state
        try:
            if config.get("strategy") == "parallel":
                # parallel 图的节点是异步函数，必须走 astream 才能在同一 superstep 内并发执行
                final_state = asyncio.run(_astream_graph(compiled_graph, init_state, sink))
            else:
                for mode, payload in compiled_graph.stream(
                    init_state,
                    config=_NESTED_GRAPH_CONFIG,
                    stream_mode=["custom", "values"],
                ):
                    if mode == "custom":
                        sink(payload)
                    else:
                        final_state = payload
        except RunCancelled:
            # 用户终止：不再发结束事件，由调用方按「已取消」处理
            raise
        except Exception:
            # 这里用 raw_sink：真正的异常比取消标记更重要，不能被取消检查覆盖
            raw_sink({"type": "end", "harness": harness_name, "status": "error", "reason": "error"})
            raise

        sink({
            "type": "end",
            "harness": harness_name,
            "status": "ok",
            "reason": _completion_reason(config, final_state),
        })
        return final_state


# ── 主智能体可调用的 harness 执行入口 ────────────────────────────────

@tool
def run_harness(
    harness_name: str,
    task: str,
    # 注入的 state 必须是宽松的 dict：用 DeepAgentState（TypedDict）建模时，
    # pydantic 会按字段（如必填的 moderator_decision）校验整个 state，主图里
    # 缺少这些字段就会在工具调用前直接抛 ValidationError。
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """执行一条已入库的 harness 流水线，并把最终汇总答复与产出文件写回当前项目。

    适用场景：任务形态正好匹配某条流水线（顺序加工 / 并行扇出 / 评审循环 / 头脑风暴），
    且用户明确要求按该流水线执行。子智能体看不到本轮对话历史，task 必须自包含。

    Args:
        harness_name: 已注册的流水线名称。
        task: 交给流水线的完整任务描述（目标、输入、期望产出）。

    Returns:
        工具结果里是流水线的最终汇总答复；产出的文件会合并进当前项目的虚拟文件系统。
    """
    config = HarnessStore.get(harness_name) or next(
        (h for h in BUILTIN_HARNESSES if h["name"] == harness_name), None
    )
    if not config:
        available = "、".join(h["name"] for h in HarnessStore.list_all()) or "（无）"
        return Command(update={"messages": [ToolMessage(
            content=f"⚠️ 未找到名为 `{harness_name}` 的 harness 流水线。可用流水线：{available}",
            tool_call_id=tool_call_id,
        )]})

    init_state = {
        "messages": [HumanMessage(content=task)],
        "files": state.get("files", {}),
        "workspace_id": workspace_of(state),
        "todos": state.get("todos", []),
        "execution_log": state.get("execution_log", []),
    }
    debug_log(
        f"[run_harness] name={harness_name} strategy={config.get('strategy')} "
        f"stages={[s.get('agent_name') for s in config.get('stages', [])]}"
    )
    try:
        compiled = HarnessEngine.compile_graph(config)
        final_state = HarnessEngine.run(config, compiled, init_state)
    except RunCancelled:
        # 用户在界面点了停止：给主智能体一条明确的工具结果，而不是让它误以为执行成功
        debug_log(f"[run_harness] 用户终止执行 name={harness_name}")
        return Command(update={"messages": [ToolMessage(
            content=(
                "用户终止了执行：该流水线已被停止，没有产生最终结果。"
                "请直接向用户说明已按要求终止，不要假设流水线已经完成。"
            ),
            tool_call_id=tool_call_id,
        )]})
    except Exception as e:
        debug_log(f"[run_harness] FAILED {type(e).__name__}: {e}")
        return Command(update={"messages": [ToolMessage(
            content=f"⚠️ Harness `{harness_name}` 执行失败：{type(e).__name__}: {e}",
            tool_call_id=tool_call_id,
        )]})

    # 子图首条消息是本次输入，只把最后一条阶段产出作为最终答复返回给主智能体
    replies = [
        m for m in (final_state.get("messages") or [])[1:] if isinstance(m, AIMessage)
    ]
    summary = str(replies[-1].content) if replies else "（流水线没有产生文本输出）"
    return Command(update={
        "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        "files": final_state.get("files", state.get("files", {})),
        "todos": final_state.get("todos", state.get("todos", [])),
        "execution_log": final_state.get("execution_log", state.get("execution_log", [])),
    })


# ── 初始化 ────────────────────────────────────────────────────────────

def initialize_builtin_harnesses():
    """若数据库为空，则用内置 harness 配置进行填充。"""
    for h in BUILTIN_HARNESSES:
        if not HarnessStore.get(h["name"]):
            HarnessStore.save(h)


# 执行数据库填充
initialize_builtin_harnesses()
