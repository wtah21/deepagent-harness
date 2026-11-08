"""
并行沙盒工作进程。

parallel 策略下，每个子智能体都在独立进程 + 独立工作区目录中执行：
  - 主进程把本次项目文件复制到临时沙盒目录，并把该目录写进请求；
  - 本进程启动后把 WorkspaceStore 的工作区重定向到沙盒目录，
    因此子智能体的一切读写（含 Docker 执行）都只影响自己那份文件树；
  - 执行结束后对比沙盒目录与基线，把新增/修改的文件内容作为「提案」回传，
    由主进程的汇总节点统一做冲突检测、合并，再写回真正的工作区。

进程间协议：stdout 每行一个 JSON 事件（与 harness 的事件格式一致），
最后一行是 {"type": "final", ...}。为保证协议不被业务输出污染，
本模块启动时会把 sys.stdout 重定向到 stderr，协议事件走保存下来的原始 stdout。

用法：python -m parallel_worker <request.json>
请求字段：agent_name / instructions / workspace_id / sandbox_dir / files / recursion_limit
"""
import sys

# 协议输出通道：真正的 stdout。必须先于其它模块导入完成重定向，
# 这样任何库在导入期打印到 stdout 的内容都不会污染协议；
# 业务代码的 print 等输出统一落到 stderr。
_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

import json
import traceback
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage

from config import DEFAULT_RECURSION_LIMIT
from persistence import WorkspaceStore
from subagents import AgentFactory
from harness import _emit_subagent_message, _STEPS_EXHAUSTED_TEXT

# 与 WorkspaceStore.scan_dir 保持一致的噪音目录/文件过滤规则
_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", "node_modules", ".venv", ".idea"}
_SKIP_SUFFIXES = (".pyc", ".pyo")
_MAX_FILE_SIZE = 1024 * 1024


def _emit(event: dict) -> None:
    """向主进程输出一条 JSON 事件。"""
    _PROTOCOL_OUT.write(json.dumps(event, ensure_ascii=False) + "\n")
    _PROTOCOL_OUT.flush()


def _snapshot(root: Path) -> dict[str, str]:
    """读取沙盒目录下全部文本文件内容，作为变更检测的基线/结果快照。"""
    snapshot: dict[str, str] = {}
    for full in root.rglob("*"):
        if not full.is_file():
            continue
        rel_parts = full.relative_to(root).parts
        if any(part in _SKIP_DIRS for part in rel_parts[:-1]):
            continue
        if full.name.endswith(_SKIP_SUFFIXES):
            continue
        try:
            if full.stat().st_size > _MAX_FILE_SIZE:
                continue
            snapshot[full.relative_to(root).as_posix()] = full.read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
    return snapshot


def _diff(before: dict[str, str], after: dict[str, str]) -> dict[str, str]:
    """找出新增或内容发生变化的文件。"""
    return {path: content for path, content in after.items() if before.get(path) != content}


def run(request: dict, emit: Callable[[dict], None] = _emit) -> None:
    """在沙盒目录中执行一个子智能体任务，并把事件与最终提案回传给主进程。"""
    agent_name = request["agent_name"]
    workspace_id = request["workspace_id"]
    sandbox_dir = Path(request["sandbox_dir"])

    # 把工作区重定向到本进程专属的沙盒目录：之后所有文件工具都只读写这里
    WorkspaceStore.bind_sandbox_root(workspace_id, sandbox_dir)
    baseline = _snapshot(sandbox_dir)

    agent = AgentFactory.get_agent(agent_name)
    final_state: dict = {}
    for mode, payload in agent.stream(
        {
            "messages": [HumanMessage(content=request["instructions"])],
            "files": request.get("files", {}),
            "workspace_id": workspace_id,
            "todos": [],
            "execution_log": [],
        },
        config={"recursion_limit": request.get("recursion_limit", DEFAULT_RECURSION_LIMIT)},
        stream_mode=["updates", "values"],
    ):
        if mode == "values":
            final_state = payload
            continue

        for node_payload in (payload or {}).values():
            produced = (node_payload or {}).get("messages") or []
            if not isinstance(produced, list):
                produced = [produced]
            for msg in produced:
                _emit_subagent_message(msg, agent_name, 1, emit)

    messages = final_state.get("messages") or []
    # 分支的完整执行轨迹（分发指令、工具调用与中间产出）不写项目会话表：
    # 会话表是前端历史回放的唯一来源，写入后会把分发指令与各分支结果铺到界面上。
    # 界面上只保留合并器基于各分支 summary 生成的最终回复。
    summary = str(messages[-1].content) if messages else ""
    if summary.strip() == _STEPS_EXHAUSTED_TEXT:
        # 步数耗尽的占位文案不是真实结论，透传出去会让界面把失败当成
        # 正常答复（与 harness.stream_subagent 的兜底同理），这里显式上报失败。
        emit({
            "type": "final",
            "status": "error",
            "agent_name": agent_name,
            "summary": (
                f"子智能体 {agent_name} 在 {request.get('recursion_limit', DEFAULT_RECURSION_LIMIT)} 步预算内未完成任务，"
                "请把目标拆小或调大 SUBAGENT_RECURSION_LIMIT 后重试。"
            ),
            "files": _diff(baseline, _snapshot(sandbox_dir)),
        })
        return
    emit({
        "type": "final",
        "status": "success",
        "agent_name": agent_name,
        "summary": summary,
        "files": _diff(baseline, _snapshot(sandbox_dir)),
    })


def main() -> int:
    request: dict = {}
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        run(request)
        return 0
    except Exception as e:
        _emit({
            "type": "final",
            "status": "error",
            "agent_name": request.get("agent_name", "unknown"),
            "summary": f"执行失败：{type(e).__name__}: {e}",
            "files": {},
            "traceback": traceback.format_exc(),
        })
        return 1
    finally:
        # 清理本沙盒可能创建的一次性 Docker 容器（未用到 Docker 时静默失败）
        workspace_id = request.get("workspace_id")
        if workspace_id:
            try:
                import docker_runtime
                docker_runtime.remove_container(workspace_id)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
