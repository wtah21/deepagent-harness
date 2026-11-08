"""
DeepAgent Harness 的 Streamlit 聊天工作区。
主工作区页面，包含聊天执行、侧边栏文件管理器、
计划跟踪器，以及 harness 流水线执行选择器。
"""
import json
import threading
import time
import traceback
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

from graph import get_main_agent
from config import (
    DEFAULT_RECURSION_LIMIT, debug_log, EXPORT_DIR,
    DOCKER_IMAGE, DOCKER_CPUS, DOCKER_MEMORY,
)
from persistence import ProjectStore, AgentStore, HarnessStore, WorkspaceStore
from context_manager import repair_tool_call_pairs
from hitl import ASK_CHOICE
from message_store import archive_messages, flush as flush_archive, load_messages, delete_messages_by_workspace
from mcp_client import (
    delete_mcp_server, load_and_register, mcp_status, probe_server,
    remove_mcp_server, save_mcp_server, validate_server_config, validate_server_name,
)
from a2a_client import (
    delete_agent, probe_agent, reload_agents, remove_agent, save_agent,
    status as a2a_status, suggest_agent_name, validate_agent_name,
)
import docker_runtime
# 进程退出钩子：应用停止时按标签停止本项目所有常驻容器，避免内存泄漏（幂等）
docker_runtime.register_exit_cleanup()
from subagents import AgentFactory
from meta_tools import (
    find_agent_conflict, find_harness_conflict, find_harnesses_referencing,
    validate_agent_config, validate_harness_config,
)
from harness import (
    HarnessEngine, BUILTIN_HARNESSES, sync_builtin_harnesses,
    create_run, update_run, finish_run, snapshot_run, active_run,
    cancel_run, check_cancelled, bind_run, RunCancelled,
)
from styles import inject_premium_styles
from components import (
    file_viewer, execution_timeline, harness_pipeline_progress, harness_diagram,
    harness_strategy_meta, agent_icon, MAIN_AGENT_NODE, MAIN_AGENT_ICON,
)

# 1. Streamlit 页面配置
st.set_page_config(page_title="DeepAgent Workspace", page_icon="🧠", layout="wide")
inject_premium_styles()

# 让内置 harness 模板的更新（如评审标准）生效到数据库记录，每次会话只同步一次
if "harnesses_synced" not in st.session_state:
    sync_builtin_harnesses()
    st.session_state.harnesses_synced = True

# 2. 会话状态初始化
if "project_id" not in st.session_state:
    projects = ProjectStore.list_all()
    if projects:
        latest = projects[0]
        st.session_state.project_id = latest["id"]
        st.session_state.thread_id = latest["thread_id"]
        st.session_state.project_name = latest["name"]
    else:
        new_proj = ProjectStore.create_new("Default Project")
        st.session_state.project_id = new_proj["id"]
        st.session_state.thread_id = new_proj["thread_id"]
        st.session_state.project_name = new_proj["name"]

if "exec_mode" not in st.session_state:
    st.session_state.exec_mode = "Main Orchestrator"
if "selected_harness" not in st.session_state:
    st.session_state.selected_harness = "Code & Test"
# 正在盯着的后台流水线 run id：执行过程由进程级注册表持有，rerun 后靠它重新接上
if "watching_run" not in st.session_state:
    st.session_state.watching_run = None

# 两种执行方式各自一份聊天历史：同属一个项目、共享文件与工作区，
# 但消息按渠道分开归档，切换模式时只显示该模式自己的那份对话。
MODE_MAIN = "Main Orchestrator"
MODE_HARNESS = "Harness Pipeline Workflow"


def _channel_of(mode: str) -> str:
    """执行模式 → 消息归档渠道。"""
    return "harness" if mode == MODE_HARNESS else "main"


current_channel = _channel_of(st.session_state.exec_mode)

def _pending_interrupt(snapshot) -> dict | None:
    """取检查点里待回答的追问（ask_user 暂停时写入的 interrupt 值）。"""
    from langgraph.types import Interrupt  # 局部导入，避免脚本顶部依赖内部类型

    for task in (getattr(snapshot, "tasks", None) or []):
        for item in (getattr(task, "interrupts", None) or []):
            if isinstance(item, Interrupt):
                return dict(item.value or {})
    return None


config = {
    "configurable": {"thread_id": st.session_state.thread_id},
    # 显式指定递归步数上限，否则 LangGraph 会使用默认的 25 步
    "recursion_limit": DEFAULT_RECURSION_LIMIT,
}

# 从检查点存储中获取当前状态
snapshot = get_main_agent().get_state(config)
state_values = snapshot.values if (snapshot and snapshot.values) else {}
# 主智能体停在 ask_user 处时会话是暂停态：此时只渲染追问面板，不接受新的聊天输入
pending_interrupt = _pending_interrupt(snapshot)
# 面板刚作答、正在恢复执行时，图要跑完这一轮才会消掉 interrupt：
# 这期间不再渲染旧面板，否则用户已经提交的追问会一直挂在页面上直到任务结束
if st.session_state.get("pending_resume") is not None:
    pending_interrupt = None

# 当前项目的工作区：即线程 id，文件内容按这个标识分开存放，项目之间互不影响
workspace_id = st.session_state.thread_id

files = state_values.get("files", {})
execution_log = state_values.get("execution_log", [])
# 历史回放以项目数据库里的完整会话为准：上下文压缩会丢弃 state 里的旧消息，
# 归档表保留的是压缩前的原文。按当前执行模式只取该渠道的那一份。
messages = load_messages(workspace_id, current_channel)
if not messages and current_channel == "main" and state_values.get("messages"):
    # 归档表上线前的旧会话：先用检查点历史渲染，并补写一次归档
    messages = state_values.get("messages", [])
    archive_messages(workspace_id, messages, channel="main")

# 旧版把文件内容直接写在 state 与项目记录里，这里把内容搬进工作区表，
# 之后 state 只留轻量索引（迁移完成后文件条目不再出现字符串内容）。
legacy_files = dict((ProjectStore.get(st.session_state.project_id) or {}).get("files") or {})
legacy_files.update(files if isinstance(files, dict) else {})
if any(isinstance(content, str) for content in legacy_files.values()):
    files = WorkspaceStore.import_legacy(workspace_id, legacy_files)
    ProjectStore.save(
        st.session_state.project_id,
        st.session_state.project_name,
        st.session_state.thread_id,
        files=files,
    )
    if snapshot and snapshot.values:
        get_main_agent().update_state(config, {"files": files}, as_node="agent")

# ── 侧边栏：实时计划与文件管理器 ──────────────────────────────────────

def _fmt_size(size) -> str:
    """字节数 → 可读大小。"""
    if not isinstance(size, (int, float)):
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _parse_json_text(text: str, label: str, expected: type, errors: list) -> object:
    """把侧边栏里的 JSON 文本解析成指定类型；空文本返回 None。"""
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(f"{label} 不是合法 JSON：{exc}")
        return None
    if not isinstance(value, expected):
        errors.append(f"{label} 必须是 {'数组' if expected is list else '对象'}")
        return None
    return value

@st.dialog("删除项目", width="small")
def _confirm_delete_project() -> None:
    """删除当前项目的二次确认：文件、对话历史与沙盒目录一并清掉，不可恢复。"""
    st.warning(
        f"确定删除项目「{st.session_state.project_name}」吗？\n\n"
        "该项目的全部文件、对话历史与沙盒目录都会一并删除，且不可恢复。"
    )
    col_cancel, col_ok = st.columns(2)
    if col_cancel.button("取消", width="stretch", key="del-project-cancel"):
        st.rerun(scope="app")
    if col_ok.button("确认删除", type="primary", width="stretch", key="del-project-ok"):
        thread_id = st.session_state.thread_id
        if active_run(thread_id, channel="main") or active_run(thread_id, channel="harness"):
            st.error("该项目仍有任务在运行，请先停止执行再删除。")
            return
        # 容器名随目录名走：先清理容器，再删沙盒目录、对话历史与项目档案
        docker_runtime.remove_container(thread_id)
        WorkspaceStore.delete_workspace(thread_id)
        delete_messages_by_workspace(thread_id)
        ProjectStore.delete(st.session_state.project_id)
        remaining = [
            p for p in ProjectStore.list_all() if p["id"] != st.session_state.project_id
        ]
        # 删的是当前项目：切到最近更新的项目；一个不剩就补一个默认项目
        nxt = remaining[0] if remaining else ProjectStore.create_new("Default Project")
        st.session_state.project_id = nxt["id"]
        st.session_state.thread_id = nxt["thread_id"]
        st.session_state.project_name = nxt["name"]
        st.session_state.watching_run = None
        st.rerun(scope="app")


@st.fragment(run_every=2.0)
def _render_vfs_panel() -> None:
    """虚拟文件系统面板：每 2 秒自刷新一次，harness 执行中新写的文件即时出现。

    数据源合并两份索引——主图 state.files（流水线结束时才统一同步）与工作区
    元数据表（write_file 工具落盘时实时登记）。执行期间后者始终领先，合并后
    不必等执行结束才看到产出；空闲时两份一致，多一次轻量查询而已。
    """
    live_files = WorkspaceStore.manifest(workspace_id)
    state_files = (get_main_agent().get_state(config).values or {}).get("files", {})
    merged = {**state_files, **live_files}
    if not merged:
        st.caption("工作区中还没有文件。")
    else:
        for path, meta in merged.items():
            size = meta.get("size") if isinstance(meta, dict) else None
            mime = meta.get("mime") if isinstance(meta, dict) else None
            title = f"📄 {path}"
            if size is not None:
                title += f"  ·  {_fmt_size(size)}"
            if mime and mime != "text/plain":
                title += f"  ·  {mime}"
            row = st.columns([0.86, 0.14], vertical_alignment="center")
            with row[0]:
                # key 绑定到路径：列表增删后展开状态仍跟着文件走，不会错位到别的文件
                with st.expander(title, expanded=False, key=f"vfs-exp-{path}"):
                    # 文件实体在本地 sandbox 目录，state 里只有元数据索引，内容按需读取
                    content = WorkspaceStore.read(workspace_id, path)
                    if content is None:
                        st.caption("（工作区中没有该文件的内容）")
                    else:
                        file_viewer(path, content)
            with row[1]:
                # 删除：同时清掉 sandbox 实体、state 索引与项目存档，
                # 否则产出文件会在侧边栏里无限堆叠
                if st.button("🗑️", key=f"del-vfs-{path}", help=f"删除 {path}", width="stretch"):
                    WorkspaceStore.delete(workspace_id, path)
                    # 清掉主图 state 里的文件索引。空会话（尚无消息）必须跳过：
                    # update_state 会从 as_node="agent" 继续路由，should_continue
                    # 读取 messages[-1]，空消息列表直接 IndexError。而空会话时
                    # state.files 本来就是空的，没有索引需要清。
                    graph_values = get_main_agent().get_state(config).values or {}
                    if graph_values.get("messages"):
                        get_main_agent().update_state(
                            config, {"files": {path: None}}, as_node="agent"
                        )
                    ProjectStore.save(
                        st.session_state.project_id,
                        st.session_state.project_name,
                        st.session_state.thread_id,
                        files={k: v for k, v in merged.items() if k != path},
                    )
                    st.toast(f"已删除 `{path}`。", icon="🗑️")

        # 一键导出：把虚拟文件系统中的全部文件导入本地（用户浏览确认后的受控通道）
        if st.button("📥 导入到本地", width="stretch", key="export_vfs_to_local"):
            safe_name = "".join(
                c for c in st.session_state.get("project_name", "") if c not in '\\/:*?"<>|'
            ).strip() or workspace_id[:8]
            dest_dir = EXPORT_DIR / safe_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            all_files = WorkspaceStore.read_all(workspace_id)
            exported, skipped = 0, []
            for vpath, content in all_files.items():
                parts = vpath.split("/")
                if any(p in ("..", ".") for p in parts):  # 防止路径逃逸
                    skipped.append(vpath)
                    continue
                dest = dest_dir.joinpath(*parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dest.write_text(content, encoding="utf-8")
                    exported += 1
                except OSError as ex:
                    skipped.append(f"{vpath}（{ex}）")
            msg = f"已将 {exported} 个文件导入本地目录：{dest_dir}"
            if skipped:
                msg += f"；跳过 {len(skipped)} 个：{', '.join(skipped[:3])}"
            st.success(msg)


with st.sidebar:
    st.markdown("<div class='sidebar-section'>当前项目</div>", unsafe_allow_html=True)
    name_col, del_col = st.columns([0.82, 0.18], vertical_alignment="center")
    with name_col:
        st.markdown(
            f"<div style='font-size:17px;font-weight:600;margin-bottom:14px;'>{st.session_state.project_name}</div>",
            unsafe_allow_html=True,
        )
    with del_col:
        if st.button("🗑️", key="del-project", help="删除当前项目", width="stretch"):
            _confirm_delete_project()

    # 项目选择器：列出数据库中的全部项目，选中即切换到对应项目
    all_projects = ProjectStore.list_all()
    project_ids = [p["id"] for p in all_projects]

    def _switch_project() -> None:
        selected = next(
            (p for p in all_projects if p["id"] == st.session_state["project_selector"]),
            None,
        )
        if not selected:
            return
        st.session_state.project_id = selected["id"]
        st.session_state.thread_id = selected["thread_id"]
        st.session_state.project_name = selected["name"]
        # 切换后要盯的是新项目自己的运行（若它在后台还有任务，会自动接上）
        st.session_state.watching_run = None

    st.selectbox(
        "切换项目",
        project_ids,
        index=next(
            (i for i, p in enumerate(all_projects) if p["id"] == st.session_state.project_id),
            0,
        ),
        format_func=lambda pid: next((p["name"] for p in all_projects if p["id"] == pid), pid),
        key="project_selector",
        on_change=_switch_project,
        disabled=not all_projects,
    )

    # 项目重命名折叠面板
    with st.expander("重命名项目", icon=":material/edit:"):
        new_name = st.text_input("项目名称", st.session_state.project_name)
        if st.button("保存名称"):
            # 容器名随目录名走：先按旧名清理容器，再重命名 sandbox 目录
            docker_runtime.remove_container(st.session_state.thread_id)
            ProjectStore.save(st.session_state.project_id, new_name, st.session_state.thread_id, files=files)
            WorkspaceStore.rename_workspace(st.session_state.thread_id, new_name)
            st.session_state.project_name = new_name
            st.rerun()

    st.divider()

    # 虚拟文件系统：fragment 内部每 2 秒刷新，执行中的新文件即时可见
    st.markdown("<div class='sidebar-section'>虚拟文件系统</div>", unsafe_allow_html=True)
    _render_vfs_panel()

    # Docker 容器沙盒状态（每个项目一个常驻容器：1 CPU / 1GB 内存）
    with st.expander("容器沙盒", icon=":material/deployed_code:", expanded=False):
        if docker_runtime.is_docker_available():
            st.caption(
                f"镜像：`{DOCKER_IMAGE}` · 限制：`{DOCKER_CPUS} CPU / {DOCKER_MEMORY}`\n\n"
                "代码执行在该项目的容器内进行，工作区挂载为 `/app/workspace`。"
            )
            st.code(
                f"docker stats {docker_runtime.container_name(workspace_id)}",
                language="bash",
            )
        else:
            st.caption("Docker 不可用：代码执行回退本地临时沙盒。")

    st.divider()

    # MCP 工具接入：连接外部 MCP 服务器，把其工具注入全局工具箱
    with st.expander("MCP 工具", icon=":material/cable:", expanded=False):
        st.caption(
            "连接外部 MCP 服务器后，其工具会注册为 `服务器名__工具名`。"
            "主智能体直接拥有全部 MCP 工具，子智能体按自身 tools 配置取用。"
        )

        # 上一次「重新连接全部」的逐台结果：重连后必须 rerun 才能刷新状态灯，
        # 因此先把汇报暂存进 session_state，等 rerun 之后再展示一次
        reload_report = st.session_state.pop("mcp_reload_report", None)
        if reload_report:
            for level, text in reload_report:
                getattr(st, level)(text)

        mcp_servers = mcp_status().get("servers", {})
        if mcp_servers:
            for mcp_name, info in mcp_servers.items():
                row = st.columns([0.76, 0.24])
                verify = info.get("verify") or {}
                if not info.get("enabled", True):
                    mark = "⚪"
                elif verify.get("ok") is False or not info.get("connected"):
                    mark = "🔴"
                else:
                    mark = "🟢"
                row[0].markdown(
                    f"{mark} `{mcp_name}` · {info.get('transport', '?')} · "
                    f"{info.get('tools', 0)} 个工具"
                )
                if info.get("error") and not info.get("connected"):
                    row[0].caption(f"⚠️ {info['error']}")
                # 只连上并列出工具不代表能用：把最近一次试调用的结果也摆出来
                if verify.get("ok") is False:
                    row[0].caption(
                        f"⚠️ 试调用 `{verify.get('tool')}` 失败：{verify.get('detail')}"
                    )
                elif verify.get("ok"):
                    row[0].caption(f"✅ 试调用 `{verify.get('tool')}` 返回正常")
                elif verify.get("detail"):
                    row[0].caption(f"ℹ️ {verify['detail']}")
                if row[1].button("🗑️", key=f"del-mcp-{mcp_name}", help="删除该 MCP 服务器"):
                    delete_mcp_server(mcp_name)
                    remove_mcp_server(mcp_name)
                    AgentFactory.clear_cache()
                    st.toast(f"已删除 MCP 服务器 `{mcp_name}`。", icon="🗑️")
                    st.rerun()

            if st.button("🔄 重新连接全部", key="mcp_reload_all", width="stretch"):
                with st.spinner("正在重新连接并试调用 MCP 服务器..."):
                    report = load_and_register(verify=True)
                AgentFactory.clear_cache()
                lines: list[tuple[str, str]] = []
                for name, info in report.items():
                    if not info.get("ok"):
                        lines.append(("error", f"`{name}` 连接失败：{info.get('error')}"))
                        continue
                    result = info.get("verify") or {}
                    if result.get("ok") is False:
                        lines.append((
                            "error",
                            f"`{name}` 已连接（{info.get('tools', 0)} 个工具），"
                            f"但试调用 `{result.get('tool')}` 失败：{result.get('detail')}",
                        ))
                    elif result.get("ok"):
                        lines.append((
                            "success",
                            f"`{name}` 验证通过：`{result.get('tool')}` 能正常返回数据。",
                        ))
                    else:
                        lines.append((
                            "info",
                            f"`{name}` 已连接（{info.get('tools', 0)} 个工具），"
                            f"未做试调用验证：{result.get('detail') or '无可用工具'}",
                        ))
                # rerun 之后状态灯才会用上这次试调用的结果，汇报也随之展示
                st.session_state.mcp_reload_report = lines
                st.rerun()
        else:
            st.caption("尚未添加 MCP 服务器。")

        st.caption(f"配置文件：`{mcp_status().get('config_path', '')}`")

        tab_add, tab_examples = st.tabs(["➕ 添加服务器", "📘 配置示例"])

        with tab_examples:
            st.markdown("**stdio**：`command` 必填；`args` / `env` 可选")
            st.code(
                json.dumps({
                    "mcpServers": {
                        "filesystem": {
                            "transport": "stdio",
                            "command": "npx",
                            "args": [
                                "-y", "@modelcontextprotocol/server-filesystem",
                                "C:\\Users\\me\\Desktop",
                            ],
                            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxx"},
                            "enabled": True,
                        }
                    }
                }, ensure_ascii=False, indent=2),
                language="json",
            )
            st.markdown("**sse**：`url` 必填；`headers` 可选")
            st.code(
                json.dumps({
                    "mcpServers": {
                        "remote-tools": {
                            "transport": "sse",
                            "url": "https://mcp.example.com/sse",
                            "headers": {"Authorization": "Bearer mcp_token_xyz123"},
                            "enabled": True,
                        }
                    }
                }, ensure_ascii=False, indent=2),
                language="json",
            )
            st.markdown("**streamable_http**：`url` 必填；`headers` 可选")
            st.code(
                json.dumps({
                    "mcpServers": {
                        "remote-http": {
                            "transport": "streamable_http",
                            "url": "https://mcp.example.com/mcp",
                            "headers": {"X-API-Key": "your-api-key"},
                            "enabled": True,
                        }
                    }
                }, ensure_ascii=False, indent=2),
                language="json",
            )

        with tab_add:
            with st.form("mcp_add_form", clear_on_submit=False):
                mcp_name = st.text_input("服务器名称（必填）", placeholder="filesystem")
                mcp_transport = st.selectbox(
                    "传输方式", ["stdio", "sse", "streamable_http"], key="mcp_transport"
                )
                command = st.text_input("command（stdio 必填）", placeholder="npx")
                args_text = st.text_area(
                    "args（JSON 数组，stdio 可选）",
                    placeholder='["-y", "@modelcontextprotocol/server-filesystem", "C:\\\\Users\\\\me\\\\Desktop"]',
                    height=80,
                )
                env_text = st.text_area(
                    "env（JSON 对象，stdio 可选）",
                    placeholder='{"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxx"}',
                    height=80,
                )
                url = st.text_input(
                    "url（sse / streamable_http 必填）",
                    placeholder="https://mcp.example.com/sse",
                )
                headers_text = st.text_area(
                    "headers（JSON 对象，远程可选）",
                    placeholder='{"Authorization": "Bearer mcp_token_xyz123"}',
                    height=80,
                )
                submitted = st.form_submit_button(
                    "➕ 添加并连接", type="primary", width="stretch"
                )

                if submitted:
                    errors: list[str] = []
                    raw: dict = {"transport": mcp_transport}
                    if mcp_transport == "stdio":
                        raw["command"] = command.strip()
                        parsed_args = _parse_json_text(args_text, "args", list, errors)
                        if parsed_args is not None:
                            raw["args"] = parsed_args
                        parsed_env = _parse_json_text(env_text, "env", dict, errors)
                        if parsed_env is not None:
                            raw["env"] = parsed_env
                    else:
                        raw["url"] = url.strip()
                        parsed_headers = _parse_json_text(headers_text, "headers", dict, errors)
                        if parsed_headers is not None:
                            raw["headers"] = parsed_headers

                    cleaned, name_error = validate_server_name(mcp_name)
                    if name_error:
                        errors.append(name_error)
                    elif cleaned in mcp_servers:
                        errors.append(f"已存在同名 MCP 服务器 `{cleaned}`，请先删除后再添加")
                    normalized, config_errors = validate_server_config(raw)
                    errors.extend(config_errors)

                    if errors:
                        st.error("配置未通过校验：\n" + "\n".join(f"- {e}" for e in errors))
                    else:
                        with st.spinner(f"正在连接 `{cleaned}` 并试调用一个工具..."):
                            probe = probe_server(cleaned, normalized)
                        verify = probe.get("verify") or {}
                        if probe.get("tools_error"):
                            st.error(f"连接失败，配置未保存：{probe['tools_error']}")
                        elif verify.get("ok") is False:
                            st.error(
                                f"`{cleaned}` 能连上并拉到 {len(probe.get('tools') or [])} 个工具，"
                                f"但试调用 `{verify.get('tool')}` 失败，配置未保存：\n\n"
                                f"`{verify.get('detail')}`\n\n"
                                "常见原因：缺少 API key、key 无效或权限不足、"
                                "URL 缺少必需的查询参数。"
                            )
                        else:
                            ok, save_error = save_mcp_server(cleaned, normalized)
                            if not ok:
                                st.error(save_error)
                            else:
                                with st.spinner("正在注册 MCP 工具..."):
                                    report = load_and_register()
                                AgentFactory.clear_cache()
                                result = report.get(cleaned, {})
                                if result.get("ok"):
                                    tool_names = result.get("tool_names") or [
                                        tool_name for tool_name, _ in probe.get("tools") or []
                                    ]
                                    if verify.get("ok"):
                                        note = f"，试调用 `{verify.get('tool')}` 返回正常"
                                    else:
                                        note = "（没有无参工具，未做试调用验证）"
                                    st.toast(
                                        f"MCP 服务器 `{cleaned}` 已添加，注册了 "
                                        f"{len(tool_names)} 个工具{note}。",
                                        icon="🔌",
                                    )
                                    st.rerun()
                                else:
                                    st.error(
                                        "配置已保存，但重新连接失败："
                                        f"{result.get('error') or '未知错误'}"
                                    )

    # A2A 远端 Agent 接入：拉取 Agent Card，把每个 Agent 注册为一个工具
    with st.expander("A2A Agent", icon=":material/handshake:", expanded=False):
        st.caption(
            "填写远端 Agent 的 URL，系统会请求 `URL/.well-known/agent-card.json`，"
            "按名片把每个 Agent 注册为工具 `a2a__名称`。"
            "主智能体直接拥有全部 A2A 工具，子智能体按自身 tools 配置取用。"
        )

        a2a = a2a_status()
        a2a_agents = a2a.get("agents", {})
        if a2a_agents:
            for agent_name, info in a2a_agents.items():
                row = st.columns([0.8, 0.2])
                if not info.get("enabled", True):
                    mark = "⚪"
                elif info.get("registered"):
                    mark = "🟢"
                else:
                    mark = "🔴"
                row[0].markdown(
                    f"{mark} `{agent_name}` · {info.get('protocol', '?')} · "
                    f"{info.get('skills', 0)} 个技能"
                )
                row[0].caption(f"URL：{info.get('url', '')}")
                if info.get("error"):
                    row[0].caption(f"⚠️ {info['error']}")
                if row[1].button("🗑️", key=f"del-a2a-{agent_name}", help="删除该 A2A Agent"):
                    delete_agent(agent_name)
                    remove_agent(agent_name)
                    AgentFactory.clear_cache()
                    st.toast(f"已删除 A2A Agent `{agent_name}`。", icon="🗑️")
                    st.rerun()

            if st.button("🔄 重新加载全部", key="a2a_reload_all", width="stretch"):
                with st.spinner("正在按 Agent Card 重新注册 A2A 工具..."):
                    reload_agents()
                AgentFactory.clear_cache()
                st.toast("A2A Agent 已重新加载。", icon="✅")
                st.rerun()
        else:
            st.caption("尚未添加 A2A Agent。")

        st.caption(f"配置文件：`{a2a.get('config_path', '')}`")
        st.caption(
            "协议兼容：名片含 `supportedInterfaces` 时按 A2A v1.0 调用（`SendMessage`），"
            "否则按 v0.3.0 调用（`message/send`）。"
        )

        with st.form("a2a_add_form", clear_on_submit=False):
            a2a_url = st.text_input(
                "Agent URL（必填）", placeholder="https://agent.example.com"
            )
            a2a_key = st.text_input(
                "API KEY（可选）",
                type="password",
                placeholder="默认按 Agent Card 的 securitySchemes 放置为 Bearer / header / query",
            )
            a2a_name = st.text_input(
                "名称（可选）", placeholder="留空则用 Agent Card 里的 name 自动生成"
            )
            submitted = st.form_submit_button(
                "🔗 添加链接", type="primary", width="stretch"
            )

            if submitted:
                with st.spinner("正在请求 Agent Card..."):
                    profile, connect_error = probe_agent(a2a_url, a2a_key)
                if connect_error:
                    st.error(f"获取 Agent Card 失败，配置未保存：{connect_error}")
                else:
                    final_name = a2a_name.strip() or suggest_agent_name(profile.name)
                    cleaned, name_error = validate_agent_name(final_name)
                    if name_error:
                        st.error(name_error)
                    elif cleaned in a2a_agents:
                        st.error(f"已存在同名 A2A Agent `{cleaned}`，请先删除后再添加")
                    else:
                        ok, save_error = save_agent(cleaned, {
                            "url": a2a_url.strip(),
                            "api_key": a2a_key.strip(),
                            "enabled": True,
                            "protocol_version": profile.protocol,
                            "card": profile.card,
                        })
                        if not ok:
                            st.error(save_error)
                        else:
                            with st.spinner("正在注册 A2A 工具..."):
                                report = reload_agents()
                            AgentFactory.clear_cache()
                            result = report.get(cleaned, {})
                            if result.get("ok"):
                                st.toast(
                                    f"A2A Agent `{cleaned}` 已添加，注册为工具 "
                                    f"`a2a__{cleaned}`。",
                                    icon="🤝",
                                )
                                st.rerun()
                            else:
                                st.error(
                                    "配置已保存，但注册失败："
                                    f"{result.get('error') or '未知错误'}"
                                )

    st.divider()

    # 配置导入：粘贴/上传 JSON → 校验预览 → 确认入库（与主智能体的元工具共用同一套校验器）
    with st.expander("导入配置", icon=":material/upload_file:", expanded=False):
        tab_paste, tab_upload = st.tabs(["📋 粘贴 JSON", "📤 上传文件"])
        raw_text = ""
        with tab_paste:
            raw_text = st.text_area(
                "粘贴 agent / harness 的 JSON 配置：",
                height=180,
                key="import_config_text",
                placeholder='{\n  "name": "spider",\n  ...\n}',
            )
        with tab_upload:
            uploaded = st.file_uploader("选择 .json 配置文件", type=["json"], key="import_config_file")
            if uploaded is not None:
                raw_text = uploaded.getvalue().decode("utf-8", errors="replace")

        if not raw_text.strip():
            st.caption("导入后即可在顶栏 Harness Pipeline 选择栏中调用。")
        else:
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as e:
                parsed = None
                st.error(f"不是合法 JSON：{e}")

            if isinstance(parsed, dict):
                kind = "harness" if "stages" in parsed else "agent"
                st.caption(f"自动识别为 **{kind}** 配置")
                if kind == "harness":
                    normalized, errors = validate_harness_config(parsed)
                    conflict = find_harness_conflict(normalized.get("name", "")) if not errors else None
                else:
                    normalized, errors = validate_agent_config(parsed)
                    conflict = find_agent_conflict(normalized.get("name", "")) if not errors else None

                if errors:
                    st.error("校验未通过：\n" + "\n".join(f"- {e}" for e in errors))
                elif conflict:
                    st.warning(conflict)
                else:
                    st.success("校验通过，入库前请确认规范化后的配置：")
                    st.json(normalized)
                    if st.button("✅ 确认导入", type="primary", width="stretch", key="confirm_import"):
                        if kind == "harness":
                            HarnessStore.save(normalized)
                        else:
                            AgentStore.save(normalized)
                        AgentFactory.clear_cache()
                        st.session_state.pop("import_config_text", None)
                        st.session_state.pop("import_config_file", None)
                        st.toast(f"已导入 {kind} `{normalized['name']}`，可在顶栏选择栏中调用。", icon="✅")
                        st.rerun()
            elif parsed is not None:
                st.error("配置必须是 JSON 对象（{...}）。")

    # 配置管理：内置模板受保护，自定义配置可删除（agent 删除前做引用完整性检查）
    with st.expander("配置管理", icon=":material/tune:", expanded=False):
        st.caption("内置模板受保护不可删除；自定义配置删除后立即从选择栏消失。")

        st.markdown("**智能体**")
        for a in AgentStore.list_all():
            is_builtin = bool(a.get("is_builtin"))
            badge = "🏷️ 内置" if is_builtin else "✏️ 自定义"
            row = st.columns([0.72, 0.28])
            row[0].markdown(f"{a.get('icon', '🤖')} `{a['name']}`（{badge}）")
            if is_builtin:
                row[1].button("🔒", key=f"lock-agent-{a['name']}", disabled=True, help="内置模板不可删除")
            elif row[1].button("🗑️", key=f"del-agent-{a['name']}", help="删除该自定义智能体"):
                refs = find_harnesses_referencing(a["name"])
                if refs:
                    st.error(f"无法删除 `{a['name']}`：仍被 harness（{'、'.join(refs)}）引用，请先处理相关 harness。")
                else:
                    AgentStore.delete(a["name"])
                    AgentFactory.clear_cache()
                    st.toast(f"已删除自定义智能体 `{a['name']}`。", icon="🗑️")
                    st.rerun()

        st.markdown("**Harness 流水线**")
        builtin_harness_names = {bh["name"] for bh in BUILTIN_HARNESSES}
        for h in HarnessStore.list_all():
            is_builtin = h["name"] in builtin_harness_names
            badge = "🏷️ 内置" if is_builtin else "✏️ 自定义"
            strategy_meta = harness_strategy_meta(h.get("strategy"))
            row = st.columns([0.72, 0.28])
            row[0].markdown(
                f"{strategy_meta['icon']} `{h['name']}`（{badge} · {strategy_meta['label']}）"
            )
            if is_builtin:
                row[1].button("🔒", key=f"lock-harness-{h['name']}", disabled=True, help="内置模板不可删除")
            elif row[1].button("🗑️", key=f"del-harness-{h['name']}", help="删除该 harness"):
                HarnessStore.delete(h["name"])
                st.toast(f"已删除 harness `{h['name']}`。", icon="🗑️")
                st.rerun()

    # 会话操作
    if st.button("🔄 Clear State / Restart", width="stretch"):
        # 可以通过写入空状态或新建一个线程来清空状态
        new_proj = ProjectStore.create_new(f"Project - {str(uuid.uuid4())[:8]}")
        st.session_state.project_id = new_proj["id"]
        st.session_state.thread_id = new_proj["thread_id"]
        st.session_state.project_name = new_proj["name"]
        # 新项目对应新的工作区，上一次执行的实时过程不再展示
        st.session_state.watching_run = None
        st.rerun()

# ── 运行时辅助函数 ────────────────────────────────────────────────────

# 思考块里分隔「不同模型调用」的推理片段
_THINK_DIVIDER = "\n\n---\n\n"

def repair_interrupted_tool_calls() -> list:
    """若历史中存在未配对完成的工具调用，就地修正消息顺序后恢复可用。"""
    snapshot = get_main_agent().get_state(config)
    if not snapshot or not snapshot.values:
        return []
    # 停在 ask_user 的追问时，未配对的 tool_call 是「正在等待用户作答」，
    # 强行补一条中断结果会让图再也无法从原位置恢复
    if _pending_interrupt(snapshot):
        return []

    history = snapshot.values.get("messages", [])
    repaired, filled = repair_tool_call_pairs(history)
    if not filled:
        return []

    # 必须整体替换：只是把 ToolMessage 追加到末尾，仍会隔着用户消息，接口照样 400
    get_main_agent().update_state(
        config,
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + repaired},
        as_node="agent",
    )
    return filled


def new_harness_progress() -> dict:
    """初始化 harness 流水线动画的状态。"""
    return {
        "harness_name": "",
        "strategy": "sequential",
        "nodes": [],
        # current 是最近开始执行的节点；active 是全部执行中的节点（并行扇出时会有多个）
        "current": None,
        "active": [],
        "completed": [],
        "round": 1,
        "total_rounds": 1,
        "status": "running",
        "reason": None,
        "failed_node": None,
    }


def apply_harness_event(progress: dict, event: dict) -> None:
    """把一条执行事件合并进动画状态。

    只关心「跑到哪个节点、第几轮、是否结束」：工具调用与中间产出不再进对话界面，
    它们仍完整保存在检查点与项目消息库里，需要排查时从那里取。
    """
    if not isinstance(event, dict):
        return

    event_type = event.get("type")
    if event_type == "start":
        progress.update(
            harness_name=event.get("harness", "Harness"),
            strategy=event.get("strategy", "sequential"),
            nodes=event.get("nodes", []),
            total_rounds=event.get("total_rounds", 1),
            current=None,
            active=[],
            completed=[],
            round=1,
            status="running",
            reason=None,
            failed_node=None,
        )
    elif event_type == "node":
        round_index = event.get("round", 1)
        if round_index != progress.get("round", 1):
            # 进入新一轮：上一轮的完成 / 进行中状态全部作废，所有节点回到未完成
            progress.update(
                round=round_index, completed=[], active=[], current=None, failed_node=None)
        node = event.get("node")
        phase = event.get("phase")
        active = progress.setdefault("active", [])
        completed = progress.setdefault("completed", [])
        if phase == "done":
            if node in active:
                active.remove(node)
            if node and node not in completed:
                completed.append(node)
            progress["current"] = active[-1] if active else None
        elif phase == "start":
            if node and node not in active:
                active.append(node)
            progress["current"] = node
        else:
            # 兼容未带 phase 的旧事件：按串行语义，新节点到达即上一个完成
            previous = progress.get("current")
            if previous and previous not in completed:
                completed.append(previous)
            progress["active"] = [node] if node else []
            progress["current"] = node
    elif event_type == "end":
        active = progress.get("active") or []
        if event.get("status") == "error":
            # 记录失败节点：end 后 current 会被清空，错误态流程条靠它标记 ✕
            progress["failed_node"] = active[-1] if active else progress.get("current")
        else:
            progress["failed_node"] = None
            for node in active:
                if node and node not in progress["completed"]:
                    progress["completed"].append(node)
        progress["active"] = []
        progress["current"] = None
        progress["status"] = "error" if event.get("status") == "error" else "done"
        progress["reason"] = event.get("reason")


def _base_sandbox_state() -> dict:
    """取当前项目线程里的沙箱状态，作为直达执行的输入。"""
    values = get_main_agent().get_state(config).values or {}
    return {
        "files": values.get("files", {}),
        # 子智能体靠这个标识定位到本项目自己的工作区
        "workspace_id": workspace_id,
        "todos": values.get("todos", []),
        "execution_log": values.get("execution_log", []),
    }


def _split_agent_tag(content: str) -> tuple:
    """拆分形如 `[coder] 正文` 的消息，返回 (智能体名, 正文)；无标签时智能体名为空串。"""
    text = str(content or "")
    if text.startswith("["):
        end = text.find("]")
        if end > 0:
            return text[1:end], text[end + 1:].lstrip()
    return "", text


def _persist_user_input(user_input: str) -> None:
    """流水线执行前先归档用户输入：即使执行中途失败，这轮提问也不会丢。

    只写归档表、不写主智能体的检查点：两种执行方式各自独立，
    Harness 的对话不应该出现在主智能体的上下文里。
    """
    archive_messages(workspace_id, [HumanMessage(content=user_input)], channel="harness")


def _persist_failure_notice(detail: str, agent_name: str = "",
                            thread_config: dict | None = None,
                            ws_id: str | None = None,
                            channel: str = "main") -> None:
    """执行失败时把失败原因作为助手消息落盘，保证失败可回放、可追溯。

    后台线程执行时必须显式传入启动时捕获的 thread_config / ws_id：
    执行期间用户可能已经切到别的项目，模块级变量就不再指向本次执行的项目。
    Harness 渠道只归档、不写主智能体的检查点，避免两种执行的历史互相污染。
    """
    body = f"⚠️ 本次执行失败：{detail}"
    message = AIMessage(content=f"[{agent_name}] {body}" if agent_name else body)
    if channel == "main":
        get_main_agent().update_state(
            thread_config or config,
            {"messages": [message]},
            as_node="agent",
        )
    archive_messages(ws_id or workspace_id, [message], channel=channel)


def _persist_direct_result(replies: list, final_state: dict,
                           default_agent: str = "",
                           thread_config: dict | None = None,
                           ws_id: str | None = None,
                           channel: str = "harness") -> None:
    """把直达执行的产出与文件写回项目，让历史、侧边栏与存档保持一致。

    结论只进归档表（该渠道自己的聊天历史），不写主智能体的 messages：
    两种执行方式共享工作区文件，但对话上下文完全独立。
    default_agent 用于给没有 `[智能体] ` 标记的结论补上标记 ——
    历史回放靠这个标记给每条消息配上对应智能体的图标。
    """
    thread_config = thread_config or config
    ws_id = ws_id or workspace_id
    base = get_main_agent().get_state(thread_config).values or {}
    messages = []
    for msg in replies:
        if not isinstance(msg, AIMessage):
            continue
        agent_name, body = _split_agent_tag(msg.content)
        agent_name = agent_name or default_agent
        messages.append(AIMessage(content=f"[{agent_name}] {body}" if agent_name else body))
    if not messages:
        messages.append(AIMessage(content="（本次执行没有产生文本输出）"))

    # 只同步产物状态，不动 messages：主智能体的下一轮不应该看见流水线的对话
    get_main_agent().update_state(
        thread_config,
        {
            "files": final_state.get("files", base.get("files", {})),
            "todos": final_state.get("todos", base.get("todos", [])),
            "execution_log": final_state.get("execution_log", base.get("execution_log", [])),
        },
        # 主智能体是 ReAct 图，其节点名为 "agent"
        as_node="agent",
    )
    # 只归档本轮写回的结论：用户输入已由 _persist_user_input 落盘，
    # harness 子图里的中间稿也不再入库，否则历史里会出现重复的用户消息与阶段稿
    archive_messages(ws_id, messages, channel=channel)


def _persist_project_files(thread_config: dict, ws_id: str,
                           project_id: str, project_name: str) -> None:
    """执行结束后把检查点里的文件索引同步进项目记录（两种模式共用）。"""
    values = get_main_agent().get_state(thread_config).values or {}
    ProjectStore.save(project_id, project_name, ws_id, files=values.get("files", {}))


# 主智能体思考过程在注册表里最多保留这么多字符：模型可能连续推理很久，
# 页面每次快照都要深拷贝一份，不设上限会把内存与刷新耗时都拖上去。
_MAIN_THINK_LIMIT = 60000

# 用户终止执行时写给未完成工具调用的结果：明确告诉模型「没有结果」，
# 否则下一轮它会以为工具已经执行成功并基于假结果继续。
_CANCELLED_TOOL_NOTICE = (
    "用户终止了执行：该工具调用已被中断，没有产生任何结果。"
    "请不要假设它已执行或已成功；如仍需结果，请在用户重新发起后再调用。"
)


def _persist_cancelled_notice(agent_name: str, thread_config: dict, ws_id: str,
                              channel: str = "harness") -> None:
    """把「用户终止了执行」作为一条助手消息落盘，保证历史里能看到终止记录。"""
    body = "⏹ 用户终止了执行，本次运行已停止。"
    message = AIMessage(content=f"[{agent_name}] {body}" if agent_name else body)
    if channel == "main":
        get_main_agent().update_state(thread_config, {"messages": [message]}, as_node="agent")
    archive_messages(ws_id, [message], channel=channel)


def _harness_worker(run_id: str, harness_config: dict, init_state: dict,
                    thread_config: dict, ws_id: str,
                    project_id: str, project_name: str) -> None:
    """后台执行流水线：只写注册表与数据库，不碰任何 st.* 接口。

    这样用户切换流水线 / 模式（触发 Streamlit rerun）时，脚本被重跑，
    但这条线程会继续把任务跑完，界面重绘时从注册表重新接上进度。
    """
    harness_name = harness_config.get("name") or ""
    # 把线程与本次运行绑定：HarnessEngine.run 在每个事件边界检查取消标记
    bind_run(run_id)

    def sink(event: dict) -> None:
        update_run(run_id, lambda run: apply_harness_event(run["progress"], event))

    try:
        graph = HarnessEngine.compile_graph(harness_config)
        final_state = HarnessEngine.run(harness_config, graph, init_state, sink=sink)
    except RunCancelled:
        debug_log(f"[run_direct_harness] 用户终止执行 run={run_id} name={harness_name}")
        update_run(run_id, lambda run: apply_harness_event(
            run["progress"], {"type": "end", "status": "error", "reason": "cancelled"}))
        finish_run(run_id, "cancelled")
        _persist_cancelled_notice(
            harness_name, thread_config, ws_id, channel="harness")
        _persist_project_files(thread_config, ws_id, project_id, project_name)
        flush_archive()
        return
    except Exception as e:
        debug_log(f"[run_direct_harness] FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}")
        # HarnessEngine.run 失败前已发 error end 事件，这里兜底再合并一次，
        # 并把失败原因落盘，重绘后以错误态流程条回放
        update_run(run_id, lambda run: apply_harness_event(
            run["progress"], {"type": "end", "status": "error", "reason": "error"}))
        finish_run(run_id, "error", f"{type(e).__name__}: {e}")
        _persist_failure_notice(
            f"Harness `{harness_name}` 执行失败：{type(e).__name__}: {e}",
            thread_config=thread_config, ws_id=ws_id, channel="harness",
        )
        return

    # harness 子图首条消息是本次输入，其余是各阶段产出的结论；
    # 历史里只保留最终结论，避免把中间稿全部塞进对话记录。
    stage_replies = [
        m for m in (final_state.get("messages") or [])[1:] if isinstance(m, AIMessage)
    ]
    finish_run(run_id, "done")
    _persist_direct_result(
        stage_replies[-1:], final_state, thread_config=thread_config, ws_id=ws_id,
        channel="harness")
    _persist_project_files(thread_config, ws_id, project_id, project_name)
    # 等归档写完再让界面重绘，否则新页面从数据库读不到本次产出
    flush_archive()


def start_harness_run(harness_name: str, user_input: str) -> None:
    """直达 harness 模式：在后台线程里执行所选流水线，界面只负责按注册表重绘进度。"""
    harness_config = HarnessStore.get(harness_name) or next(
        (h for h in BUILTIN_HARNESSES if h["name"] == harness_name), None
    )
    if not harness_config:
        st.error(f"未找到名为 `{harness_name}` 的 Harness。")
        return
    # 同一渠道同时只跑一条：两条流水线写同一个工作区会互相覆盖文件。
    # 两种执行方式彼此独立，主智能体在跑时仍可以再起一条流水线。
    if active_run(workspace_id, channel="harness"):
        st.warning("已有一条流水线在运行，请等它结束或点击停止后再执行。")
        return

    # 执行前先落盘用户输入：即使执行中途失败（如工具调用过多导致异常），消息历史也完整
    _persist_user_input(user_input)
    debug_log(
        f"[run_direct_harness] name={harness_name} "
        f"stages={[s.get('agent_name') for s in harness_config.get('stages', [])]} "
        f"strategy={harness_config.get('strategy')} "
        f"max_review_rounds={harness_config.get('max_review_rounds')}"
    )

    run = create_run(
        workspace_id,
        harness_name,
        str(harness_config.get("strategy") or "sequential"),
        new_harness_progress(),
        channel="harness",
    )
    # 输入状态与 thread 配置必须在脚本线程里取好再交给后台：
    # 线程跑起来之后用户可能已经切了项目，模块级变量就不可信了
    init_state = {"messages": [HumanMessage(content=user_input)], **_base_sandbox_state()}
    threading.Thread(
        target=_harness_worker,
        args=(run["id"], harness_config, init_state, dict(config), workspace_id,
              st.session_state.project_id, st.session_state.project_name),
        name=f"harness-{run['id']}",
        daemon=True,
    ).start()
    # 记住要盯的运行 id：rerun 之后靠它从注册表把进度重新接上
    st.session_state.watching_run = run["id"]


def new_main_progress() -> dict:
    """主智能体的后台进度：只有思考过程与最终答复，没有节点流程条。"""
    return {"thinking": "", "reply": ""}


def _fill_cancelled_tool_calls(thread_config: dict, ws_id: str) -> None:
    """给终止时尚未拿到结果的工具调用补一条「用户终止了执行」的工具消息。

    不补的话，检查点里会留下只有 tool_calls、没有工具结果的回合，
    下一轮请求会被接口直接拒绝（历史无法通过校验）。
    """
    snapshot = get_main_agent().get_state(thread_config)
    values = (snapshot.values if snapshot else {}) or {}
    messages = list(values.get("messages") or [])
    if not messages:
        return
    answered = {
        str(getattr(m, "tool_call_id", "") or "")
        for m in messages if isinstance(m, ToolMessage)
    }
    filled: list[ToolMessage] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for call in (msg.tool_calls or []):
            call_id = str(call.get("id") or "")
            if not call_id or call_id in answered:
                continue
            answered.add(call_id)
            filled.append(ToolMessage(
                content=_CANCELLED_TOOL_NOTICE,
                tool_call_id=call_id,
                name=str(call.get("name") or ""),
            ))
    if not filled:
        return
    get_main_agent().update_state(thread_config, {"messages": filled}, as_node="agent")
    archive_messages(ws_id, filled, channel="main")


def _persist_main_result(thread_config: dict, ws_id: str,
                         project_id: str, project_name: str) -> None:
    """主智能体跑完后把检查点里的消息与文件落盘到主智能体这份历史。"""
    values = get_main_agent().get_state(thread_config).values or {}
    archive_messages(ws_id, values.get("messages") or [], channel="main")
    ProjectStore.save(project_id, project_name, ws_id, files=values.get("files", {}))
    flush_archive()


def _main_agent_worker(run_id: str, payload, thread_config: dict, ws_id: str,
                       project_id: str, project_name: str) -> None:
    """后台执行主智能体：只写注册表与数据库，不碰任何 st.* 接口。

    与 harness 模式一样放到后台线程，用户切换流水线 / 模式或误点选择器都不会
    打断正在跑的任务。执行过程只记录「思考过程 + 最终答复」，不渲染节点流程条。
    """
    bind_run(run_id)
    think: list[str] = []
    reply = ""

    def _push_thinking() -> None:
        text = "".join(think)[-_MAIN_THINK_LIMIT:]
        update_run(run_id, lambda run, value=text: run["progress"].update(thinking=value))

    try:
        stream = get_main_agent().stream(
            payload, config=thread_config, stream_mode=["messages", "updates"]
        )
        for mode, chunk in stream:
            # 每个事件都是一次边界：用户点了停止就立即收尾
            check_cancelled()
            if mode == "messages":
                # messages 流的 chunk 是 (AIMessageChunk, metadata) 元组，
                # 不解包的话 additional_kwargs 永远取不到，思考过程全程为空
                msg_chunk = chunk[0] if isinstance(chunk, tuple) else chunk
                delta = (getattr(msg_chunk, "additional_kwargs", None) or {}).get("reasoning_content")
                if delta:
                    think.append(str(delta))
                    _push_thinking()
                continue
            if mode != "updates" or not isinstance(chunk, dict):
                continue
            for node_name, node_payload in chunk.items():
                # 停在 ask_user 时 updates 里只有 __interrupt__，没有新消息
                if node_name == "__interrupt__" or not isinstance(node_payload, dict):
                    continue
                produced = node_payload.get("messages") or []
                if not isinstance(produced, list):
                    produced = [produced]
                for msg in produced:
                    if not isinstance(msg, AIMessage):
                        continue
                    if msg.content and not (msg.tool_calls or []):
                        reply = str(msg.content)
                        update_run(run_id, lambda run, value=reply: run["progress"].update(reply=value))
                    elif think and think[-1] != _THINK_DIVIDER:
                        # 一步模型输出结束：思考块内用分隔线切到下一步
                        think.append(_THINK_DIVIDER)
                        _push_thinking()
    except RunCancelled:
        debug_log(f"[main_orchestrator] 用户终止执行 run={run_id}")
        finish_run(run_id, "cancelled")
        # 未完成的工具调用补「用户终止了执行」，再落一条可见提示，
        # 否则历史里只剩一条没有回复的用户消息，用户看不出是被自己停掉的
        _fill_cancelled_tool_calls(thread_config, ws_id)
        _persist_cancelled_notice("", thread_config, ws_id, channel="main")
        _persist_main_result(thread_config, ws_id, project_id, project_name)
        return
    except Exception as e:
        debug_log(f"[main_orchestrator] FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finish_run(run_id, "error", f"{type(e).__name__}: {e}")
        _persist_failure_notice(
            f"主智能体执行失败：{type(e).__name__}: {e}",
            thread_config=thread_config, ws_id=ws_id, channel="main",
        )
        return

    finish_run(run_id, "done")
    _persist_main_result(thread_config, ws_id, project_id, project_name)


def start_main_run(turn_input: str | Command) -> None:
    """主智能体编排模式：在后台线程里执行，界面只按注册表呈现思考过程与最终答复。

    turn_input 是新的一轮用户输入；用户回答 ask_user 的追问时传 Command(resume=答案)，
    图会从暂停处继续本轮任务，不会再产生一条新的用户消息。
    """
    if active_run(workspace_id, channel="main"):
        st.warning("主智能体已在执行中，请等它结束或点击停止后再提交。")
        return

    if isinstance(turn_input, Command):
        payload = turn_input
    else:
        user_msg = HumanMessage(content=turn_input)
        # 发送时立即归档用户消息：主智能体的结果要等执行完才落盘，
        # 若不先归档，执行期间的 rerun 回放历史会"丢"这条用户消息
        archive_messages(workspace_id, [user_msg], channel="main")
        payload = {
            "messages": [user_msg],
            # 让主智能体与其委派的子智能体都落在本项目的工作区上
            "workspace_id": workspace_id,
        }

    debug_log(
        "[main_orchestrator] start stream | "
        + ("resume" if isinstance(payload, Command) else f"input={turn_input[:40]}")
    )
    run = create_run(
        workspace_id, MAIN_AGENT_NODE, "main", new_main_progress(), channel="main")
    threading.Thread(
        target=_main_agent_worker,
        args=(run["id"], payload, dict(config), workspace_id,
              st.session_state.project_id, st.session_state.project_name),
        name=f"main-{run['id']}",
        daemon=True,
    ).start()
    st.session_state.watching_run = run["id"]


# ── 主聊天布局 ────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">Multi-agent workspace</div>
        <div class="hero-title">🧠 DeepAgent <span class="gradient-text">Harness Workspace</span></div>
        <p class="hero-sub">拆解任务、调度子智能体与外部工具，把执行过程和产出留在同一个工作区。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 工作区选择器
col1, col2, col3 = st.columns([0.34, 0.42, 0.24])

with col1:
    exec_modes = [MODE_MAIN, MODE_HARNESS]
    # 用 session_state 承载选中值：同时传 default 和 key 会触发 Streamlit 的重置告警
    if "exec_mode_segment" not in st.session_state:
        st.session_state.exec_mode_segment = st.session_state.exec_mode
    selected_mode = st.segmented_control(
        "执行模式",
        exec_modes,
        key="exec_mode_segment",
        label_visibility="collapsed",
    )
    # 分段控件允许取消选择，取消时保持上一次的执行模式
    if selected_mode and selected_mode != st.session_state.exec_mode:
        st.session_state.exec_mode = selected_mode
        # 两种模式各有一份聊天历史，切换后立即重跑，
        # 让页面其余部分（历史回放、进度区）按新模式重新取数
        st.rerun()

with col2:
    if st.session_state.exec_mode == MODE_HARNESS:
        harness_list = HarnessStore.list_all()
        # 选择项展示策略图标与中文名，让不同编排方式在入口处就能区分
        harness_display = {
            h["name"]: (
                f"{harness_strategy_meta(h.get('strategy'))['icon']} {h['name']}"
                f" · {harness_strategy_meta(h.get('strategy'))['label']}"
            )
            for h in harness_list
        }
        all_harnesses = list(harness_display)
        if (
            "harness_selector" not in st.session_state
            or st.session_state.harness_selector not in all_harnesses
        ):
            st.session_state.harness_selector = (
                st.session_state.selected_harness
                if st.session_state.selected_harness in all_harnesses
                else (all_harnesses[0] if all_harnesses else None)
            )
        if all_harnesses:
            st.session_state.selected_harness = st.selectbox(
                "Harness 流水线",
                all_harnesses,
                key="harness_selector",
                label_visibility="collapsed",
                format_func=lambda name: harness_display.get(name, name),
            )
        else:
            st.caption("还没有可用的 harness 流水线。")
    else:
        st.caption("主智能体自动选择工具与子智能体。")

@st.fragment(run_every=1.0)
def render_stat_chips(file_count: int, message_count: int) -> None:
    """右上角的统计胶囊。

    这里每秒自刷新一次，所以「运行中」标记来自后台注册表而不是本次脚本的局部变量：
    用户切换流水线触发 rerun 后，标志依然跟着真实执行状态亮灭，而不是凭空消失。
    """
    # 只显示当前模式的运行标记：两种执行方式相互独立，各看各的
    live = active_run(workspace_id, channel=current_channel)
    live_html = ""
    if live:
        if live.get("cancel_requested"):
            # 已点停止、线程正要收尾：明确告诉用户正在终止，别让人以为按钮没效果
            live_html = (
                "<span class='stat-chip live'>"
                "<span class='run-dot'></span>⏹ 正在终止…</span>"
            )
        else:
            icon = (
                MAIN_AGENT_ICON
                if live.get("channel") == "main"
                else harness_strategy_meta(live.get("strategy"))["icon"]
            )
            elapsed = max(0, int(time.time() - float(live.get("started_at") or time.time())))
            minutes, seconds = divmod(elapsed, 60)
            spent = f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"
            live_html = (
                "<span class='stat-chip live'>"
                "<span class='run-dot'></span>"
                f"{icon} 运行中 · {spent}</span>"
            )
    st.markdown(
        "<div style='display:flex;justify-content:flex-end;align-items:center;gap:6px;padding-top:6px'>"
        + live_html
        + f"<span class='stat-chip'>文件 <strong>{file_count}</strong></span>"
        f"<span class='stat-chip'>消息 <strong>{message_count}</strong></span>"
        "</div>",
        unsafe_allow_html=True,
    )


# 每轮（两条用户消息之间）只有最后一条纯文本 AIMessage 是对用户的正式答复；
# 其余文本都是模型在工具调用间隙写的过程性旁白，不进对话界面。
# 同一轮里各步的思考过程聚合成一段，挂在最终答复上（历史回放时默认收起）。
# 这份扫描要放在右上角统计之前：消息条数按界面实际显示的条数算
# （用户消息 + 正式答复），不含工具调用与工具输出——它们不进对话界面。
final_reply_ids: set[int] = set()
round_reasoning: dict[int, str] = {}
_last_reply = None
_reasoning_buf: list[str] = []


def _flush_round() -> None:
    if _last_reply is None:
        return
    final_reply_ids.add(id(_last_reply))
    merged = _THINK_DIVIDER.join(item for item in _reasoning_buf if item)
    if merged:
        round_reasoning[id(_last_reply)] = merged


for _msg in messages:
    if isinstance(_msg, HumanMessage):
        _flush_round()
        _last_reply = None
        _reasoning_buf = []
    elif isinstance(_msg, AIMessage):
        _text = (_msg.additional_kwargs or {}).get("reasoning_content")
        if isinstance(_text, str) and _text.strip():
            _reasoning_buf.append(_text.strip())
        if str(_msg.content or "").strip() and not (_msg.tool_calls or []):
            _last_reply = _msg
_flush_round()

with col3:
    # 条数 = 用户消息数 + 正式答复数，与下方对话界面的渲染口径一致
    render_stat_chips(
        len(files),
        sum(isinstance(m, HumanMessage) for m in messages) + len(final_reply_ids),
    )

# Harness 模式下展示所选流水线的拓扑：四种策略的图形结构互不相同
if st.session_state.exec_mode == MODE_HARNESS:
    selected_config = next(
        (h for h in HarnessStore.list_all() if h["name"] == st.session_state.selected_harness),
        None,
    )
    if selected_config:
        harness_diagram(selected_config)


# 渲染对话历史：工具调用逐条渲染成独立 expander（标题是工具名，块内分「参数 / 结果」），
# 消息保持 chat_message 格式，避免大段输出平铺把消息挤到页面深处
if not messages:
    if st.session_state.exec_mode == MODE_HARNESS:
        _empty_title = "从一条流水线开始"
        _empty_desc = (
            "描述这次要完成的目标，所选流水线里的子智能体会按节点顺序接力执行，"
            "产出与结论会写进当前项目的工作区。"
        )
    else:
        _empty_title = "从一个任务开始"
        _empty_desc = (
            "描述你要完成的目标，主智能体会拆解任务、挑选子智能体与已接入的 MCP / A2A 工具，"
            "并把执行过程与产出留存到项目工作区。"
        )
    st.markdown(
        f"""
        <div class="empty-state">
            <div style="font-size:30px">🧠</div>
            <h3>{_empty_title}</h3>
            <p>{_empty_desc}</p>
            <div class="hint-row">
                <span class="agent-pill">🔍 调研与抓取</span>
                <span class="agent-pill">💻 代码实现</span>
                <span class="agent-pill">🧪 测试与评审</span>
                <span class="agent-pill">📊 数据处理</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 对话界面只保留「用户输入 + 最终答复」；工具调用与中间过程仍在检查点与项目库里
for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content and id(msg) in final_reply_ids:
            agent_name, body = _split_agent_tag(msg.content)
            # 子智能体产出的消息用该智能体自己的图标；无标记的消息都是主图
            # 直接写入的主智能体回复，统一用主智能体图标，避免回退到默认图标
            avatar = agent_icon(agent_name) if agent_name else MAIN_AGENT_ICON
            with st.chat_message("assistant", avatar=avatar):
                reasoning = round_reasoning.get(id(msg)) or ""
                if reasoning:
                    # 与实时执行时的思考块同一视觉语言，历史里保留为折叠态
                    with st.status("思考过程", state="complete", expanded=False):
                        st.markdown(reasoning)
                st.markdown(body)

# 实时执行过程：数据来自进程级注册表，不依赖本次脚本的局部变量。
# 用户切换流水线 / 模式会触发 rerun，但任务仍在后台线程里继续跑；
# fragment 每秒自刷新一次，任务结束再整体 rerun，让历史回放接上最终产出。
def _stop_button(run_id: str, cancelling: bool = False) -> None:
    """把底部聊天栏的发送按钮换成「停止」按钮。

    Streamlit 的 chat_input 发送按钮无法在 Python 层替换：运行时这里禁用它，
    再由样式表隐藏原生发送按钮，并把下面这个带 key 的容器固定到它的位置上。
    点击后进入「正在终止」态：执行线程要在下一个事件边界才会收尾，
    这个即时反馈能避免用户以为按钮没生效而反复点击。
    """
    st.markdown("<span class='run-marker'></span>", unsafe_allow_html=True)
    with st.container(key="stop_slot"):
        if st.button(
            "…" if cancelling else "⏹",
            key=f"stop_run_{run_id}",
            help="正在终止…" if cancelling else "终止本次执行",
            disabled=cancelling,
        ):
            cancel_run(run_id)
            st.rerun(scope="app")


def _render_harness_run(snap: dict) -> None:
    """Harness 直达模式的实时进度：节点流程条 + 停止按钮。"""
    meta = harness_strategy_meta(snap.get("strategy"))
    running = snap["status"] == "running"
    cancelling = bool(snap.get("cancel_requested"))
    st.divider()
    with st.chat_message("assistant"):
        if not running:
            title = f"**{meta['icon']} {snap['harness_name']}** 执行结束"
        elif cancelling:
            title = f"**{meta['icon']} {snap['harness_name']}** 正在终止…"
        else:
            title = f"**{meta['icon']} {snap['harness_name']}** 正在执行中…"
        st.markdown(title)
        harness_pipeline_progress(snap["progress"])
    if running:
        _stop_button(snap["id"], cancelling)
        return
    # 产出已在后台线程落盘，整体 rerun 后由历史回放呈现，避免过程与结论各显示一遍
    st.session_state.watching_run = None
    st.rerun(scope="app")


def _render_main_run(snap: dict) -> None:
    """主智能体的执行过程：只呈现思考过程与最终答复，不渲染节点流程条。

    思考文本由后台线程实时写进注册表，这里每秒取一次快照重绘；
    任务结束后整体 rerun，改由历史回放展示（含折叠的思考块）。
    """
    progress = snap.get("progress") or {}
    thinking = str(progress.get("thinking") or "")
    reply = str(progress.get("reply") or "")
    running = snap["status"] == "running"
    cancelling = bool(snap.get("cancel_requested"))

    st.divider()
    with st.chat_message("assistant", avatar=MAIN_AGENT_ICON):
        if running:
            elapsed = max(0, int(time.time() - float(snap.get("started_at") or time.time())))
            if cancelling:
                label = "正在终止…"
            elif thinking:
                label = f"正在思考… {elapsed}s"
            else:
                # 模型不一定返回推理内容：没有思考文本时只保留运行中的状态标题，
                # 避免在思考块里留一个空的「…」让人以为卡住了
                label = f"正在执行… {elapsed}s"
            with st.status(
                label,
                expanded=bool(thinking),
                state="running",
            ):
                if thinking:
                    st.markdown(thinking)
        else:
            if thinking:
                with st.status("思考过程", state="complete", expanded=False):
                    st.markdown(thinking)
            if reply:
                st.markdown(reply)
            elif snap["status"] == "cancelled":
                st.caption("⏹ 已按用户要求终止本次执行。")
    if running:
        _stop_button(snap["id"], cancelling)
        return
    st.session_state.watching_run = None
    st.rerun(scope="app")


@st.fragment(run_every=1.0)
def render_active_run() -> None:
    """当前执行模式自己的实时过程。

    进度数据来自进程级注册表，不依赖本次脚本的局部变量：用户切换流水线 / 模式
    会触发 rerun，但任务仍在后台线程里继续跑。两种模式的历史互不混排，
    因此这里只接住当前模式那一条运行。
    """
    run_id = st.session_state.get("watching_run")
    snap = snapshot_run(run_id) if run_id else None
    if snap is not None and snap.get("channel") != current_channel:
        st.session_state.watching_run = None
        snap = None
    if snap is None:
        # 浏览器刷新会丢掉 session_state，这里从注册表接上仍在跑的那一条
        live = active_run(workspace_id, channel=current_channel)
        if not live:
            return
        run_id = st.session_state.watching_run = live["id"]
        snap = live

    if snap.get("channel") == "main":
        _render_main_run(snap)
    else:
        _render_harness_run(snap)


render_active_run()

# ── 追问面板：主智能体停在 ask_user 处时，在这里回答问题 ──────────────
# 任务此时是暂停态：用户只能在面板作答，作答后用 Command(resume=...) 回到
# 被暂停的位置继续本轮任务，答案作为工具结果回传模型，不会多出一条用户消息。
if pending_interrupt and st.session_state.exec_mode != MODE_HARNESS:
    with st.container(border=True):
        st.markdown(f"**{MAIN_AGENT_ICON} 主智能体需要你确认**")
        st.markdown(pending_interrupt.get("question") or "")
        if pending_interrupt.get("context"):
            st.caption(pending_interrupt["context"])

        if pending_interrupt.get("kind") == ASK_CHOICE:
            # 任务有多种方案：把方案列成按钮，点一下即按该方案继续执行
            for idx, option in enumerate(pending_interrupt.get("options") or []):
                if st.button(option, key=f"pending_option_{idx}", width="stretch"):
                    st.session_state.pending_resume = option
                    st.rerun()
            st.caption("选择一个方案后，任务会从暂停处继续执行。")
        else:
            # 缺的是信息：直接让用户把答案写进来
            with st.form(key="pending_input_form", clear_on_submit=True):
                answer = st.text_input(
                    "你的答复",
                    placeholder="直接写下答案，提交后主智能体会从此处继续执行…",
                )
                if st.form_submit_button("提交", width="stretch") and answer.strip():
                    st.session_state.pending_resume = answer.strip()
                    st.rerun()
            st.caption("任务已暂停在此处，提交答复后会从原位置继续执行。")


# 用户输入框：追问未作答时禁用，避免与 resume 值产生歧义
user_input = st.chat_input(
    # 当前渠道在跑时输入框禁用，发送按钮由样式表换成底部的停止按钮
    "正在执行中，点击右下角按钮可终止"
    if active_run(workspace_id, channel=current_channel)
    else ("请先在上方面板作答" if pending_interrupt else "描述任务，或输入要执行的命令…"),
    disabled=bool(pending_interrupt) or bool(
        active_run(workspace_id, channel=current_channel)),
)

# 面板作答：恢复被暂停的图，继续执行本轮任务
resume_value = st.session_state.pop("pending_resume", None)
if resume_value is not None:
    start_main_run(Command(resume=resume_value))
    st.rerun()

if user_input:
    # 1. 渲染用户提示词
    with st.chat_message("user"):
        st.markdown(user_input)

    if active_run(workspace_id, channel=current_channel):
        # 两种执行方式各自独立：只有当前渠道在跑时才拒绝新输入
        st.warning("当前模式已有一条任务在运行，请等它结束或点击停止后再提交。")
    else:
        # 2. 若上一轮运行被中断，先补齐缺失的工具结果，避免会话状态卡死
        repaired_tools = repair_interrupted_tool_calls()
        if repaired_tools:
            st.warning(f"⚠️ 上一轮运行被中断，已补齐工具调用记录：{', '.join(repaired_tools)}")

        # 3. 按执行模式分发（两者都在后台线程里跑，误点选择器不会打断执行）：
        #    Harness  → 直接编译执行流水线，渲染每个节点的执行过程
        #    Main     → 由主智能体编排（拆解 / 分发子智能体 / 整合结果）
        if st.session_state.exec_mode == MODE_HARNESS:
            start_harness_run(st.session_state.selected_harness, user_input)
        else:
            start_main_run(user_input)

        # 4. rerun 进入干净的回放形态，执行过程由进度区从注册表接上
        st.rerun()

# ── 时间线与审计日志 ──────────────────────────────────────────────────
st.divider()
execution_timeline(execution_log)
