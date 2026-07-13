"""
Streamlit Chat Workspace for DeepAgent Harness.
Main workspace page with chat execution, sidebar file manager,
planning tracker, and agent/harness runtime execution selector.
"""
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph import deepagent
from persistence import ProjectStore, AgentStore, HarnessStore
from subagents import AgentFactory
from harness import HarnessEngine, BUILTIN_HARNESSES
from styles import inject_premium_styles
from components import file_viewer, status_badge, execution_timeline

# 1. Streamlit configuration
st.set_page_config(page_title="DeepAgent Workspace", page_icon="🧠", layout="wide")
inject_premium_styles()

# 2. Session State Initialization
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
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "coder"
if "selected_harness" not in st.session_state:
    st.session_state.selected_harness = "Code & Review"

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Retrieve current state from checkpointer
snapshot = deepagent.get_state(config)
state_values = snapshot.values if (snapshot and snapshot.values) else {}

todos = state_values.get("todos", [])
files = state_values.get("files", {})
execution_log = state_values.get("execution_log", [])
messages = state_values.get("messages", [])

# ── Sidebar: Live Plan & File Manager ─────────────────────────────────

with st.sidebar:
    st.markdown(f"### 📁 Project: **{st.session_state.project_name}**")
    
    # Project Rename Expander
    with st.expander("✏️ Rename Project"):
        new_name = st.text_input("New project name:", st.session_state.project_name)
        if st.button("Update Name"):
            ProjectStore.save(st.session_state.project_id, new_name, st.session_state.thread_id, files=files)
            st.session_state.project_name = new_name
            st.rerun()

    st.divider()

    # Plan
    st.markdown("### 📋 Active Plan")
    if not todos:
        st.caption("No active tasks in plan.")
    else:
        for t in todos:
            icon = status_badge(t.get("status", "pending"))
            st.markdown(f"{icon} {t.get('content')}")

    st.divider()

    # Virtual Filesystem
    st.markdown("### 📄 Virtual Filesystem")
    if not files:
        st.caption("No files created in sandbox.")
    else:
        for path, content in files.items():
            with st.expander(f"📄 {path}", expanded=False):
                file_viewer(path, content)
                
    st.divider()

    # Session actions
    if st.button("🔄 Clear State / Restart", use_container_width=True):
        # We can clear state by writing an empty state or creating a new thread
        new_proj = ProjectStore.create_new(f"Project - {str(uuid.uuid4())[:8]}")
        st.session_state.project_id = new_proj["id"]
        st.session_state.thread_id = new_proj["thread_id"]
        st.session_state.project_name = new_proj["name"]
        st.rerun()

# ── Main Chat Layout ──────────────────────────────────────────────────

st.markdown("# 🧠 DeepAgent <span class='gradient-text'>Harness Workspace</span>", unsafe_allow_html=True)
st.caption("Enterprise-grade multi-agent coding platform running on free tier models.")

# Workspace Selectors
col1, col2, col3 = st.columns(3)

with col1:
    exec_modes = ["Main Orchestrator", "Direct Agent Direct-Chat", "Harness Pipeline Workflow"]
    st.session_state.exec_mode = st.selectbox(
        "Execution Mode", 
        exec_modes, 
        index=exec_modes.index(st.session_state.exec_mode) if st.session_state.exec_mode in exec_modes else 0
    )

with col2:
    if st.session_state.exec_mode == "Direct Agent Direct-Chat":
        all_agents = [a["name"] for a in AgentStore.list_all()]
        st.session_state.selected_agent = st.selectbox("Select Agent Target", all_agents, index=0 if all_agents else None)
    elif st.session_state.exec_mode == "Harness Pipeline Workflow":
        all_harnesses = [h["name"] for h in HarnessStore.list_all()]
        st.session_state.selected_harness = st.selectbox("Select Harness Pipeline", all_harnesses, index=0 if all_harnesses else None)
    else:
        st.info("💡 Main agent routes tools automatically.")

with col3:
    # Stats / Quick Action
    st.metric(label="Sandbox Virtual Files", value=len(files))

# Render conversation history
st.divider()
for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content:
            with st.chat_message("assistant"):
                st.markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        if msg.content and len(str(msg.content).strip()) > 0:
            with st.expander(f"🔧 Tool execution: `{msg.name or 'tool'}`", expanded=False):
                st.code(str(msg.content)[:2000] + ("..." if len(str(msg.content)) > 2000 else ""))

# User Prompt Input
user_input = st.chat_input("Explain the task or provide commands here...")

if user_input:
    # 1. Render user prompt
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Assistant execution
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("DeepAgent is running..."):
            try:
                if st.session_state.exec_mode == "Main Orchestrator":
                    # Run standard orchestrator stream
                    for chunk in deepagent.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                        stream_mode="values",
                    ):
                        if "messages" in chunk and chunk["messages"]:
                            last_msg = chunk["messages"][-1]
                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                full_response = last_msg.content
                                placeholder.markdown(full_response)
                            elif isinstance(last_msg, ToolMessage):
                                st.write(f"🔧 executed tool: `{last_msg.name}`")

                elif st.session_state.exec_mode == "Direct Agent Direct-Chat":
                    # Direct chat sends a direct request to a specific agent using the task tool internally
                    agent_name = st.session_state.selected_agent
                    # Format as task call in the main agent flow to keep graph trace aligned
                    payload = f"task('{agent_name}', '{user_input}')"
                    for chunk in deepagent.stream(
                        {"messages": [HumanMessage(content=f"Delegate to {agent_name}: {user_input}")]},
                        config=config,
                        stream_mode="values",
                    ):
                        if "messages" in chunk and chunk["messages"]:
                            last_msg = chunk["messages"][-1]
                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                full_response = last_msg.content
                                placeholder.markdown(full_response)

                elif st.session_state.exec_mode == "Harness Pipeline Workflow":
                    # Run target harness pipeline
                    harness_name = st.session_state.selected_harness
                    for chunk in deepagent.stream(
                        {"messages": [HumanMessage(content=f"Run harness pipeline '{harness_name}' with prompt: {user_input}")]},
                        config=config,
                        stream_mode="values",
                    ):
                        if "messages" in chunk and chunk["messages"]:
                            last_msg = chunk["messages"][-1]
                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                full_response = last_msg.content
                                placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"Error during runtime execution: {str(e)}")

        # Save files to persistence store
        final_snapshot = deepagent.get_state(config)
        if final_snapshot and final_snapshot.values:
            ProjectStore.save(
                st.session_state.project_id,
                st.session_state.project_name,
                st.session_state.thread_id,
                files=final_snapshot.values.get("files", {})
            )
        st.rerun()

# ── Timeline & Audit logs ────────────────────────────────────────────
st.divider()
execution_timeline(execution_log)
