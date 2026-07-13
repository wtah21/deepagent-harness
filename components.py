"""
Reusable premium UI components for the DeepAgent Harness Streamlit platform.
Implements custom cards, code viewers, execution logs, and Mermaid charts.
"""
import streamlit as st
from config import TOOL_CATEGORIES, AGENT_ICONS, AGENT_COLORS


def agent_card(agent_config: dict, key: str = "agent") -> None:
    """Render a premium card representing an agent configuration."""
    name = agent_config.get("name", "Unnamed")
    icon = agent_config.get("icon", "🤖")
    color = agent_config.get("color", "#6366f1")
    desc = agent_config.get("description", "No description provided.")
    tools_list = agent_config.get("tools", [])
    is_builtin = agent_config.get("is_builtin", False)

    html_content = f"""
    <div class="glass-card" style="border-left: 4px solid {color};">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 24px;">{icon}</span>
                <span style="font-size: 18px; font-weight: 600; color: #f8fafc;">{name}</span>
            </div>
            {"<span style='background: rgba(99,102,241,0.2); border: 1px solid rgba(99,102,241,0.4); color: #818cf8; font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 500;'>BUILT-IN</span>" if is_builtin else ""}
        </div>
        <p style="font-size: 14px; color: #cbd5e1; margin-bottom: 12px; line-height: 1.4;">{desc}</p>
        <div style="display: flex; flex-wrap: wrap; gap: 6px;">
            {" ".join([f"<span class='agent-pill' style='font-size: 11px; padding: 2px 8px;'>⚙️ {t}</span>" for t in tools_list[:5]])}
            {f"<span class='agent-pill' style='font-size: 11px; padding: 2px 8px;'>+{len(tools_list)-5} more</span>" if len(tools_list) > 5 else ""}
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def harness_diagram(harness_config: dict) -> None:
    """Render a Mermaid flowchart showing the layout of a workflow harness."""
    strategy = harness_config.get("strategy", "sequential")
    stages = harness_config.get("stages", [])

    if not stages:
        st.caption("No stages configured yet.")
        return

    lines = ["graph LR"]
    lines.append("  START([Start])")
    lines.append("  END([End])")

    if strategy == "sequential":
        prev = "START"
        for idx, stage in enumerate(stages):
            node_id = f"stage_{idx}"
            label = f"{stage['agent_name']}\\n({idx+1}. Sequential)"
            lines.append(f'  {node_id}["{label}"]')
            lines.append(f"  {prev} --> {node_id}")
            prev = node_id
        lines.append(f"  {prev} --> END")

    elif strategy == "parallel":
        lines.append("  MERGE{Merge}")
        for idx, stage in enumerate(stages):
            node_id = f"stage_{idx}"
            label = f"{stage['agent_name']}\\n(Parallel)"
            lines.append(f'  {node_id}["{label}"]')
            lines.append(f"  START --> {node_id}")
            lines.append(f"  {node_id} --> MERGE")
        lines.append("  MERGE --> END")

    elif strategy == "review_loop":
        if len(stages) >= 2:
            creator_name = stages[0]["agent_name"]
            reviewer_name = stages[1]["agent_name"]
            lines.append(f'  CREATOR["{creator_name}\\n(Creator)"]')
            lines.append(f'  REVIEWER["{reviewer_name}\\n(Reviewer)"]')

            lines.append("  START --> CREATOR")
            lines.append("  CREATOR --> REVIEWER")
            lines.append('  REVIEWER -->|Revisions Needed| CREATOR')
            lines.append('  REVIEWER -->|Approved| END')
        else:
            lines.append('  CREATOR["Invalid Review Stage"]')
            lines.append("  START --> CREATOR --> END")

    mermaid_code = "\n".join(lines)
    st.components.v1.html(
        f"""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); padding: 16px; border-radius: 12px;">
            <pre class="mermaid" style="background: transparent; display: flex; justify-content: center; margin: 0;">
                {mermaid_code}
            </pre>
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
        </script>
        """,
        height=200,
        scrolling=False
    )


def file_viewer(filename: str, content: str) -> None:
    """Render a virtual file inside an expander with language syntax highlighting."""
    ext = filename.split(".")[-1] if "." in filename else ""
    lang_map = {
        "py": "python", "js": "javascript", "ts": "typescript",
        "html": "html", "css": "css", "md": "markdown",
        "json": "json", "sh": "bash", "toml": "toml", "yaml": "yaml"
    }
    lang = lang_map.get(ext.lower(), "python")
    st.code(content, language=lang)


def execution_timeline(logs: list[dict]) -> None:
    """Display an audit execution log with custom status icons and expandable notes."""
    if not logs:
        st.caption("No events in execution log yet.")
        return

    st.write("### Execution Timeline")
    for log in reversed(logs):
        timestamp = log.get("timestamp", "").split("T")[-1][:8]  # hh:mm:ss
        agent_name = log.get("agent_name", "system")
        action = log.get("action", "")
        summary = log.get("summary", "")
        status = log.get("status", "success")

        icon = "🟢" if status == "success" else "🔴"
        title = f"{icon} **{timestamp}** - `{agent_name}` called tool `{action}`"

        with st.expander(title, expanded=False):
            st.markdown(f"**Action Details:** {summary}")
            st.caption(f"Full timestamp: {log.get('timestamp')}")


def tool_picker(selected_tools: list[str]) -> list[str]:
    """Render structured checkboxes for picking tools by category."""
    from tool_registry import TOOL_REGISTRY

    # Group tools by category
    cats = {}
    for name, meta in TOOL_REGISTRY.items():
        cat = meta.category
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(meta)

    chosen = []
    # Render selectors category-wise
    for cat_name, cat_meta in TOOL_CATEGORIES.items():
        tools_in_cat = cats.get(cat_name, [])
        if not tools_in_cat:
            continue

        st.markdown(f"**{cat_meta.get('icon', '🔧')} {cat_meta.get('label', cat_name)}**")
        st.caption(cat_meta.get("description", ""))

        cols = st.columns(3)
        for i, t in enumerate(tools_in_cat):
            col = cols[i % 3]
            is_checked = t.name in selected_tools
            # Checkbox wrapper
            checked = col.checkbox(
                label=f"`{t.name}`",
                value=is_checked,
                help=f"{t.description} (Risk: {t.risk_level})",
                key=f"tool_pick_{t.name}"
            )
            if checked:
                chosen.append(t.name)
        st.divider()

    return chosen


def status_badge(status: str) -> str:
    """Return an emoji status indicator."""
    return {"completed": "✅", "in_progress": "🔄", "pending": "⬜"}.get(status, "⬜")
