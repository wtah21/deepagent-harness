"""
DeepAgent Harness 的可复用高级 UI 组件。
实现了自定义卡片、代码查看器、执行日志和 Mermaid 图表。
"""
import html
import json

import streamlit as st
from config import AGENT_ICONS, AGENT_COLORS


# 四种编排策略各自的图标与名称：列表、选择器、流程条统一从这里取
HARNESS_STRATEGY_META = {
    "sequential": {"icon": "➡️", "label": "顺序加工", "color": "#3b82f6"},
    "parallel": {"icon": "⚡", "label": "并行扇出", "color": "#f59e0b"},
    "review_loop": {"icon": "🔄", "label": "评审循环", "color": "#ef4444"},
    "brainstorming": {"icon": "💡", "label": "头脑风暴", "color": "#8b5cf6"},
}


def harness_strategy_meta(strategy: str) -> dict:
    """取某种策略的图标与名称；未知策略回退到顺序加工。"""
    return HARNESS_STRATEGY_META.get(
        str(strategy or "sequential"), HARNESS_STRATEGY_META["sequential"]
    )


def harness_diagram(harness_config: dict) -> None:
    """按编排策略渲染结构不同的 Mermaid 流程图。

    四种策略的图形态刻意不同：顺序是直线、并行是分叉汇聚、评审是带回环的链、
    头脑风暴是主持人与角色之间的环形发言，避免所有策略看起来都像顺序执行。
    """
    strategy = harness_config.get("strategy", "sequential")
    stages = harness_config.get("stages", [])
    meta = harness_strategy_meta(strategy)
    if not stages:
        st.caption("尚未配置阶段。")
        return

    if strategy == "sequential":
        lines = ["graph LR", '  START(["🚀 开始"])']
        prev = "START"
        for idx, stage in enumerate(stages):
            node_id = f"stage_{idx}"
            lines.append(
                f'  {node_id}["{agent_icon(stage["agent_name"])} {stage["agent_name"]}"]')
            lines.append(f"  {prev} --> {node_id}")
            prev = node_id
        lines.append(f'  {prev} --> END(["🏁 结束"])')

    elif strategy == "parallel":
        lines = [
            "graph TD",
            '  START(["🚀 开始"])',
            '  DISPATCH{{"⚡ 分发器（内部）"}}',
            '  AGG{{"🧩 合并器（内部）"}}',
            '  END(["🏁 汇总回复"])',
            "  START --> DISPATCH",
        ]
        for idx, stage in enumerate(stages):
            node_id = f"stage_{idx}"
            lines.append(
                f'  {node_id}["{agent_icon(stage["agent_name"])} {stage["agent_name"]}"]')
            lines.append(f"  DISPATCH --> {node_id}")
            lines.append(f"  {node_id} --> AGG")
        lines.append("  AGG --> END")

    elif strategy == "review_loop":
        if len(stages) < 2:
            st.caption("评审循环至少需要创建者与评审者两个阶段。")
            return
        creators, reviewer = stages[:-1], stages[-1]
        lines = ["graph LR", '  START(["🚀 开始"])']
        prev = "START"
        creator_ids = []
        for idx, stage in enumerate(creators):
            node_id = f"creator_{idx}"
            lines.append(
                f'  {node_id}["{agent_icon(stage["agent_name"])} {stage["agent_name"]}"]')
            lines.append(f"  {prev} --> {node_id}")
            prev = node_id
            creator_ids.append(node_id)
        lines.append(f'  REVIEWER{{"🔍 {reviewer["agent_name"]}"}}')
        lines.append(f"  {prev} --> REVIEWER")
        lines.append('  REVIEWER -->|"✅ APPROVED"| END(["🏁 结束"])')
        lines.append(
            f'  REVIEWER -.->|"↺ NEEDS_FIX（最多 {harness_config.get("max_review_rounds", 3)} 轮）"| '
            f"{creator_ids[0]}")

    else:  # brainstorming：主持人在中间，角色轮流发言再回到主持人
        lines = [
            "graph LR",
            '  START(["🚀 开始"])',
            '  MOD(("🎙️ 主持人（内部）"))',
            '  END(["🏁 结论"])',
            "  START --> MOD",
        ]
        for idx, stage in enumerate(stages):
            node_id = f"role_{idx}"
            lines.append(
                f'  {node_id}["{agent_icon(stage["agent_name"])} {stage["agent_name"]}"]')
            lines.append(f"  MOD --> {node_id}")
            lines.append(f"  {node_id} -.->|发言| MOD")
        lines.append("  MOD --> END")

    mermaid_code = "\n".join(lines)

    # 参数区：这条流水线能处理什么任务 + 每个节点（含平台内部编排节点）做什么。
    # 它与流程图放进同一个 iframe 卡片：两者本就是同一条流水线的两种描述，
    # 分成两张卡片会让人误以为它们没有关系。
    purpose = str(harness_config.get("purpose") or "").strip()
    duties = []
    for stage in stages:
        desc = str(stage.get("description") or "").strip()
        if desc:
            duties.append(
                f"<li><strong>{agent_icon(stage['agent_name'])} "
                f"{html.escape(stage['agent_name'])}</strong>：{html.escape(desc)}</li>"
            )
    duties.extend(_internal_duties(strategy))
    spec_html = ""
    if purpose or duties:
        purpose_html = (
            f"<div class='flow-purpose'>🎯 <strong>适用任务</strong>：{html.escape(purpose)}</div>"
            if purpose else ""
        )
        duties_html = (
            "<div class='flow-duties-title'>🧩 节点职责</div>"
            f"<ul class='flow-duties'>{''.join(duties)}</ul>"
            if duties else ""
        )
        spec_html = f"<div class='flow-spec'>{purpose_html}{duties_html}</div>"

    # CSS / JS 用非 f-string 写入，避免所有大括号都要转义
    shell_css = """
    <style>
      html, body { margin: 0; padding: 0; background: transparent; }
      body { font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; }
      .flow-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 12px 14px 14px 14px;
        box-sizing: border-box;
      }
      .flow-head { font-size: 12px; color: #94a3b8; margin-bottom: 8px; }
      .flow-graph {
        background: transparent; display: flex; justify-content: center;
        margin: 0; padding: 0; overflow-x: auto;
        font-size: 12px; color: #64748b;
      }
      .flow-graph svg { max-width: 100%; height: auto; }
      .flow-error { color: #f87171; font-size: 12px; }
      .flow-spec {
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px dashed rgba(148, 163, 184, 0.22);
      }
      .flow-purpose { font-size: 13px; line-height: 1.65; color: #cbd5e1; }
      .flow-purpose strong, .flow-duties li strong { color: #e2e8f0; }
      .flow-duties-title {
        margin-top: 10px; font-size: 12px; font-weight: 600;
        letter-spacing: 0.04em; color: #94a3b8;
      }
      .flow-duties {
        margin: 6px 0 0 0; padding-left: 18px;
        font-size: 12.5px; line-height: 1.75; color: #94a3b8;
      }
    </style>
    """
    # mermaid 异步渲染：定义用 JSON 编码进 JS 字符串（防特殊字符/注入），
    # 页面初始只有一个很小的"正在渲染"占位，渲染完成后再把 SVG 插入 DOM。
    # 不直接放 <pre class='mermaid'> 源码——源码文本先占一大块高度，CDN 渲染成
    # 紧凑 SVG 后高度骤缩，而 st.iframe 的高度测量跟不上收缩，
    # 卡片下方会留下一大截空白（且各策略空白大小随源码行数变化）。
    shell_js = """
    <script type="module">
      // 矫正 documentElement.scrollHeight：视口（iframe）比内容高时，浏览器的
      // html.scrollHeight 返回视口高度而不是内容高度，Streamlit 内置测量脚本
      // 的公式里包含这个值，于是每次 DOM/resize 触发都会上报旧的大高度，
      // iframe 永远缩不回去——切换流水线后卡片下方的大段空白就是这么来的。
      // 用实例 getter 把它矫正成内容真实高度后，内置脚本与本文件末尾的
      // 自报脚本上报的就是同一个正确值，不会再互相覆盖。
      try {
        Object.defineProperty(document.documentElement, 'scrollHeight', {
          configurable: true,
          get: function () {
            const b = document.body;
            return b ? Math.ceil(Math.max(b.getBoundingClientRect().height, b.scrollHeight, b.offsetHeight)) : 0;
          }
        });
      } catch (e) { /* 个别环境不允许覆盖时退回内置测量行为 */ }
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({ startOnLoad: false, theme: 'dark' });
      const definition = """ + json.dumps(mermaid_code) + """;
      const holder = document.getElementById('flow-graph');
      // 高度自报：st.iframe 的内置测量脚本在 srcdoc 更新（切换流水线）后
      // 不再发出 setSize 消息，iframe 高度会卡在上一个流水线图的旧值，
      // 卡片下方因此留出一大截空白。宿主只接受来自 iframe 自身
      // contentWindow 的消息，所以这里从 iframe 内部直接向宿主上报，
      // 通道与内置脚本完全相同（streamlit:iframe:setSize）。
      let lastW = 0, lastH = 0;
      function reportSize() {
        const b = document.body;
        if (!b) return;
        const r = b.getBoundingClientRect();
        // 刻意不用 documentElement 的 scrollHeight/offsetHeight：
        // 视口（iframe）比内容高时它们返回视口高度，上报后高度会被
        // 钉死在旧的大值上，永远缩不回去；body 由内容自然撑开，不受影响
        const h = Math.ceil(Math.max(r.height, b.scrollHeight, b.offsetHeight));
        const w = Math.ceil(b.offsetWidth);
        if (w === lastW && h === lastH) return;
        lastW = w; lastH = h;
        window.parent.postMessage({ type: 'streamlit:iframe:setSize', width: w, height: h }, '*');
      }
      if (window.ResizeObserver) {
        new ResizeObserver(reportSize).observe(document.body);
      }
      window.addEventListener('resize', reportSize);
      window.addEventListener('load', reportSize);
      // 字体/子资源加载晚于渲染完成时高度仍会变，兜底多报几次
      [100, 300, 800].forEach(t => setTimeout(reportSize, t));
      try {
        // 第三个参数把 mermaid 的临时渲染容器钉在 holder 里：渲染过程中
        // mermaid 会在容器里建临时节点，默认挂在 body 上，失败时残留的
        // 错误节点会把 body 撑高，而宿主的高度测量取的就是 body 尺寸，
        // 卡片下方会多出一截空白（切换策略后残留不重置，间距越变越大）。
        const { svg } = await mermaid.render('flow-graph-svg', definition, holder);
        holder.innerHTML = svg;
      } catch (err) {
        const tip = document.createElement('div');
        tip.className = 'flow-error';
        tip.textContent = '流程图渲染失败：' + String((err && err.message) || err);
        holder.textContent = '';
        holder.appendChild(tip);
      }
      reportSize();
    </script>
    """

    card_html = (
        "<div class='flow-card'>"
        + f"<div class='flow-head'>{meta['icon']} {html.escape(meta['label'])}</div>"
        + "<div id='flow-graph' class='flow-graph'>正在渲染流程图…</div>"
        + spec_html
        + "</div>"
    )

    # 高度自适应交给 st.iframe(height="content")：它会在 iframe 内注入测量脚本，
    # 把真实内容高度回传给宿主页面。components.html 只能给固定高度，
    # 装不下时会把靠下的节点与参数区直接裁掉（且没有滚动条提示）。
    st.iframe(shell_css + card_html + shell_js, width="stretch", height="content")
    # 与下方的对话消息拉开距离，避免流程图与第一条消息贴在一起
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


def _internal_duties(strategy: str) -> list[str]:
    """平台内部编排节点（分发器 / 合并器 / 主持人）的职责说明。"""
    if strategy == "parallel":
        return [
            "<li><strong>⚡ 分发器（内部）</strong>：把任务拆成互不依赖的子任务，"
            "并为每个子智能体写出自包含的执行指令。</li>",
            "<li><strong>🧩 合并器（内部）</strong>：核对各分支产出、处理文件冲突，"
            "并生成给用户的最终答复。</li>",
        ]
    if strategy == "brainstorming":
        return [
            "<li><strong>🎙️ 主持人（内部）</strong>：安排发言顺序、挑出分歧点，"
            "达成共识或达到轮次上限时结束会议。</li>",
        ]
    # sequential 没有内部编排节点；review_loop 的评审者本身就是最后一个 stage
    return []


_REASON_TEXT = {
    "approved": "评审通过",
    "consensus": "讨论收敛",
    "max_rounds": "达到轮次上限",
    "completed": "执行完成",
    "error": "执行失败",
}

_ICON_FALLBACK = "🤖"
_ICON_CACHE: dict[str, str] = {}

# 主智能体不是注册智能体，单独给它一个执行过程里的身份与图标
MAIN_AGENT_NODE = "Main Agent"
MAIN_AGENT_ICON = "🧠"


def agent_icon(name: str) -> str:
    """取某个智能体自己的图标；查不到时回退到通用图标。

    每次重绘都会调用，因此结果缓存在模块级字典里，避免频繁查库。
    """
    if not name:
        return _ICON_FALLBACK
    if name == MAIN_AGENT_NODE:
        return MAIN_AGENT_ICON
    if name == "moderator":
        # 头脑风暴的中立主持人不是注册智能体，单独给一个会议主持图标
        return "🎙️"
    if name in _ICON_CACHE:
        return _ICON_CACHE[name]

    from persistence import AgentStore
    from prompts import BUILTIN_TEMPLATES

    config = AgentStore.get(name) or BUILTIN_TEMPLATES.get(name) or {}
    icon = config.get("icon") or _ICON_FALLBACK
    _ICON_CACHE[name] = icon
    return icon


def _node_pill(node: str, active: set, completed: set, status: str, failed_node) -> str:
    """按节点状态渲染一枚流程节点（完成 / 进行中 / 失败 / 未开始）。"""
    if status == "error" and node == failed_node:
        css_class, icon = "harness-node failed", "✕"
    elif status != "running" or node in completed:
        css_class, icon = "harness-node done", "✓"
    elif node in active:
        css_class, icon = "harness-node active", "◐"
    else:
        css_class, icon = "harness-node", "○"
    return f"<span class='{css_class}'>{icon} {agent_icon(node)} {node}</span>"


def harness_pipeline_progress(state: dict) -> None:
    """按编排策略渲染不同形态的实时执行流程条。

    工具调用与中间产出不进对话界面（仍完整保存在检查点与项目库里），
    这里只展示「哪个节点跑到第几轮」：顺序是直线、并行是分叉汇聚、
    评审循环带回环标记、头脑风暴展示主持人与角色的轮转。
    """
    nodes = state.get("nodes") or []
    if not nodes:
        return

    strategy = state.get("strategy", "sequential")
    meta = harness_strategy_meta(strategy)
    # active 可能同时有多个节点（并行扇出各分支同时在跑）；
    # current 只用于兼容没有 active 字段的旧状态
    active = set(state.get("active") or [])
    if not active and state.get("current"):
        active = {state["current"]}
    completed = set(state.get("completed") or [])
    status = state.get("status", "running")
    failed_node = state.get("failed_node")

    def pill(node: str) -> str:
        return _node_pill(node, active, completed, status, failed_node)

    arrow = "<span class='harness-arrow'>→</span>"
    if strategy == "parallel" and len(nodes) >= 3:
        # 分发器 → 并排的并发分支 → 合并器，而不是伪装成一条顺序链
        branches = "".join(pill(node) for node in nodes[1:-1])
        body = (
            pill(nodes[0]) + arrow
            + "<span class='harness-branches'>"
            + f"<span class='harness-branch-label'>⚡ 并行 {len(nodes) - 2} 路</span>"
            + branches
            + "</span>"
            + arrow + pill(nodes[-1])
        )
    elif strategy == "review_loop":
        body = arrow.join(pill(node) for node in nodes)
        body += "<span class='harness-loop'>↺ 未通过回到创建阶段</span>"
    elif strategy == "brainstorming":
        body = "<span class='harness-loop'>⇄</span>".join(pill(node) for node in nodes)
        body += "<span class='harness-loop'>轮流发言</span>"
    else:
        body = arrow.join(pill(node) for node in nodes)

    total_rounds = state.get("total_rounds", 1)
    round_html = ""
    if total_rounds and total_rounds > 1:
        round_html = f"<span class='harness-round'>第 {state.get('round', 1)}/{total_rounds} 轮</span>"

    if status != "running":
        round_html += f"<span class='harness-round'>{_REASON_TEXT.get(state.get('reason'), '已结束')}</span>"

    st.markdown(
        f"""
        <div class="harness-flow">
            <span class="harness-flow-label">{meta['icon']} {state.get('harness_name', 'Harness')} · {meta['label']}</span>
            {body}
            {round_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def file_viewer(filename: str, content: str) -> None:
    """在折叠面板中渲染一个虚拟文件，并带语法高亮。"""
    ext = filename.split(".")[-1] if "." in filename else ""
    lang_map = {
        "py": "python", "js": "javascript", "ts": "typescript",
        "html": "html", "css": "css", "md": "markdown",
        "json": "json", "sh": "bash", "toml": "toml", "yaml": "yaml"
    }
    lang = lang_map.get(ext.lower(), "python")
    st.code(content, language=lang)


def execution_timeline(logs: list[dict]) -> None:
    """以自定义状态图标和可展开说明的形式，展示审计执行日志。"""
    if not logs:
        st.caption("执行日志中还没有事件。")
        return

    st.markdown("#### 执行时间线")
    for log in reversed(logs):
        timestamp = log.get("timestamp", "").split("T")[-1][:8]  # 时:分:秒
        agent_name = log.get("agent_name", "system")
        action = log.get("action", "")
        summary = log.get("summary", "")
        status = log.get("status", "success")

        icon = "🟢" if status == "success" else "🔴"
        title = f"{icon} **{timestamp}** · `{agent_name}` · `{action}`"

        with st.expander(title, expanded=False):
            st.markdown(f"**详情：** {summary}")
            st.caption(f"完整时间戳：{log.get('timestamp')}")
