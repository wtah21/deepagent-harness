"""
DeepAgent Harness 的高级视觉主题与自定义 CSS 注入。

配色与字体由 .streamlit/config.toml 统一管理，这里只补充原生主题表达不了的部分：
玻璃拟态卡片、渐变标题、Harness 流程节点动画、聊天气泡与滚动条细节。
"""
import streamlit as st


def inject_premium_styles():
    """向 Streamlit 应用注入深色玻璃拟态样式。"""
    st.markdown(
        """
        <style>
        :root {
            --brand: #6366f1;
            --brand-soft: #818cf8;
            --brand-ghost: rgba(99, 102, 241, 0.12);
            --surface: rgba(30, 41, 59, 0.55);
            --surface-strong: rgba(19, 28, 49, 0.85);
            --line: rgba(148, 163, 184, 0.14);
            --text-dim: #94a3b8;
        }

        /* 玻璃拟态卡片 */
        .glass-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 16px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(2, 6, 23, 0.35);
            transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
        }

        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.35);
            box-shadow: 0 14px 36px rgba(79, 70, 229, 0.16);
            transform: translateY(-2px);
        }

        /* 智能体 / 工具标签 */
        .agent-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 500;
            color: #cbd5e1;
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid var(--line);
        }

        /* 顶部标题区 */
        .hero {
            padding: 4px 0 10px 0;
        }

        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-dim);
            margin-bottom: 6px;
        }

        .hero-title {
            font-size: 34px;
            font-weight: 700;
            line-height: 1.2;
            letter-spacing: -0.02em;
            margin: 0 0 6px 0;
        }

        .hero-sub {
            font-size: 14px;
            color: var(--text-dim);
            margin: 0;
        }

        .gradient-text {
            background: linear-gradient(120deg, #818cf8 0%, #c084fc 55%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }

        /* 顶部状态标签 */
        .stat-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 500;
            color: #cbd5e1;
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid var(--line);
            margin-right: 6px;
            /* 窄屏下不允许折行压扁：保持胶囊形状，必要时整体收缩 */
            white-space: nowrap;
            flex-shrink: 0;
        }

        .stat-chip strong {
            color: #e2e8f0;
            font-weight: 600;
        }

        /* 运行态胶囊：一条流水线还在后台执行时点亮，随执行状态自动消失 */
        .stat-chip.live {
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.12);
            border-color: rgba(34, 197, 94, 0.35);
        }

        .stat-chip.live .run-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #22c55e;
            animation: run-pulse 1.4s ease-in-out infinite;
        }

        @keyframes run-pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.35; transform: scale(0.8); }
        }

        /* 运行态：把聊天输入栏的发送按钮换成停止按钮 ─────────────────
           chat_input 的发送按钮无法在 Python 层替换，运行时先禁用它，
           再隐藏原生按钮、把带 key 的停止按钮固定到它的位置上。
           实测发送按钮右边缘距视口右约 100px、中心距视口底约 79px，
           且这两个距离不随窗口宽度变化，因此用固定值对齐。 */
        body:has(.run-marker) button[data-testid="stChatInputSubmitButton"] {
            display: none !important;
        }

        .st-key-stop_slot {
            position: fixed !important;
            right: 96px;
            bottom: 60px;
            z-index: 1000001;
            width: auto !important;
        }

        .st-key-stop_slot [data-testid="stButton"],
        .st-key-stop_slot .stButton {
            width: auto !important;
        }

        .st-key-stop_slot button {
            width: 2.5rem;
            height: 2.5rem;
            min-height: 2.5rem;
            padding: 0;
            border-radius: 50%;
            border: 1px solid rgba(239, 68, 68, 0.6);
            background: rgba(239, 68, 68, 0.2);
            color: #fecaca;
            font-size: 14px;
            line-height: 1;
            box-shadow: 0 4px 14px rgba(2, 6, 23, 0.45);
        }

        .st-key-stop_slot button:hover {
            background: rgba(239, 68, 68, 0.38);
            border-color: #ef4444;
            color: #ffffff;
        }

        /* 侧边栏分组标题 */
        .sidebar-section {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-dim);
            margin: 6px 0 8px 0;
        }

        /* 空状态引导卡 */
        .empty-state {
            text-align: center;
            padding: 44px 24px;
            border: 1px dashed rgba(148, 163, 184, 0.22);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.45), rgba(15, 23, 42, 0.2));
            margin: 8px 0 20px 0;
        }

        .empty-state h3 {
            margin: 10px 0 6px 0;
            font-size: 18px;
            color: #e2e8f0;
        }

        .empty-state p {
            margin: 0 auto;
            max-width: 560px;
            font-size: 14px;
            color: var(--text-dim);
            line-height: 1.6;
        }

        .empty-state .hint-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 8px;
            margin-top: 18px;
        }

        /* 时间线 */
        .timeline-item {
            padding-left: 20px;
            border-left: 2px solid #334155;
            position: relative;
            margin-bottom: 12px;
        }

        .timeline-item::before {
            content: '';
            width: 8px;
            height: 8px;
            background: var(--brand);
            border-radius: 50%;
            position: absolute;
            left: -5px;
            top: 6px;
        }

        /* Harness 流水线实时进度动画 */
        .harness-flow {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            background: var(--surface-strong);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 12px;
        }

        .harness-flow-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-dim);
            letter-spacing: 0.02em;
            margin-right: 4px;
        }

        .harness-node {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-dim);
            background: rgba(148, 163, 184, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.12);
            transition: all 0.3s ease;
        }

        .harness-node.done {
            color: #6ee7b7;
            background: rgba(16, 185, 129, 0.08);
            border-color: rgba(16, 185, 129, 0.45);
        }

        .harness-node.active {
            color: #e0e7ff;
            background: rgba(99, 102, 241, 0.16);
            border-color: var(--brand);
            animation: harness-pulse 1.4s ease-in-out infinite;
        }

        .harness-node.failed {
            color: #fca5a5;
            background: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.5);
        }

        @keyframes harness-pulse {
            0%   { box-shadow: 0 0 6px rgba(99, 102, 241, 0.25); }
            50%  { box-shadow: 0 0 22px rgba(99, 102, 241, 0.7); }
            100% { box-shadow: 0 0 6px rgba(99, 102, 241, 0.25); }
        }

        .harness-arrow {
            color: #475569;
            font-size: 12px;
        }

        /* 并行策略：把并发分支装进一个虚线分组，和线性箭头区分开 */
        .harness-branches {
            display: inline-flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
            padding: 4px 10px;
            border: 1px dashed rgba(148, 163, 184, 0.35);
            border-radius: 10px;
            background: rgba(245, 158, 11, 0.04);
        }

        .harness-branch-label {
            font-size: 11px;
            color: var(--text-dim);
            letter-spacing: 0.03em;
        }

        /* 评审循环与头脑风暴：回环 / 轮转提示 */
        .harness-loop {
            font-size: 11px;
            color: var(--brand-soft);
        }

        /* 流程图下方的配置参数：适用任务 + 节点职责 */
        .harness-spec {
            margin: 10px 0 4px 0;
            padding: 12px 14px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: rgba(148, 163, 184, 0.04);
        }

        .harness-purpose {
            font-size: 13px;
            line-height: 1.65;
            color: #cbd5e1;
        }

        .harness-purpose strong {
            color: #e2e8f0;
        }

        .harness-duties-title {
            margin-top: 10px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.04em;
            color: var(--text-dim);
        }

        .harness-duties {
            margin: 6px 0 0 0;
            padding-left: 18px;
            font-size: 12.5px;
            line-height: 1.75;
            color: var(--text-dim);
        }

        .harness-duties li strong {
            color: #e2e8f0;
            font-weight: 600;
        }

        .harness-round {
            margin-left: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--brand-soft);
        }

        /* 聊天气泡 */
        [data-testid="stChatMessage"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: rgba(99, 102, 241, 0.08);
            border-color: rgba(99, 102, 241, 0.18);
        }

        /* 折叠面板：更轻的边框与更清晰的标题 */
        [data-testid="stExpander"] {
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
            background: rgba(15, 23, 42, 0.35);
            margin-bottom: 8px;
            overflow: hidden;
        }

        [data-testid="stExpander"] summary {
            font-weight: 500;
            color: #cbd5e1;
            padding: 8px 12px;
        }

        [data-testid="stExpander"] summary:hover {
            color: #f1f5f9;
            background: rgba(99, 102, 241, 0.06);
        }

        /* 分段控件与标签页 */
        [data-testid="stSegmentedControl"] div[role="radiogroup"] {
            gap: 4px;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 4px;
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid var(--line);
        }

        [data-testid="stTabs"] button[role="tab"] {
            border-radius: 8px 8px 0 0;
        }

        /* 输入区聚焦发光 */
        [data-testid="stChatInput"] {
            border-radius: 12px;
        }

        [data-testid="stChatInput"]:focus-within {
            box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.5), 0 6px 24px rgba(79, 70, 229, 0.18);
            border-radius: 12px;
        }

        /* 滚动条 */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.25);
            border-radius: 999px;
            border: 2px solid transparent;
            background-clip: content-box;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(148, 163, 184, 0.4);
            background-clip: content-box;
        }

        /* 隐藏 Streamlit 默认页脚与菜单 */
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
