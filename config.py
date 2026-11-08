"""
DeepAgent Harness 的集中式配置模块。
作为模型设置、路径、工具配置和各项默认值的唯一数据源。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── 路径配置 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "deepagent_data"
DB_PATH = DATA_DIR / "agents.db"
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.db"
EXPORT_DIR = DATA_DIR / "exports"
# 全局 MCP 服务器配置（stdio / sse / streamable_http），由侧边栏维护
MCP_CONFIG_PATH = DATA_DIR / "mcp_servers.json"
# 全局 A2A 远端 Agent 配置（URL / API KEY / 缓存的 Agent Card），由侧边栏维护
A2A_CONFIG_PATH = DATA_DIR / "a2a_agents.json"
# 工作区文件实体目录：每个项目一个子目录，同时是 Docker 容器的挂载源
SANDBOX_DIR = PROJECT_ROOT / "sandbox"

# 自动创建数据目录
DATA_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)
SANDBOX_DIR.mkdir(exist_ok=True)

# ── Docker 隔离执行 ──────────────────────────────────────────
# 每个项目一个常驻容器：--cpus=1.0 -m 1g，挂载 sandbox/<项目> → /app/workspace
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "python:3.11-slim")
DOCKER_CPUS = "1.0"
DOCKER_MEMORY = "1g"
CONTAINER_PREFIX = "harness_ws_"


# ── 调试日志（临时排查用）─────────────────────────────────────
DEBUG_LOG_PATH = PROJECT_ROOT / "ui_debug.log"


def debug_log(msg: str) -> None:
    """写文件日志：终端 stdout 在部分托管环境下不可见，排查卡点用它。"""
    from datetime import datetime

    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%H:%M:%S.%f} | {msg}\n")
    except OSError:
        pass

# ── 模型配置 ──────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "deepseek-v4-flash")
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# ── 上下文管理（混合分层压缩）──────────────────────────────────
# 0 表示调用接口自动探测模型上下文窗口；显式配置时优先使用该值
CONTEXT_WINDOW_TOKENS = int(os.getenv("CONTEXT_WINDOW_TOKENS", "0"))
# 占用超过窗口的该比例时触发压缩
CONTEXT_TRIGGER_RATIO = float(os.getenv("CONTEXT_TRIGGER_RATIO", "0.7"))
# 最近 N 轮完整保留
CONTEXT_RECENT_ROUNDS = int(os.getenv("CONTEXT_RECENT_ROUNDS", "5"))
# 最近 N 轮之内（即 5-15 轮前）做摘要压缩，更早的直接丢弃
CONTEXT_SUMMARY_ROUNDS = int(os.getenv("CONTEXT_SUMMARY_ROUNDS", "15"))

# ── 请求限流 ──────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
REQUESTS_PER_MINUTE_LIMIT = 30

# ── JEV 决策模型（TypeSafe System One）──────────────────────────
JEV_API_KEY = os.getenv("JEV_API_KEY", "")
JEV_BASE_URL = os.getenv("JEV_BASE_URL", "https://api.typesafe.ai/v1")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")

# ── A2A 远端 Agent ────────────────────────────────────────────
# 拉取 Agent Card 的超时（秒）
A2A_CARD_TIMEOUT = int(os.getenv("A2A_CARD_TIMEOUT", "20"))
# 调用远端 Agent 的阻塞等待超时（秒）
A2A_TIMEOUT = int(os.getenv("A2A_TIMEOUT", "120"))

# ── 工具输出治理 ──────────────────────────────────────────────
# 单个工具结果进入上下文前的 token 预算，超过则压缩
TOOL_OUTPUT_TOKEN_BUDGET = int(os.getenv("TOOL_OUTPUT_TOKEN_BUDGET", "3000"))
# 交给 JEV 判断策略时，工具结果保留的字符数
TOOL_OUTPUT_JEV_SAMPLE_CHARS = int(os.getenv("TOOL_OUTPUT_JEV_SAMPLE_CHARS", "1000"))
# 结构化截断时完整保留的前几条记录
TOOL_OUTPUT_HEAD_RECORDS = int(os.getenv("TOOL_OUTPUT_HEAD_RECORDS", "5"))
# JEV 置信度低于该值时改用本地启发式判断（JEV 的 confidence 为两个候选的概率差）
TOOL_OUTPUT_JEV_MIN_CONFIDENCE = float(os.getenv("TOOL_OUTPUT_JEV_MIN_CONFIDENCE", "0.2"))
# 单次 JEV 调用的超时（秒）。JEV 只是压缩策略的选择器，超时后本地启发式同样可用，
# 因此不需要给网络留太长时间——每次超长输出都干等会直接把整条流水线拖慢。
TOOL_OUTPUT_JEV_TIMEOUT = int(os.getenv("TOOL_OUTPUT_JEV_TIMEOUT", "6"))
# 过滤代码在沙箱中的执行超时（秒）
TOOL_OUTPUT_FILTER_TIMEOUT = int(os.getenv("TOOL_OUTPUT_FILTER_TIMEOUT", "30"))

# ── 智能体默认配置 ────────────────────────────────────────────
DEFAULT_RECURSION_LIMIT = 200
DEFAULT_TOOLS = ["ls", "read_file", "write_file", "edit_file", "execute_python", "write_todos"]
# 子智能体（ReAct 循环）专用递归上限。子图与外层 harness 图各自独立计数，
# 而 ReAct 每轮"模型节点→工具节点"消耗 2 个超步：200 步只够约 100 次工具调用。
# 实测 Spec & Build 的 coder 单 stage 写 21 个文件（约 130KB 代码）就用满了 200 步
# 且离完成只差几步——按每次调用约 3 步（读/写/验证）估算，600 步给足约 300 次调用。
SUBAGENT_RECURSION_LIMIT = int(os.getenv("SUBAGENT_RECURSION_LIMIT", "600"))

# ── 执行安全 ──────────────────────────────────────────────────
DANGEROUS_TOOLS = ["shell_command", "install_package"]
EXECUTION_TIMEOUT = 30
MAX_OUTPUT_LENGTH = 5000

# ── UI 配置 ───────────────────────────────────────────────────
APP_NAME = "DeepAgent Harness"
APP_ICON = "\U0001f9e0"
APP_VERSION = "1.0.0"

# ── 工具分类 ──────────────────────────────────────────────────
TOOL_CATEGORIES = {
    "filesystem": {"icon": "\U0001f4c1", "label": "Filesystem", "description": "Read, write, and manage virtual files"},
    "execution": {"icon": "\u26a1", "label": "Code Execution", "description": "Run code and shell commands"},
    "web": {"icon": "\U0001f310", "label": "Web & Search", "description": "Search the web and fetch URLs"},
    "planning": {"icon": "\U0001f4cb", "label": "Planning", "description": "Task planning and tracking"},
    "delegation": {"icon": "\U0001f916", "label": "Delegation", "description": "Delegate to other agents"},
    "meta": {"icon": "\U0001f527", "label": "Meta", "description": "Create agents and harnesses"},
    "mcp": {"icon": "\U0001f50c", "label": "MCP", "description": "Tools provided by external MCP servers"},
    "a2a": {"icon": "\U0001f91d", "label": "A2A", "description": "Remote agents connected over the A2A protocol"},
}

# ── 智能体图标 ────────────────────────────────────────────────
AGENT_ICONS = ["\U0001f916", "\U0001f9e0", "\U0001f4bb", "\U0001f50d", "\U0001f6e1\ufe0f", "\U0001f4dd", "\U0001f9ea", "\U0001f3a8", "\u2699\ufe0f", "\U0001f52c", "\U0001f4ca", "\U0001f680", "\U0001f512", "\U0001f4da", "\U0001f41b", "\U0001f3d7\ufe0f"]

AGENT_COLORS = [
    "#6366f1", "#8b5cf6", "#a855f7", "#d946ef",
    "#ec4899", "#f43f5e", "#ef4444", "#f97316",
    "#eab308", "#84cc16", "#22c55e", "#14b8a6",
    "#06b6d4", "#0ea5e9", "#3b82f6", "#6366f1",
]
