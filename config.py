"""
Centralized configuration for DeepAgent Harness.
Single source of truth for model settings, paths, tool configs, and defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "deepagent_data"
DB_PATH = DATA_DIR / "agents.db"
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.db"
EXPORT_DIR = DATA_DIR / "exports"

# Auto-create data directories
DATA_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# ── Model Configuration ───────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEEPAGENT_MODEL", "llama-3.3-70b-versatile")
DEFAULT_MAX_TOKENS = int(os.getenv("DEEPAGENT_MAX_TOKENS", "8000"))
DEFAULT_TEMPERATURE = float(os.getenv("DEEPAGENT_TEMPERATURE", "0.1"))

# ── Rate Limiting ─────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
REQUESTS_PER_MINUTE_LIMIT = 30

# ── Agent Defaults ────────────────────────────────────────────
DEFAULT_RECURSION_LIMIT = 50
DEFAULT_TOOLS = ["ls", "read_file", "write_file", "edit_file", "execute_python", "write_todos"]

# ── Execution Safety ──────────────────────────────────────────
DANGEROUS_TOOLS = ["shell_command", "save_to_disk", "install_package"]
EXECUTION_TIMEOUT = 30
MAX_OUTPUT_LENGTH = 5000

# ── UI Configuration ──────────────────────────────────────────
APP_NAME = "DeepAgent Harness"
APP_ICON = "\U0001f9e0"
APP_VERSION = "1.0.0"

# ── Tool Categories ───────────────────────────────────────────
TOOL_CATEGORIES = {
    "filesystem": {"icon": "\U0001f4c1", "label": "Filesystem", "description": "Read, write, and manage virtual files"},
    "execution": {"icon": "\u26a1", "label": "Code Execution", "description": "Run code and shell commands"},
    "web": {"icon": "\U0001f310", "label": "Web & Search", "description": "Search the web and fetch URLs"},
    "planning": {"icon": "\U0001f4cb", "label": "Planning", "description": "Task planning and tracking"},
    "delegation": {"icon": "\U0001f916", "label": "Delegation", "description": "Delegate to other agents"},
    "meta": {"icon": "\U0001f527", "label": "Meta", "description": "Create agents and harnesses"},
}

# ── Agent Icons ───────────────────────────────────────────────
AGENT_ICONS = ["\U0001f916", "\U0001f9e0", "\U0001f4bb", "\U0001f50d", "\U0001f6e1\ufe0f", "\U0001f4dd", "\U0001f9ea", "\U0001f3a8", "\u2699\ufe0f", "\U0001f52c", "\U0001f4ca", "\U0001f680", "\U0001f512", "\U0001f4da", "\U0001f41b", "\U0001f3d7\ufe0f"]

AGENT_COLORS = [
    "#6366f1", "#8b5cf6", "#a855f7", "#d946ef",
    "#ec4899", "#f43f5e", "#ef4444", "#f97316",
    "#eab308", "#84cc16", "#22c55e", "#14b8a6",
    "#06b6d4", "#0ea5e9", "#3b82f6", "#6366f1",
]
