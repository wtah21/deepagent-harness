"""
Main orchestrator agent builder.
Sets up the primary ReAct agent with all available tools, the main system prompt,
and the SQLite checkpointer for persistence across sessions.
"""
import os
import sqlite3
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

from config import (
    CHECKPOINT_DB_PATH,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE
)
from state import DeepAgentState
from prompts import MAIN_PROMPT

# Import all modules to populate the TOOL_REGISTRY
import tools
import subagents
import harness
from tool_registry import TOOL_REGISTRY

# Ensure checkpoint database parent directory exists
CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1. Setup the SQLite checkpointer
conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
# Enable WAL mode for concurrent access
conn.execute("PRAGMA journal_mode=WAL")
checkpointer = SqliteSaver(conn)

# 2. Build the main agent model
model_name = os.getenv("DEEPAGENT_MODEL", DEFAULT_MODEL)
temperature = float(os.getenv("DEEPAGENT_TEMPERATURE", DEFAULT_TEMPERATURE))
max_tokens = int(os.getenv("DEEPAGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS))
api_key = os.getenv("GROQ_API_KEY", "")

model = ChatGroq(
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    api_key=api_key
)

# 3. Resolve all tools registered for the main agent
main_tools = [meta.func for meta in TOOL_REGISTRY.values()]

# 4. Create the main ReAct agent
deepagent = create_react_agent(
    model,
    tools=main_tools,
    prompt=MAIN_PROMPT,
    state_schema=DeepAgentState,
    checkpointer=checkpointer,
)
