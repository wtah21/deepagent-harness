"""
Dynamic Agent Factory and Delegation tool for DeepAgent Harness.
Compiles LangGraph React agents on the fly based on configurations
from the database or builtin templates.
"""
import os
import datetime
from typing import Annotated, Any, Dict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command

from state import DeepAgentState
from persistence import AgentStore
from prompts import BUILTIN_TEMPLATES
from tool_registry import get_tools_by_names

load_dotenv()


class AgentFactory:
    """Caching factory for LangGraph agents compiled from configurations."""
    _cache: Dict[str, Any] = {}

    @classmethod
    def get_agent(cls, agent_name: str) -> Any:
        """Get or compile the react agent by name.
        Checks cache first, then database configs, then builtins.
        """
        # Check cache
        if agent_name in cls._cache:
            return cls._cache[agent_name]

        # Retrieve config
        config = AgentStore.get(agent_name)
        if not config:
            config = BUILTIN_TEMPLATES.get(agent_name)

        if not config:
            raise ValueError(f"Agent '{agent_name}' is not registered in the database or built-ins.")

        # Resolve tool functions
        tools_list = get_tools_by_names(config.get("tools", []))

        # Build LLM
        model_name = config.get("model") or os.getenv("DEEPAGENT_MODEL", "llama-3.3-70b-versatile")
        temperature = config.get("temperature", 0.1)
        max_tokens = config.get("max_tokens", 8000)
        api_key = os.getenv("GROQ_API_KEY", "")

        llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )

        # Create Agent
        agent = create_react_agent(
            llm,
            tools=tools_list,
            prompt=config.get("system_prompt", ""),
            state_schema=DeepAgentState
        )

        cls._cache[agent_name] = agent
        return agent

    @classmethod
    def clear_cache(cls):
        """Reset the cached agents to pick up new configurations."""
        cls._cache.clear()


@tool
def task(
    subagent: str,
    instructions: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Delegate a self-contained unit of work to a specialist sub-agent.
    subagent: name of the custom or built-in agent (e.g. coder, reviewer, debugger, etc.).
    instructions: a complete, standalone task description — the sub-agent has no chat history.
    """
    try:
        agent = AgentFactory.get_agent(subagent)
    except ValueError as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error resolving agent: {str(e)}", 
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Invoke subagent sharing the same virtual files, todos, and execution logs
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=instructions)],
            "files": state.get("files", {}),
            "todos": state.get("todos", []),
            "execution_log": state.get("execution_log", []),
        }
    )

    final_reply = result["messages"][-1].content
    
    # Audit log entry
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "agent_name": subagent,
        "action": "task",
        "summary": f"Executed task: {instructions[:100]}...",
        "status": "success"
    }
    
    current_logs = list(state.get("execution_log", []))
    current_logs.append(log_entry)

    # Return updated files, todos, execution logs, and messages
    return Command(
        update={
            "files": result.get("files", state.get("files", {})),
            "todos": result.get("todos", state.get("todos", [])),
            "execution_log": current_logs,
            "messages": [ToolMessage(f"[{subagent}] {final_reply}", tool_call_id=tool_call_id)],
        }
    )


# ── Initialization ────────────────────────────────────────────────────

def initialize_builtin_agents():
    """Populate SQLite database with built-in agent templates if empty."""
    for name, config in BUILTIN_TEMPLATES.items():
        if not AgentStore.exists(name):
            AgentStore.save(config)


# Register this tool in the central registry
try:
    from tool_registry import register_tool
    register_tool(task, "task", "Delegate a self-contained unit of work to a specialist sub-agent", "delegation")
except Exception as e:
    print(f"Warning registering task tool: {e}")

# Run database populator
initialize_builtin_agents()
