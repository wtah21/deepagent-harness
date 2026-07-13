"""
Harness Engine for DeepAgent.
Compiles and executes multi-agent workflows (pipelines, parallel fan-out, review loops)
by dynamically creating LangGraph StateGraphs.
"""
import datetime
from typing import Annotated, Any, Dict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from state import DeepAgentState, Todo
from subagents import AgentFactory
from persistence import HarnessStore

# ── Builtin Harness Templates ────────────────────────────────────────

BUILTIN_HARNESSES = [
    {
        "name": "Code & Review",
        "description": "Sequential loop where a Coder writes code and a Reviewer reviews it, looping for fixes.",
        "strategy": "review_loop",
        "stages": [
            {"agent_name": "coder", "instructions": "Write a complete python implementation for the user request.", "pass_files": True},
            {"agent_name": "reviewer", "instructions": "Review the code in the virtual filesystem for errors, style, and correctness. If changes are needed, list them. If approved, write 'APPROVED'.", "pass_files": True}
        ],
        "max_review_rounds": 3,
        "icon": "🔄",
        "color": "#ef4444"
    },
    {
        "name": "Design & Build",
        "description": "An Architect designs the layout and state schema, then a Coder implements it.",
        "strategy": "sequential",
        "stages": [
            {"agent_name": "langgraph-architect", "instructions": "Design the graph architecture and state schema. Write the design to design.py", "pass_files": True},
            {"agent_name": "coder", "instructions": "Read design.py and implement the full solution as designed.", "pass_files": True}
        ],
        "max_review_rounds": 1,
        "icon": "🏗️",
        "color": "#8b5cf6"
    },
    {
        "name": "Research & Write",
        "description": "Researcher gathers technical documentation, and Coder implements code based on research.",
        "strategy": "sequential",
        "stages": [
            {"agent_name": "researcher", "instructions": "Search the web and gather technical references for the task. Write research_notes.md.", "pass_files": True},
            {"agent_name": "coder", "instructions": "Read research_notes.md and implement the requested application code.", "pass_files": True}
        ],
        "max_review_rounds": 1,
        "icon": "📚",
        "color": "#14b8a6"
    }
]


class HarnessEngine:
    """Orchestrates dynamic compilation and running of multi-agent harnesses."""

    @staticmethod
    def compile_graph(config: dict) -> Any:
        """Compile a dynamic LangGraph StateGraph from a harness configuration."""
        builder = StateGraph(DeepAgentState)
        strategy = config.get("strategy", "sequential")
        stages = config.get("stages", [])

        if not stages:
            raise ValueError("Harness must have at least one stage configured.")

        if strategy == "sequential":
            prev_node = START
            for idx, stage in enumerate(stages):
                node_name = f"stage_{idx}_{stage['agent_name']}"

                def make_node(stg, name):
                    def node_fn(state: DeepAgentState) -> dict:
                        agent = AgentFactory.get_agent(stg["agent_name"])
                        prompt = stg["instructions"]

                        # Determine state files passing
                        passed_files = state.get("files", {}) if stg.get("pass_files", True) else {}

                        # Call the subagent
                        result = agent.invoke({
                            "messages": [HumanMessage(content=prompt)],
                            "files": passed_files,
                            "todos": state.get("todos", []),
                            "execution_log": state.get("execution_log", []),
                        })

                        # Log execution
                        log_entry = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "agent_name": stg["agent_name"],
                            "action": f"harness_{name}",
                            "summary": f"Executed instruction: {prompt[:100]}...",
                            "status": "success"
                        }
                        logs = list(state.get("execution_log", []))
                        logs.append(log_entry)

                        # Output updates
                        return {
                            "files": result.get("files", state.get("files", {})),
                            "todos": result.get("todos", state.get("todos", [])),
                            "execution_log": logs,
                            "messages": [AIMessage(content=f"[{stg['agent_name']}] {result['messages'][-1].content}")]
                        }
                    return node_fn

                builder.add_node(node_name, make_node(stage, node_name))
                builder.add_edge(prev_node, node_name)
                prev_node = node_name

            builder.add_edge(prev_node, END)

        elif strategy == "parallel":
            # We run stages sequentially in code execution for simplification in free tier,
            # but model it as independent nodes branching from START and merging.
            merge_node_name = "parallel_merge"
            
            # Create merge node
            def merge_fn(state: DeepAgentState) -> dict:
                log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "agent_name": "harness_engine",
                    "action": "parallel_merge",
                    "summary": "Merged parallel branches successfully.",
                    "status": "success"
                }
                logs = list(state.get("execution_log", []))
                logs.append(log_entry)
                return {"execution_log": logs}

            builder.add_node(merge_node_name, merge_fn)

            for idx, stage in enumerate(stages):
                node_name = f"stage_{idx}_{stage['agent_name']}"

                def make_parallel_node(stg, name):
                    def node_fn(state: DeepAgentState) -> dict:
                        agent = AgentFactory.get_agent(stg["agent_name"])
                        result = agent.invoke({
                            "messages": [HumanMessage(content=stg["instructions"])],
                            "files": state.get("files", {}) if stg.get("pass_files", True) else {},
                            "todos": state.get("todos", []),
                            "execution_log": state.get("execution_log", []),
                        })
                        return {
                            "files": result.get("files", {}),
                            "messages": [AIMessage(content=f"[{stg['agent_name']}] {result['messages'][-1].content}")]
                        }
                    return node_fn

                builder.add_node(node_name, make_parallel_node(stage, node_name))
                builder.add_edge(START, node_name)
                builder.add_edge(node_name, merge_node_name)

            builder.add_edge(merge_node_name, END)

        elif strategy == "review_loop":
            if len(stages) < 2:
                raise ValueError("Review loop strategy requires at least 2 stages: Creator & Reviewer.")

            creator = stages[0]
            reviewer = stages[1]
            max_rounds = config.get("max_review_rounds", 3)

            # Node 1: Creator
            def creator_fn(state: DeepAgentState) -> dict:
                feedback = ""
                # Find last reviewer message in history
                for msg in reversed(state.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.content.startswith(f"[{reviewer['agent_name']}]"):
                        feedback = msg.content
                        break

                prompt = creator["instructions"]
                if feedback:
                    prompt += f"\n\nPrevious Reviewer Feedback:\n{feedback}\nPlease fix the issues."

                agent = AgentFactory.get_agent(creator["agent_name"])
                result = agent.invoke({
                    "messages": [HumanMessage(content=prompt)],
                    "files": state.get("files", {}),
                    "todos": state.get("todos", []),
                    "execution_log": state.get("execution_log", []),
                })

                return {
                    "files": result.get("files", state.get("files", {})),
                    "todos": result.get("todos", state.get("todos", [])),
                    "messages": [AIMessage(content=f"[{creator['agent_name']}] {result['messages'][-1].content}")]
                }

            # Node 2: Reviewer
            def reviewer_fn(state: DeepAgentState) -> dict:
                agent = AgentFactory.get_agent(reviewer["agent_name"])
                result = agent.invoke({
                    "messages": [HumanMessage(content=reviewer["instructions"])],
                    "files": state.get("files", {}),
                    "todos": state.get("todos", []),
                    "execution_log": state.get("execution_log", []),
                })
                return {
                    "messages": [AIMessage(content=f"[{reviewer['agent_name']}] {result['messages'][-1].content}")]
                }

            builder.add_node("creator", creator_fn)
            builder.add_node("reviewer", reviewer_fn)

            builder.add_edge(START, "creator")
            builder.add_edge("creator", "reviewer")

            # Routing decision
            def check_loop(state: DeepAgentState) -> Literal["creator", END]:
                msgs = state.get("messages", [])
                review_count = sum(1 for m in msgs if isinstance(m, AIMessage) and m.content.startswith(f"[{reviewer['agent_name']}]"))

                if review_count >= max_rounds:
                    return END

                last_msg = msgs[-1].content if msgs else ""
                # Look for approval keywords
                if any(x in last_msg.lower() for x in ["approved", "looks good", "no issues", "no changes needed"]):
                    return END

                return "creator"

            builder.add_conditional_edges("reviewer", check_loop, {"creator": "creator", END: END})

        return builder.compile()


@tool
def run_harness(
    harness_name: str,
    initial_instructions: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Execute a multi-agent workflow harness pipeline by name.
    harness_name: name of the registered harness pipeline.
    initial_instructions: instructions or task prompt to start the harness.
    """
    config = HarnessStore.get(harness_name)
    if not config:
        # Check builtins
        for h in BUILTIN_HARNESSES:
            if h["name"] == harness_name:
                config = h
                break

    if not config:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error: Harness '{harness_name}' not found.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Compile and execute the harness graph
    try:
        compiled_graph = HarnessEngine.compile_graph(config)
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error compiling harness graph: {str(e)}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    # Initial state payload
    init_state = {
        "messages": [HumanMessage(content=initial_instructions)],
        "files": state.get("files", {}),
        "todos": state.get("todos", []),
        "execution_log": state.get("execution_log", []),
    }

    # Run the graph
    result = compiled_graph.invoke(init_state)

    # Log execution
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "agent_name": "harness_engine",
        "action": "run_harness",
        "summary": f"Completed harness '{harness_name}' run.",
        "status": "success"
    }
    logs = list(result.get("execution_log", []))
    logs.append(log_entry)

    # Find the last message
    final_reply = result["messages"][-1].content if result.get("messages") else "Harness run complete."

    return Command(
        update={
            "files": result.get("files", state.get("files", {})),
            "todos": result.get("todos", state.get("todos", [])),
            "execution_log": logs,
            "messages": [ToolMessage(f"[Harness: {harness_name}] {final_reply}", tool_call_id=tool_call_id)],
        }
    )


# ── Initialization ────────────────────────────────────────────────────

def initialize_builtin_harnesses():
    """Populate database with built-in harness configs if empty."""
    for h in BUILTIN_HARNESSES:
        if not HarnessStore.get(h["name"]):
            HarnessStore.save(h)


# Register run_harness in the global registry
try:
    from tool_registry import register_tool
    register_tool(run_harness, "run_harness", "Execute a multi-agent workflow harness pipeline", "meta")
except Exception as e:
    print(f"Warning registering run_harness tool: {e}")

# Run database populator
initialize_builtin_harnesses()
