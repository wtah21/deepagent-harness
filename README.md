# 🧠 DeepAgent Harness — Enterprise Multi-Agent Coding Platform

An enterprise-grade, state-of-the-loop multi-agent coding platform built completely on **LangChain** and **LangGraph** (without relying on any restrictive pre-packaged agent frameworks). DeepAgent Harness allows you to build custom AI agents, orchestrate complex pipelines (Sequential, Parallel, and Review loops), manage sandbox file environments, and test code execution—all powered by Groq's fast inference tier models.

## 🚀 Key Features

*   🏭 **Dynamic Agent Studio**: Create custom sub-agents via UI. Assign distinct system instructions, choose custom toolkits, configure model parameters, and customize their theme color and icons.
*   🔗 **Harness Pipeline Designer**: Connect multiple custom agents into complex workflow pipelines. Design Sequential steps, parallel fan-outs, and self-correcting Review-Loop pipelines (e.g. Coder -> Reviewer loop) with live visual topology previews via Mermaid rendering.
*   📁 **Isolated Project namespaces**: Switch between projects instantly. Each project maintains its own isolated virtual filesystem sandbox, chat logs, and execution traces.
*   💾 **Durable SQLite Checkpointer**: Fully persistent graph states. App restarts, workspace changes, and session swaps restore flawlessly by leveraging SqliteSaver.
*   🛠️ **Expanded 12+ Tool Suite**: Web searching (DuckDuckGo), URL text scraper, Python execution sandbox, terminal commands shell, package installations, virtual/real filesystem imports/exports, and diff comparisons.

---

## 🛠️ System Architecture

```
                 Streamlit Frontend (Premium dark glassmorphism layout)
              ┌──────────────────────────┬───────────────────────────┐
              │                          │                           │
         Agent Studio            Harness Designer             Chat Workspace
       (CRUD custom agent)     (Stage connections/Mermaid)  (Direct chat/work pipeline)
              └──────────────────────────┼───────────────────────────┘
                                         │
                        Dynamic Graph Compile Engine
              ┌──────────────────────────┴───────────────────────────┐
              │                                                      │
         AgentFactory                                          HarnessEngine
    (Compiles custom LLMs)                             (Generates dynamic StateGraphs)
              └──────────────────────────┬───────────────────────────┘
                                         │
                           LangGraph Checkpointer (SQLite)
```

---

## 💻 Setup & Installation

Ensure you have [uv](https://github.com/astral-sh/uv) or standard Python installed.

### 1. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 2. Configure Environment Keys

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_...
DEEPAGENT_MODEL=llama-3.3-70b-versatile
```

### 3. Launch App

```bash
uv run streamlit run streamlit_app.py
```

---

## 📄 File Architecture

| File | Type | Purpose |
|---|---|---|
| `config.py` | Config | Global configuration constants, rate limits, theme color lists, paths. |
| `state.py` | Schema | DeepAgentState TypedDict definition (messages, virtual files, execution logs). |
| `persistence.py` | Storage | SQLite CRUD databases for agents, harnesses, and projects. |
| `tool_registry.py` | Registry | Global lookup mapping string names to callable Python tool functions. |
| `tools.py` | Modules | Real and virtual filesystem operations, Python runtimes, web tools. |
| `prompts.py` | Templates | Main system prompt and 10+ default agent configuration templates. |
| `subagents.py` | Factory | Caching Factory compilation of custom LLM ReAct loops. |
| `harness.py` | Engine | Flowchart compilation algorithms for multi-agent workflows. |
| `graph.py` | Orchestration | Top-level Main ReAct Agent instantiation and SQLite checkpointer hook. |
| `styles.py` | Style | Customized CSS injections for dark styling and glassmorphism. |
| `components.py` | Component | HTML components, timeline views, and Mermaid javascript configurations. |
| `streamlit_app.py` | Workspace | Home chatbot panel and multi-page routing. |
| `pages/1_🤖_Agent_Studio.py` | Page | UI for dynamic agent creation and editing. |
| `pages/2_🔗_Harness_Designer.py` | Page | Workflow pipeline stage builder and flowchart viewer. |
| `pages/3_📁_Projects.py` | Page | Project directory namespace supervisor. |
| `pages/4_💬_Direct_Chat.py` | Page | Clean, minimalist normal chatbot interface. |

---

## 🛡️ Sandbox Code Safety Notice

Tools such as `execute_python`, `shell_command`, and `save_to_disk` execute code directly in your local environment. Do not expose this Streamlit interface publicly without introducing standard Docker isolation, firewalls, and restricted authorization mechanisms.
