# 🧠 DeepAgent Harness

基于 **LangChain / LangGraph** 从零构建的多智能体编码平台：不依赖任何封装好的 Agent 框架，主智能体作为"任务编排者"负责拆解、分发与整合，子智能体与执行流水线（Harness Pipeline）均可通过对话动态创建，模型生成的代码在 Docker 容器沙盒中隔离执行。

## ✨ 核心特性

### 🏭 编排引擎

- **动态流水线编译**：运行时把子智能体编排成 LangGraph StateGraph，支持四种拓扑——顺序流水线、并行扇出 + 结果汇聚、评审自纠错循环（Coder → Reviewer 循环修正）、群组头脑风暴；
- **任务编排者**：主智能体自动拆解任务（write\_todos）、按职能描述选型调用子智能体，同一子任务重试 2 次失败后用通用工具兜底完成；
- **对话式配置**：内置 config\_designer 工具，对话中即可创建 / 修改子智能体（指令、工具集、模型参数、主题色）与完整流水线，配置自动入库。

### 🛡️ 上下文工程

- **混合分层压缩**：以 `pre_model_hook` 在每次调用模型前检测上下文占用，超窗口 70% 触发——核心需求永不压缩、近 5 轮保留原文、更早轮次 LLM 生成结构化摘要、增量压缩避免重复总结；主智能体 / 子智能体分别采用分层 / 纯摘要两种策略；
- **工具输出治理**：超长工具结果进入上下文前先压缩，由决策模型路由为「结构化截断」（保留元数据与前几条记录）或「程序化过滤」（生成过滤代码在沙箱中执行筛选），置信度不足时本地启发式兜底。

### 🔒 安全执行

- **Docker 沙盒**：每个项目一个常驻容器（`--cpus=1.0 -m 1g`），卷挂载工作区目录，`docker exec` 执行模型生成的代码，容器内写文件即宿主机产物；Docker 不可用时自动降级本地沙盒。

### 🔌 生态协议

- **MCP 客户端**：支持 stdio / SSE / streamable\_http 三种传输，后台事件循环常驻会话，把异步 MCP 调用桥接成同步 LangGraph 工具并自动注册进工具注册表；
- **A2A 客户端**：同时兼容 v0.3.0 / v1.0 双版本协议差异，通过 `/.well-known/agent-card.json` 自动发现远端智能体，封装为本地工具供主 / 子智能体调用。

### 💾 持久化与人机协同

- **三层持久化**：SQLite 业务库（智能体 / 流水线 / 项目 / 文件元数据）+ SqliteSaver 图检查点（跨会话完整恢复对话与图状态）+ 消息归档库；
- **人机协同（HITL）**：基于 LangGraph `interrupt / resume`，主智能体缺关键信息时暂停追问（自由输入或选项按钮），作答后从原位置继续执行；
- **虚拟文件系统**：项目隔离的沙盒文件树，流水线执行中 agent 新写入的文件实时可见（Streamlit fragment 自刷新），支持在线预览、删除与一键导出。

## 🏗️ 系统架构

```
                      Streamlit 工作台（streamlit_app.py）
        ┌──────────────┬──────────────────┬───────────────────┐
        │  对话 / 流水线执行  │  虚拟文件系统（实时）  │  MCP / A2A 管理面板  │
        └──────────────┴────────┬─────────┴───────────────────┘
                                │
                 主智能体 = 任务编排者（graph.py，ReAct）
        ┌───────────┬──────────┼──────────────┬──────────────┐
   config_designer  ask_user  run_harness   子智能体工具    MCP / A2A 工具
   (生成配置入库)    (HITL)   (执行流水线)   (按需分发)     (外部能力接入)
                                │
                 Harness 引擎（harness.py，动态 StateGraph）
        ┌──────────┬──────────┬──────────────┬──────────────┐
     顺序流水线   并行扇出+汇聚   评审自纠错循环    群组头脑风暴
                                │
        ┌───────────────────────┴───────────────────────┐
        │  SqliteSaver 检查点  │  SQLite 业务库  │  Docker 沙盒  │
        └───────────────────────────────────────────────┘
```

## 📂 模块说明

| 模块                     | 职责                                                                   |
| ---------------------- | -------------------------------------------------------------------- |
| `graph.py`             | 主智能体装配：任务编排者 + 工具集组装 + SqliteSaver 检查点                               |
| `harness.py`           | Harness 引擎：动态编译多智能体 StateGraph，进度事件流与后台运行注册表                         |
| `subagents.py`         | AgentFactory：编译自定义子智能体（ReAct 循环），按签名缓存                               |
| `context_manager.py`   | 混合分层上下文压缩（hybrid / summary 两种策略，增量压缩）                                |
| `tool_output_guard.py` | 超长工具输出治理：结构化截断 / 程序化过滤，决策模型路由                                        |
| `mcp_client.py`        | MCP 客户端：三种传输、异步桥接同步、工具自动注册                                           |
| `a2a_client.py`        | A2A 客户端：双版本协议兼容、Agent Card 发现、远端智能体工具化                               |
| `docker_runtime.py`    | Docker 隔离执行运行时：常驻容器 + 卷挂载 + docker exec                              |
| `hitl.py`              | 人机协同：interrupt / resume 追问（input / choice 两种形态）                      |
| `persistence.py`       | SQLite 持久化：ProjectStore / WorkspaceStore / AgentStore / HarnessStore |
| `tools.py`             | 内置工具：虚拟 / 真实文件系统、Python 执行、终端命令、Web 搜索与抓取                            |
| `state.py`             | DeepAgentState：消息、虚拟文件、执行日志等图状态定义                                    |
| `streamlit_app.py`     | 工作台前端：对话、流水线执行视图、虚拟文件系统、MCP/A2A 管理                                   |

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows（Linux/macOS: source .venv/bin/activate）
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env`：

```env
# 必需：OpenAI 兼容接口的模型服务
DEEPSEEK_API_KEY=sk-...
BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-v4-flash

# 可选：工具输出治理的决策模型（不配则本地启发式兜底）
# JEV_API_KEY=...
# JEV_BASE_URL=https://api.typesafe.ai/v1
# JEV_MODEL=jev-latest
```

> 任何 OpenAI 兼容接口均可接入（通过 `BASE_URL` 切换），模型名称用 `MODEL_NAME` 指定。

### 3. 启动应用

```bash
streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

### 4.（可选）启用 Docker 沙盒

安装并启动 Docker 后应用自动检测并启用容器隔离执行；未安装 Docker 时自动降级为本地沙盒执行。

```bash
docker pull python:3.11-slim
```

## 💡 使用指南

1. **创建子智能体**：直接在对话框描述需求，例如"帮我创建一个负责代码评审的子智能体"，主智能体会调用 config\_designer 生成配置入库；
2. **编排流水线**：例如"把这些智能体串成一条流水线：架构师 → 编码 → 评审，评审不通过就打回重写"，引擎自动编译为评审循环拓扑；
3. **执行任务**：既可以让主智能体自动分发，也可以在流水线模式中手动试运行某条 Harness；执行过程中侧边栏虚拟文件系统实时显示各阶段产出；
4. **接入外部能力**：侧边栏配置 MCP 服务器（stdio / SSE / HTTP）或 A2A 远端智能体，工具与远端 Agent 自动注册进主 / 子智能体的工具清单；
5. **项目隔离**：每个项目拥有独立的虚拟文件系统、对话历史与沙盒容器，可随时切换、重命名或删除。

## ⚙️ 配置参考

| 环境变量                       | 默认值                 | 说明                    |
| -------------------------- | ------------------- | --------------------- |
| `DEEPSEEK_API_KEY`         | —                   | 模型服务 API Key（必需）      |
| `BASE_URL`                 | —                   | OpenAI 兼容接口地址         |
| `MODEL_NAME`               | `deepseek-v4-flash` | 默认模型                  |
| `MAX_TOKENS`               | `1024`              | 单次生成上限                |
| `TEMPERATURE`              | `0.7`               | 采样温度                  |
| `CONTEXT_WINDOW_TOKENS`    | 自动探测                | 显式指定模型上下文窗口           |
| `CONTEXT_TRIGGER_RATIO`    | `0.7`               | 上下文压缩触发比例             |
| `SUBAGENT_RECURSION_LIMIT` | `600`               | 子智能体递归上限（约 300 次工具调用） |
| `DOCKER_IMAGE`             | `python:3.11-slim`  | 沙盒容器镜像                |

## 🛡️ 安全须知

模型生成的代码默认在 Docker 容器中隔离执行（CPU / 内存受限，文件系统随项目目录挂载）。若未安装 Docker，代码将在**本地环境直接执行**，请勿在不受信任的网络环境中公开本应用，并为模型服务配置合理的输出内容边界。

## 📄 License

详见 [LICENSE](LICENSE)。
