"""
内置「智能体配置设计师」工具（config_designer）。

这是一条固定流水线，直接绑定在主智能体上，不再作为独立子智能体：
  - 系统现状（工具 / 智能体 / harness 清单）在每次调用时注入提示词，无需 list_* 工具；
  - 模型通过 with_structured_output 输出 ConfigDraft 结构化配置草案；
  - 本模块负责校验、冲突检查与入库，并把结果序列化成结构化 JSON（DesignerResult）返回，
    主智能体可直接按字段提取任务结果。
"""
import json
import os
from typing import Literal, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from config import DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, debug_log
from prompts import CONFIG_DESIGNER_PROMPT
from meta_tools import (
    describe_agents, describe_harnesses, describe_tools,
    find_agent_conflict, find_harness_conflict,
    validate_agent_config, validate_harness_config,
)
from persistence import AgentStore, HarnessStore


# ── 结构化输出 Schema ────────────────────────────────────────────────

class AgentDraft(BaseModel):
    """自定义智能体的配置草案。"""
    name: str = Field(description="智能体名称：字母开头，仅字母/数字/下划线/连字符，不超过 40 字符")
    description: str = Field(description="一句话职责描述，供用户与流水线选型时参考")
    system_prompt: str = Field(description="该智能体的系统提示词，明确角色、行为规则与输出格式")
    tools: list[str] = Field(description="从「当前可用工具清单」中挑选的工具名，最小必要集合")
    icon: str = Field(default="🤖", description="有辨识度的 emoji 图标")
    color: str = Field(default="#3b82f6", description="十六进制主题色，如 #3b82f6")
    temperature: float = Field(default=0.1, description="采样温度，0 到 2 之间")
    max_tokens: int = Field(default=8000, description="单次回复的最大 token 数")


class HarnessStageDraft(BaseModel):
    """流水线单个阶段的配置草案。"""
    agent_name: str = Field(description=(
        "执行该阶段的智能体名，必须来自「已注册智能体」清单；"
        "禁止填 dispatcher / aggregator / moderator 等编排节点，这些节点由平台内部实现"))
    instructions: str = Field(description=(
        "该阶段在本流程中的具体职责、输入与期望产出；"
        "brainstorming 策略下还要写清该角色不可替代的立场与利益诉求（各角色应互相冲突）；"
        "review_loop 策略下最后一个阶段是评审者，必须要求它输出独立的「## 结论：APPROVED」"
        "或「## 结论：NEEDS_FIX」结论行"))
    description: str = Field(description=(
        "面向使用者的一句话节点职责说明（20-40 字），描述该节点在流程中做什么、"
        "向下一步交付什么，会展示在前端流程图的节点说明里；不要复述 instructions 全文"))
    pass_files: bool = Field(default=True, description="是否把虚拟文件系统传给该阶段")


class HarnessDraft(BaseModel):
    """多智能体流水线（harness）的配置草案。"""
    name: str = Field(description="流水线名称，不超过 60 字符")
    description: str = Field(description="工作流用途描述")
    purpose: str = Field(description=(
        "该流水线能处理的任务类型与适用边界（40-80 字），写清适合什么场景、不适合什么场景，"
        "会展示在前端流程图的「适用任务」一栏；不要与 description 重复"))
    strategy: Literal["sequential", "parallel", "review_loop", "brainstorming"] = Field(
        default="sequential",
        description=(
            "工作流编排策略选择："
            "sequential=顺序加工（适用于前后强依赖的标准化多步骤流水线，如先设计再实现、先研究再写作）；"
            "parallel=扇出汇总（适用于可拆分为多个独立维度并行的批量任务，如同时实现互不依赖的多个模块）；"
            "review_loop=创建加评审（适用于需要「编写-审查-修改」闭环的质量控制任务，如编码后由评审者挑错并循环修复）；"
            "brainstorming=头脑风暴（适用于需要多角色辩论、博弈与权衡取舍的复杂决策任务，"
            "如成本/质量/安全/工期多维权衡的方案选型或方案挑刺；"
            "stages 中每个角色必须立场互相冲突，max_review_rounds 表示最大发言次数，通常设 4-8）。"
        ),
    )
    stages: list[HarnessStageDraft] = Field(description=(
        "阶段序列，agent_name 必须已注册；只写真正执行任务的子智能体，"
        "不要写 dispatcher / aggregator / moderator 等编排节点（由平台内部自动生成）"))
    max_review_rounds: int = Field(
        default=1,
        description="review_loop 的最大评审轮数；brainstorming 下表示最大发言次数（4-8 为宜）",
    )
    icon: str = Field(default="🔗", description="emoji 图标")
    color: str = Field(default="#3b82f6", description="十六进制主题色")


class ConfigDraft(BaseModel):
    """配置设计师的模型输出：一份 agent/harness 配置，或一次澄清。"""
    kind: Literal["agent", "harness", "clarify"] = Field(
        description="agent=创建智能体配置；harness=创建流水线配置；clarify=信息不足需要澄清")
    message: str = Field(description="给用户的说明：设计意图与取舍，或澄清问题")
    agent: Optional[AgentDraft] = Field(default=None, description="kind=agent 时必填")
    harness: Optional[HarnessDraft] = Field(default=None, description="kind=harness 时必填")


class DesignerResult(BaseModel):
    """配置设计师返回给主智能体的结构化任务结果。"""
    status: Literal["saved", "clarify", "invalid", "error"] = Field(
        description="saved=已校验入库；clarify=需要补充信息；invalid=未通过校验；error=生成失败")
    kind: Literal["agent", "harness", "clarify", "unknown"] = Field(
        default="unknown", description="本次产出针对的配置类型")
    name: str = Field(default="", description="入库成功后生效的配置名")
    message: str = Field(description="设计说明或澄清问题，可直接转述给用户")
    config: Optional[dict] = Field(default=None, description="入库后的规范化配置")
    errors: list[str] = Field(default_factory=list, description="校验或冲突问题清单")


# DeepSeek（思考模式）不支持 response_format=json_schema，也不支持强制 tool_choice，
# 因此走 json_mode（response_format={"type": "json_object"}）+ 提示词内嵌 JSON Schema；
# 思考 token 消耗大，单独实例化并放宽输出上限，避免配置 JSON 被截断。
_model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL)
_temperature = float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE))
_max_tokens = int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS))
structured_model = ChatOpenAI(
    model=_model_name, temperature=_temperature, max_tokens=max(8192, _max_tokens),
    api_key=os.getenv("DEEPSEEK_API_KEY", ""), base_url=os.getenv("BASE_URL", ""),
).with_structured_output(ConfigDraft, method="json_mode")

_SCHEMA_DOC = json.dumps(ConfigDraft.model_json_schema(), ensure_ascii=False, indent=2)

# 首次生成 + 一次校验错误自动修正
_MAX_ATTEMPTS = 2


def _build_context() -> str:
    """把系统现状注入提示词，替代 list_tools/list_agents/list_harnesses 工具调用。"""
    return (
        "## 当前可用工具清单（agent.tools 只能从这里挑选）\n"
        + describe_tools()
        + "\n\n## 已注册智能体（harness.stages 的 agent_name 只能从这里挑选）\n"
        + describe_agents()
        + "\n\n## 已注册 harness 流水线\n"
        + describe_harnesses()
    )


def _design(task: str) -> DesignerResult:
    """固定流水线：生成配置草案 → 校验 → 入库，返回结构化结果。"""
    debug_log(f"[config-designer] enter | task={task[:60]}")
    system = f"{CONFIG_DESIGNER_PROMPT.replace('{schema}', _SCHEMA_DOC)}\n\n{_build_context()}"
    messages: list = [SystemMessage(content=system), HumanMessage(content=task)]

    for attempt in range(_MAX_ATTEMPTS):
        try:
            debug_log(f"[config-designer] invoking model (attempt {attempt + 1})")
            draft: ConfigDraft = structured_model.invoke(messages)
            debug_log(f"[config-designer] model ok | kind={draft.kind}")
        except Exception as e:
            debug_log(f"[config-designer] model failed: {type(e).__name__}: {e}")
            return DesignerResult(
                status="error",
                message=f"配置生成失败：{type(e).__name__}: {e}",
                errors=[f"{type(e).__name__}: {e}"],
            )

        if draft.kind == "clarify":
            return DesignerResult(
                status="clarify", kind="clarify",
                message=draft.message or "需要更多信息才能生成配置。",
            )

        # 模型偶尔会省略必填的嵌套对象（Schema 里非 required），反馈后重试一次
        missing = None
        if draft.kind == "agent" and draft.agent is None:
            missing = "agent"
        elif draft.kind == "harness" and draft.harness is None:
            missing = "harness"
        if missing:
            debug_log(f"[config-designer] missing object: kind={draft.kind} {missing}=None")
            if attempt + 1 < _MAX_ATTEMPTS:
                messages = messages + [
                    AIMessage(content=draft.message or "（提交了一份配置）"),
                    HumanMessage(
                        content=f"你输出了 kind={draft.kind}，但缺少必填的 \"{missing}\" 配置对象。"
                                f"请重新输出完整 JSON：\"{missing}\" 字段必须包含符合 Schema 的完整配置对象，"
                                "不得为 null 或省略。"),
                ]
                continue
            return DesignerResult(
                status="invalid", kind=draft.kind,
                message=draft.message or "配置生成不完整。",
                errors=[f"缺少必填的 \"{missing}\" 配置对象，不得为 null 或省略"],
            )

        # 草案 → dict → 统一校验（与侧边栏导入同一套防呆规则）
        if draft.kind == "agent":
            normalized, errors = validate_agent_config(draft.agent.model_dump())
            conflict = find_agent_conflict(normalized.get("name", "")) if not errors else None
        else:
            normalized, errors = validate_harness_config(draft.harness.model_dump())
            conflict = find_harness_conflict(normalized.get("name", "")) if not errors else None

        if not errors and not conflict:
            if draft.kind == "agent":
                AgentStore.save(normalized)
            else:
                HarnessStore.save(normalized)
            # 惰性导入，避免 subagents ↔ config_designer 的循环依赖
            from subagents import AgentFactory
            AgentFactory.clear_cache()
            debug_log(f"[config-designer] saved kind={draft.kind} name={normalized.get('name')}")
            return DesignerResult(
                status="saved", kind=draft.kind, name=normalized.get("name", ""),
                message=draft.message, config=normalized,
            )

        # 校验/冲突失败：把错误清单反馈给模型自动修正一次
        problems = list(errors) + ([conflict] if conflict else [])
        if attempt + 1 < _MAX_ATTEMPTS:
            messages = messages + [
                AIMessage(content=draft.message or "（提交了一份配置）"),
                HumanMessage(content="以上配置未通过校验，问题清单：\n"
                             + "\n".join(f"- {p}" for p in problems)
                             + "\n请严格对照问题逐条修正后，重新输出完整配置。"),
            ]
            continue
        return DesignerResult(
            status="invalid", kind=draft.kind, name=normalized.get("name", ""),
            message=draft.message, errors=problems,
        )

    return DesignerResult(status="error", message="配置生成未成功，请重试。")


@tool
def config_designer(instructions: str) -> str:
    """把需求转化为可运行的 agent / harness 配置，并完成校验与入库。

    Args:
        instructions: 完整的配置需求，必须自包含——说明要创建的配置类型（agent 或 harness）、
            目标用途、需要的能力与约束；缺少关键信息时列清楚，配置设计师会返回澄清问题。

    Returns:
        JSON 字符串。字段：status（saved=已校验入库 / clarify=需要澄清 / invalid=校验未通过 /
        error=生成失败）、kind（agent 或 harness 或 clarify）、name（入库的配置名）、
        message（设计说明或澄清问题）、config（入库后的规范化配置，失败时为 null）、
        errors（校验或冲突问题清单）。
    """
    return json.dumps(_design(instructions).model_dump(), ensure_ascii=False, indent=2)
