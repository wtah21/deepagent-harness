"""
元创建（meta-creation）支撑模块 —— 本项目的核心特色。

生成 agent / harness 配置的能力由主智能体上的 config_designer 工具提供
（见 config_designer.py）：它通过结构化输出产出配置草案，并调用本模块完成校验与入库。
本模块提供两条入口共用的防呆规则：
  - 统一校验：validate_agent_config / validate_harness_config 同时服务于
    config_designer 入库与侧边栏导入，保证两条入口的规则完全一致。
  - 内置保护：与内置模板重名的配置一律拒绝入库，避免覆盖系统基石角色。
  - 现状描述：describe_tools / describe_agents / describe_harnesses 把系统
    现状注入 config_designer 的提示词，替代原来的 list_* 工具调用。
"""
import re

# 显式导入 tools，确保基础工具先于本模块完成注册（校验 tools 名称时依赖完整注册表）
import tools  # noqa: F401
from config import AGENT_ICONS, AGENT_COLORS
from persistence import AgentStore, HarnessStore
from prompts import BUILTIN_TEMPLATES
from tool_registry import TOOL_REGISTRY
from harness import BUILTIN_HARNESSES

# 智能体名称会被 task 工具与文件路径引用，因此限制为安全的标识符
_AGENT_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,39}$")
_VALID_STRATEGIES = {"sequential", "parallel", "review_loop", "brainstorming"}


# ── 冲突检查（结构化输出入库与侧边栏导入共用）────────────────────────

def find_agent_conflict(name: str) -> str | None:
    """检查智能体名称冲突，返回冲突说明；无冲突返回 None。"""
    if name in BUILTIN_TEMPLATES:
        return f"'{name}' 是内置智能体模板，不允许覆盖"
    existing = AgentStore.get(name)
    if existing is not None:
        if existing.get("is_builtin"):
            return f"'{name}' 是内置智能体模板，不允许覆盖"
        return f"已存在同名自定义智能体 '{name}'（可先在侧边栏「配置管理」中删除）"
    return None


def find_harness_conflict(name: str) -> str | None:
    """检查 harness 名称冲突，返回冲突说明；无冲突返回 None。"""
    if any(h["name"] == name for h in BUILTIN_HARNESSES):
        return f"'{name}' 是内置 harness 模板，不允许覆盖"
    if HarnessStore.get(name) is not None:
        return f"已存在同名 harness '{name}'（可先在侧边栏「配置管理」中删除）"
    return None


def find_harnesses_referencing(agent_name: str) -> list[str]:
    """找出 stages 中引用了某智能体的全部 harness，删除前用于引用完整性检查。"""
    refs = []
    for h in HarnessStore.list_all():
        if any(s.get("agent_name") == agent_name for s in h.get("stages", [])):
            refs.append(h["name"])
    return refs


# ── 校验器（结构化输出入库与侧边栏导入共用）──────────────────────────

def validate_agent_config(config: dict) -> tuple[dict, list[str]]:
    """校验并规范化一份智能体配置，返回 (规范化配置, 错误列表)。

    校验通过时返回的配置可直接写入 AgentStore；失败时第一个元素为空 dict。
    """
    if not isinstance(config, dict):
        return {}, ["配置必须是一个 JSON 对象"]

    errors: list[str] = []

    name = str(config.get("name", "")).strip()
    if not name:
        errors.append("缺少 name（智能体名称）")
    elif not _AGENT_NAME_RE.match(name):
        errors.append("name 只能包含字母、数字、下划线与连字符，且以字母开头（长度 ≤ 40）")

    description = str(config.get("description", "")).strip()
    if not description:
        errors.append("缺少 description（职责描述，供用户与主智能体选择时参考）")

    system_prompt = str(config.get("system_prompt", "")).strip()
    if not system_prompt:
        errors.append("缺少 system_prompt（系统提示词，决定该智能体的行为）")

    tools_names = config.get("tools", [])
    if not isinstance(tools_names, list) or not tools_names:
        errors.append("tools 必须是非空的工具名列表")
    else:
        unknown = [t for t in tools_names if not isinstance(t, str) or t not in TOOL_REGISTRY]
        if unknown:
            errors.append(f"tools 中包含未注册的工具名: {unknown}，请从可用工具清单中挑选")

    temperature = config.get("temperature", 0.1)
    try:
        temperature = float(temperature)
        if not 0 <= temperature <= 2:
            errors.append("temperature 必须在 0 到 2 之间")
    except (TypeError, ValueError):
        errors.append("temperature 必须是数字")

    max_tokens = config.get("max_tokens", 8000)
    try:
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            errors.append("max_tokens 必须是正整数")
    except (TypeError, ValueError):
        errors.append("max_tokens 必须是正整数")

    if errors:
        return {}, errors

    normalized = {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        # 去重但保持顺序
        "tools": list(dict.fromkeys(tools_names)),
        "icon": str(config.get("icon") or AGENT_ICONS[0]),
        "color": str(config.get("color") or AGENT_COLORS[0]),
        "model": str(config.get("model") or ""),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "is_builtin": False,
    }
    return normalized, []


def validate_harness_config(config: dict) -> tuple[dict, list[str]]:
    """校验并规范化一份 harness 配置，返回 (规范化配置, 错误列表)。"""
    if not isinstance(config, dict):
        return {}, ["配置必须是一个 JSON 对象"]

    errors: list[str] = []

    name = str(config.get("name", "")).strip()
    if not name:
        errors.append("缺少 name（harness 名称）")
    elif len(name) > 60:
        errors.append("name 过长（不超过 60 字符）")

    description = str(config.get("description", "")).strip()
    if not description:
        errors.append("缺少 description（工作流用途描述）")

    strategy = str(config.get("strategy") or "sequential").strip()
    if strategy not in _VALID_STRATEGIES:
        errors.append(f"strategy 必须是 {'/'.join(sorted(_VALID_STRATEGIES))} 之一")

    stages_raw = config.get("stages", [])
    stages: list[dict] = []
    if not isinstance(stages_raw, list) or not stages_raw:
        errors.append("stages 必须是非空列表，每个元素形如 "
                      '{"agent_name": "...", "instructions": "...", "pass_files": true}')
    else:
        for i, raw in enumerate(stages_raw):
            if not isinstance(raw, dict):
                errors.append(f"stages[{i}] 必须是对象")
                continue
            agent_name = str(raw.get("agent_name", "")).strip()
            instructions = str(raw.get("instructions", "")).strip()
            if not agent_name:
                errors.append(f"stages[{i}] 缺少 agent_name")
            elif not (AgentStore.get(agent_name) is not None or agent_name in BUILTIN_TEMPLATES):
                errors.append(f"stages[{i}] 引用的智能体 '{agent_name}' 尚不存在，"
                              "请先创建对应智能体，或改用已有智能体")
            if not instructions:
                errors.append(f"stages[{i}] 缺少 instructions（该阶段的具体职责）")
            stages.append({
                "agent_name": agent_name,
                "instructions": instructions,
                "pass_files": bool(raw.get("pass_files", True)),
                # 节点职责：给 UI 的流程图做说明，缺省时留空串
                "description": str(raw.get("description", "")).strip(),
            })

    max_rounds = config.get("max_review_rounds", 1)
    try:
        max_rounds = int(max_rounds)
        if max_rounds < 1:
            errors.append("max_review_rounds 必须 ≥ 1")
    except (TypeError, ValueError):
        errors.append("max_review_rounds 必须是整数")

    if strategy == "review_loop" and len(stages_raw) < 2:
        errors.append("review_loop 策略至少需要 2 个阶段（创建者 + 评审者）")
    if strategy == "brainstorming" and len(stages_raw) < 2:
        errors.append("brainstorming 策略至少需要 2 个参会角色（立场互相冲突的智能体）")

    if errors:
        return {}, errors

    normalized = {
        "name": name,
        "description": description,
        # 适用任务：告诉使用者这条流水线能接什么活；老配置没写时退化为 description
        "purpose": str(config.get("purpose") or description).strip(),
        "strategy": strategy,
        "stages": stages,
        "max_review_rounds": max_rounds,
        "icon": str(config.get("icon") or "🔗"),
        "color": str(config.get("color") or AGENT_COLORS[0]),
    }
    return normalized, []


# ── 系统现状描述（注入主智能体提示词，替代 list_* 工具调用）──────────

def describe_tools() -> str:
    """输出系统当前注册的全部工具清单（名称、分类、风险等级与用途）。"""
    lines = []
    for meta in TOOL_REGISTRY.values():
        info = meta.to_dict()
        lines.append(f"- {info['name']} [{info['category_label']} | {info['risk_level']}]: {info['description']}")
    return "\n".join(lines) if lines else "（尚无任何注册工具）"


def describe_agents() -> str:
    """输出全部已注册智能体（内置与自定义）的清单，含名称、工具集与职责。"""
    seen = {cfg["name"]: cfg for cfg in AgentStore.list_all()}
    for name, cfg in BUILTIN_TEMPLATES.items():
        seen.setdefault(name, cfg)

    lines = []
    for cfg in seen.values():
        kind = "内置" if cfg.get("is_builtin") else "自定义"
        lines.append(
            f"- {cfg.get('icon', '')} {cfg['name']}（{kind}）"
            f" tools={cfg.get('tools', [])} — {cfg.get('description', '')}"
        )
    return "\n".join(lines) if lines else "（尚无任何智能体配置）"


def describe_harnesses() -> str:
    """输出全部已注册 harness 流水线（内置与自定义）的清单，含策略与阶段构成。"""
    seen = {cfg["name"]: cfg for cfg in HarnessStore.list_all()}
    for cfg in BUILTIN_HARNESSES:
        seen.setdefault(cfg["name"], cfg)

    lines = []
    for cfg in seen.values():
        flow = " -> ".join(s.get("agent_name", "?") for s in cfg.get("stages", []))
        purpose = cfg.get("purpose") or cfg.get("description", "")
        lines.append(
            f"- {cfg.get('icon', '')} {cfg['name']}（{cfg.get('strategy', 'sequential')}）"
            f" 流程: {flow} — {purpose}"
        )
    return "\n".join(lines) if lines else "（尚无任何 harness 配置）"
