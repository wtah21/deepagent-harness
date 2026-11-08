"""
人机协同（human-in-the-loop）：主智能体缺关键信息时暂停本轮，向用户追问。

用 LangGraph 的 interrupt 实现：工具里调用 interrupt() 让图停在当前位置，
检查点会保存暂停位置与追问内容；用户作答后用 Command(resume=...) 从原位置恢复，
本轮任务继续执行。答案作为工具结果回传模型，不会变成一条新的用户消息。

两种追问形态由模型按场景选择：
  - input：需要用户提供信息（用户自由输入）；
  - choice：任务有多种执行方案（界面把方案列成按钮供选择）。
"""
from typing import Annotated, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool
from langgraph.types import Command, interrupt

# 面板上最多展示的候选方案数量，避免方案过多把页面撑长
MAX_OPTIONS = 4

# 追问的两种形态
ASK_INPUT = "input"
ASK_CHOICE = "choice"

# 前端据此识别「这是追问工具」，不把它渲染成普通工具调用块
ASK_USER_TOOL_NAME = "ask_user"


def _ask_user(
    question: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    kind: str = ASK_INPUT,
    options: Optional[List[str]] = None,
    context: str = "",
) -> Command:
    """暂停当前任务向用户追问，拿到答复后作为工具结果继续执行。"""
    question = str(question or "").strip()
    candidate = [
        str(item).strip() for item in (options or []) if str(item).strip()
    ][:MAX_OPTIONS]
    # 方案为空时只能让用户自己输入，否则退化成「没有选项的选择题」
    mode = ASK_CHOICE if (str(kind or "").strip() == ASK_CHOICE and candidate) else ASK_INPUT
    context = str(context or "").strip()

    # 图在这里暂停：面板读到 question / kind / options / context，
    # 用户作答后以 Command(resume=答案) 恢复，interrupt 的返回值就是答案
    answer = interrupt({
        "question": question,
        "kind": mode,
        "options": candidate,
        "context": context,
    })

    if mode == ASK_CHOICE:
        rendered = "\n".join(f"{idx}. {opt}" for idx, opt in enumerate(candidate, 1))
        notice = (
            f"用户在以下方案中选择了：{answer}\n\n"
            f"（此前的问题：{question}\n候选方案：\n{rendered}）\n"
            "请按用户选定的方案继续完成本轮任务，不要重复追问同一个问题。"
        )
    else:
        notice = (
            f"用户已答复：{answer}\n\n"
            f"（此前的问题：{question}）\n"
            "请据此继续完成本轮任务，不要重复追问同一个问题。"
        )
    return Command(update={
        "messages": [
            ToolMessage(content=notice, tool_call_id=tool_call_id, name=ASK_USER_TOOL_NAME)
        ],
    })


ask_user = StructuredTool.from_function(
    func=_ask_user,
    name=ASK_USER_TOOL_NAME,
    description=(
        "在缺少关键信息、无法从上下文或工具结果中推断时，暂停任务向用户追问。\n\n"
        "Args:\n"
        "    question: 需要用户回答的问题，一句话说清缺的是什么。\n"
        "    kind: 追问形态。需要用户提供信息（用户名、路径、目标地址、偏好细节等）时填 "
        "'input'；任务存在多种可行方案、需要用户挑一个时填 'choice'。\n"
        "    options: 仅 kind='choice' 时提供：2-4 条互斥、可直接执行的方案，"
        "每条写清「做法 + 主要代价或产出」，界面会把它们渲染成按钮。\n"
        "    context: 为什么要问，简述你已经掌握的情况与不同选择的后果。\n\n"
        "Returns:\n"
        "    用户输入的答复或选中的方案（工具结果形式返回）。图会在调用本工具时暂停，"
        "用户作答后从原位置继续执行，不会中断整轮任务。"
    ),
)
