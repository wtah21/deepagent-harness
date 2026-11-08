"""
让 ChatOpenAI 保留第三方 provider 的思考过程（reasoning_content）。

langchain-openai 1.6 起明确不解析非 OpenAI 官方字段（reasoning_content /
reasoning_details），而 DeepSeek 等 OpenAI 兼容接口每次都会返回该字段。
这里只在「取结果」这一步把它补回 AIMessage.additional_kwargs，请求侧
仍然走原生 ChatOpenAI，行为与之前完全一致。
"""
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI

# DeepSeek / 多数兼容接口使用的思考过程字段名
REASONING_KEY = "reasoning_content"


def _reasoning_of(message: Any) -> str:
    """从 dict 或 OpenAI SDK 的 message 对象里取出 reasoning_content。"""
    if isinstance(message, dict):
        text = message.get(REASONING_KEY)
    else:
        text = getattr(message, REASONING_KEY, None)
        if text is None:
            # SDK 把未声明的字段放进 model_extra
            text = (getattr(message, "model_extra", None) or {}).get(REASONING_KEY)
    return text if isinstance(text, str) else ""


class ReasoningChatOpenAI(ChatOpenAI):
    """在返回结果里补回 reasoning_content 的 ChatOpenAI。"""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        reasoning = ""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                reasoning = _reasoning_of(choices[0].get("message") or {})
        else:
            try:
                reasoning = _reasoning_of(response.choices[0].message)
            except (AttributeError, IndexError):
                reasoning = ""
        if reasoning and result.generations:
            result.generations[0].message.additional_kwargs[REASONING_KEY] = reasoning
        return result

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None
    ):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None or not isinstance(generation_chunk.message, AIMessageChunk):
            return generation_chunk
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        delta = (choices[0].get("delta") if choices else None) or {}
        reasoning = _reasoning_of(delta)
        if reasoning:
            generation_chunk.message.additional_kwargs[REASONING_KEY] = reasoning
        return generation_chunk
