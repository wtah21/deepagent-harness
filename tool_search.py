"""
主智能体的工具语义检索。

工具数量超过阈值时，主智能体不再一次性绑定全部工具，而是只绑定：
  - 核心工具（文件系统 / 代码执行 / 规划）与编排工具，始终常驻；
  - 一个 `search_tool` 检索工具。

检索链路：
  用户需求（必要时结合上下文改写）→ BGE-M3 dense 向量 + Milvus BM25 稀疏检索
  → hybrid_search 用 RRFRanker 融合 → 本地 bge-reranker-v2-m3 重排 → 取 Top-K 工具名。

工具索引只在工具名/描述发生变化时增量重建，签名缓存在 deepagent_data/tool_index.json。
任何外部依赖（Milvus / embedding / rerank）不可用时都降级为关键词匹配，不影响主流程。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Callable, Optional

import requests
from langchain_core.tools import BaseTool, StructuredTool

from config import (
    EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_DIM, EMBEDDING_MODEL,
    MILVUS_COLLECTION, MILVUS_TOKEN, MILVUS_URL,
    RERANK_BATCH_SIZE, RERANK_DEVICE, RERANK_MAX_LENGTH, RERANK_MODEL_PATH,
    TOOL_INDEX_CACHE_PATH, TOOL_SEARCH_CANDIDATES, TOOL_SEARCH_REWRITE, TOOL_SEARCH_TOP_K,
    debug_log,
)
from tool_registry import TOOL_REGISTRY

SEARCH_TOOL_NAME = "search_tool"
_HTTP_TIMEOUT = 20
_MILVUS_TIMEOUT = 5

_client_lock = threading.Lock()
_client: Any = None
_collection_ready = False

# 复用 HTTP 连接，省掉每次 embedding 请求的 TLS 握手（约 100-200ms）
_http_local = threading.local()


def _http() -> requests.Session:
    session = getattr(_http_local, "session", None)
    if session is None:
        session = requests.Session()
        _http_local.session = session
    return session


# ── 工具文档 ──────────────────────────────────────────────────────────

def _tool_doc(tool: Any) -> dict[str, str]:
    """把工具整理成可检索的文档：名称、分类、描述与用于 BM25 的全文。"""
    name = getattr(tool, "name", "") or getattr(tool, "__name__", "")
    meta = TOOL_REGISTRY.get(name)
    description = (getattr(tool, "description", "") or "").strip()
    if not description and meta:
        description = meta.description
    category = meta.category if meta else "delegation"
    text = f"{name}\n分类：{category}\n{description}"
    return {"name": name, "category": category, "description": description, "text": text}


def _signature(doc: dict[str, str]) -> str:
    return f"{doc['category']}\x00{doc['description']}"


# ── 向量化与重排 ──────────────────────────────────────────────────────

def _embed(texts: list[str]) -> list[list[float]]:
    """调用硅基流动 BGE-M3 生成 dense 向量。"""
    if not EMBEDDING_API_KEY:
        raise RuntimeError("未配置 SiliconEmbedding_API_KEY")
    resp = _http().post(
        f"{EMBEDDING_BASE_URL}/embeddings",
        headers={
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
            "encoding_format": "float",
        },
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json().get("data") or []
    vectors = [item.get("embedding") or [] for item in data]
    if len(vectors) != len(texts) or any(len(v) != EMBEDDING_DIM for v in vectors):
        raise RuntimeError(
            f"embedding 返回维度异常：期望 {len(texts)}x{EMBEDDING_DIM}，实际 "
            f"{[len(v) for v in vectors]}"
        )
    return vectors


# ── 本地重排（bge-reranker-v2-m3）──────────────────────────────────────

class LocalReranker:
    """延迟敏感场景下的 CrossEncoder 封装。

    优化点：
      - 进程内单例，模型只加载一次；
      - 启动后后台预热，首次检索不再等待模型加载与 CUDA kernel 初始化；
      - fp16（GPU）+ max_length 截断 + 批量推理，工具描述很短，256 token 足够；
      - CPU 模式下限制线程数，避免和 Streamlit 抢核反而更慢。
    """

    def __init__(self) -> None:
        self._torch: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = "cpu"
        self._loading = False
        self._load_lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def loading(self) -> bool:
        return self._loading

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._loading = True
            try:
                self._load()
            finally:
                self._loading = False

    def _load(self) -> None:
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = RERANK_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        started = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(
            RERANK_MODEL_PATH, local_files_only=True
        )
        # transformers 4.56 起用 dtype，旧版本用 torch_dtype
        dtype_key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL_PATH, local_files_only=True, **{dtype_key: dtype},
        )
        model = model.to(device).eval()

        if device == "cpu":
            # 默认会用满 32 线程，和界面/其它请求抢核，反而拖慢单次推理
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        debug_log(
            f"[tool-search] 重排模型就绪 device={device} dtype={dtype} "
            f"加载 {(time.perf_counter() - started):.2f}s"
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        """批量打分；分数越大越相关，只用于排序，不再做 sigmoid。"""
        self.load()
        torch = self._torch
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(documents), RERANK_BATCH_SIZE):
                batch = [[query, doc] for doc in documents[start:start + RERANK_BATCH_SIZE]]
                inputs = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=RERANK_MAX_LENGTH, return_tensors="pt",
                )
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                logits = self._model(**inputs).logits.view(-1)
                scores.extend(logits.float().tolist())
        return scores

    def warmup(self) -> None:
        """跑一次极短推理，把 CUDA kernel / 线程池预热掉。"""
        self.score("预热", ["warmup"])


_reranker: Optional[LocalReranker] = None
_reranker_lock = threading.Lock()


def get_reranker() -> LocalReranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = LocalReranker()
    return _reranker


def warmup_reranker() -> None:
    """供启动时的后台线程调用；失败只记录日志，不影响检索降级路径。"""
    try:
        get_reranker().warmup()
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[tool-search] 重排模型预热失败：{type(exc).__name__}: {exc}")


def _rerank(query: str, candidates: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    """本地 CrossEncoder 重排；候选不超过 top_k 时直接跳过模型。"""
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates[:top_k]
    reranker = get_reranker()
    if not reranker.ready and reranker.loading:
        # 模型还在后台加载：不要让 18s 的加载时间进入首屏延迟，先用 RRF 顺序
        debug_log("[tool-search] 重排模型后台加载中，本轮先用 RRF 顺序")
        return candidates[:top_k]
    try:
        started = time.perf_counter()
        documents = [f"{c['name']}：{c['description']}" for c in candidates]
        scores = reranker.score(query, documents)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        debug_log(
            f"[tool-search] 本地重排 {len(candidates)}->{top_k} "
            f"耗时 {(time.perf_counter() - started) * 1000:.0f}ms"
        )
        return [candidates[i] for i in order[:top_k]]
    except Exception as exc:  # noqa: BLE001 重排只是优化，失败保留 RRF 顺序
        debug_log(f"[tool-search] 重排失败，保留 RRF 顺序：{type(exc).__name__}: {exc}")
        return candidates[:top_k]


# ── Milvus ────────────────────────────────────────────────────────────

def _get_client() -> Any:
    """惰性创建 Milvus 客户端（导入本模块时不会建立连接）。"""
    global _client
    if _client is not None:
        return _client
    if not MILVUS_URL:
        raise RuntimeError("未配置 Mivilus_URL")
    from pymilvus import MilvusClient

    with _client_lock:
        if _client is None:
            _client = MilvusClient(uri=MILVUS_URL, token=MILVUS_TOKEN, timeout=_MILVUS_TIMEOUT)
    return _client


def _ensure_collection(client: Any) -> None:
    """集合不存在时创建：dense 向量 + BM25 Function 自动生成的 sparse 向量。"""
    global _collection_ready
    if _collection_ready or client.has_collection(MILVUS_COLLECTION):
        _collection_ready = True
        return
    from pymilvus import DataType, Function, FunctionType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("name", DataType.VARCHAR, max_length=128)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("description", DataType.VARCHAR, max_length=4096)
    schema.add_field(
        "text", DataType.VARCHAR, max_length=8192,
        enable_analyzer=True, analyzer_params={"type": "chinese"},
    )
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="text_bm25",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    ))

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="AUTOINDEX", metric_type="BM25")
    client.create_collection(
        collection_name=MILVUS_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    _collection_ready = True
    debug_log(f"[tool-search] 创建集合 {MILVUS_COLLECTION}")


def _load_signatures() -> dict[str, str]:
    try:
        data = json.loads(TOOL_INDEX_CACHE_PATH.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in (data.get("tools") or {}).items()}
    except (OSError, ValueError):
        return {}


def _save_signatures(signatures: dict[str, str]) -> None:
    try:
        TOOL_INDEX_CACHE_PATH.write_text(
            json.dumps({"tools": signatures}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        debug_log(f"[tool-search] 写入索引缓存失败：{exc}")


def sync_tool_index(tools: list[Any]) -> int:
    """把可检索工具写入 Milvus；只对签名变化的工具重新向量化。

    返回本次真正写入（新增或更新）的工具数。任何异常都被吞掉并降级，
    因为工具检索是增强能力，不能阻塞主智能体的正常启动。
    """
    docs = [_tool_doc(t) for t in tools]
    docs = [d for d in docs if d["name"]]
    if not docs:
        return 0

    signatures = _load_signatures()
    changed = [d for d in docs if signatures.get(d["name"]) != _signature(d)]
    try:
        client = _get_client()
        _ensure_collection(client)

        if changed:
            vectors = _embed([d["text"] for d in changed])
            rows = [
                {
                    "id": d["name"],
                    "name": d["name"],
                    "category": d["category"],
                    "description": d["description"][:4000],
                    "text": d["text"],
                    "dense": vector,
                }
                for d, vector in zip(changed, vectors)
            ]
            client.upsert(collection_name=MILVUS_COLLECTION, data=rows)
            for d in changed:
                signatures[d["name"]] = _signature(d)

        # 已下线/改名的工具：从集合里删掉，避免检索到不存在的工具
        current_names = {d["name"] for d in docs}
        stale = [name for name in signatures if name not in current_names]
        if stale:
            client.delete(collection_name=MILVUS_COLLECTION, filter=f"id in {json.dumps(stale)}")
            for name in stale:
                signatures.pop(name, None)

        _save_signatures(signatures)
        if changed or stale:
            debug_log(f"[tool-search] 索引同步：写入 {len(changed)}，清理 {len(stale)}")
        return len(changed)
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[tool-search] 索引同步失败（降级为关键词检索）：{type(exc).__name__}: {exc}")
        return 0


# ── 混合检索 ──────────────────────────────────────────────────────────

def _keyword_fallback(query: str, candidates: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    """Milvus / embedding 不可用时的兜底：按字符二元组重叠度排序。"""
    tokens = set(re.findall(r"[a-z0-9_]+", query.lower())) | set(
        p for p in re.findall(r"[\u4e00-\u9fff]{2,}", query)
    )
    scored = []
    for doc in candidates:
        haystack = f"{doc['name']} {doc['description']}".lower()
        score = sum(1 for token in tokens if token and token in haystack)
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scored[:top_k] if score > 0] or candidates[:top_k]


def search_tools(
    query: str,
    candidates: list[dict[str, str]],
    exclude: Optional[list[str]] = None,
    top_k: int = TOOL_SEARCH_TOP_K,
) -> list[dict[str, str]]:
    """混合检索工具：dense（BGE-M3）+ sparse（BM25）→ RRF → rerank。"""
    excluded = {str(name) for name in (exclude or [])}
    pool = [d for d in candidates if d["name"] not in excluded]
    if not pool:
        return []

    try:
        client = _get_client()
        _ensure_collection(client)
        vector = _embed([query])[0]

        from pymilvus import AnnSearchRequest, RRFRanker

        filter_expr = None
        if excluded:
            filter_expr = f"name not in {json.dumps(sorted(excluded))}"
        requests_ = [
            AnnSearchRequest(
                data=[vector], anns_field="dense",
                param={"metric_type": "COSINE"}, limit=TOOL_SEARCH_CANDIDATES,
            ),
            AnnSearchRequest(
                data=[query], anns_field="sparse",
                param={"metric_type": "BM25"}, limit=TOOL_SEARCH_CANDIDATES,
            ),
        ]
        raw = client.hybrid_search(
            collection_name=MILVUS_COLLECTION,
            reqs=requests_,
            ranker=RRFRanker(60),
            limit=TOOL_SEARCH_CANDIDATES,
            filter=filter_expr,
            output_fields=["name", "category", "description", "text"],
        )
        by_name = {d["name"]: d for d in pool}
        hits: list[dict[str, str]] = []
        for hit_list in raw:
            for hit in hit_list:
                entity = hit.get("entity") or hit.get("fields") or {}
                name = entity.get("name") or str(hit.get("id", ""))
                doc = by_name.get(name)
                if doc and doc not in hits:
                    hits.append(doc)
        if hits:
            debug_log(f"[tool-search] hybrid 命中 {len(hits)}，query={query[:60]}")
            return _rerank(query, hits, top_k)
        debug_log(f"[tool-search] hybrid 无命中，降级关键词 query={query[:60]}")
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[tool-search] 混合检索失败（降级关键词）：{type(exc).__name__}: {exc}")

    return _keyword_fallback(query, pool, top_k)


def warmup_search(candidates: list[dict[str, str]]) -> None:
    """预热完整检索链路：embedding 连接、Milvus 查询通道与重排。

    首次检索的 2-3 秒主要花在 gRPC 通道与 HTTPS 握手，这里提前跑一次小查询消掉。
    """
    sample = candidates[:2]
    if not sample:
        return
    try:
        started = time.perf_counter()
        search_tools("预热工具检索", sample, top_k=1)
        debug_log(f"[tool-search] 检索链路预热完成 {(time.perf_counter() - started) * 1000:.0f}ms")
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[tool-search] 检索链路预热失败：{type(exc).__name__}: {exc}")


# ── 意图改写 ──────────────────────────────────────────────────────────

_PRONOUNS = ("那", "这个", "那个", "它", "上面", "刚才", "前一", "继续", "再")


def _needs_rewrite(question: str) -> bool:
    """短问题或带指代/省略的问题才需要结合上下文改写。"""
    text = (question or "").strip()
    if len(text) <= 12:
        return True
    return any(word in text for word in _PRONOUNS)


def rewrite_query(question: str, messages: list[Any], llm: Any) -> str:
    """用一次轻量模型调用把依赖上下文的问题改写成可独立检索的描述。"""
    question = (question or "").strip()
    if not question or llm is None or not TOOL_SEARCH_REWRITE or not _needs_rewrite(question):
        return question

    recent: list[str] = []
    for message in messages[-6:]:
        role = getattr(message, "type", "") or ""
        if role == "human":
            content = str(getattr(message, "content", "") or "").strip()
            if content:
                recent.append(f"用户：{content[:300]}")
        elif role == "ai":
            content = str(getattr(message, "content", "") or "").strip()
            if content:
                recent.append(f"助手：{content[:200]}")

    prompt = (
        "把用户最新的一句话改写成一个独立、完整、便于检索工具的需求描述。\n"
        "要求：结合对话上下文补全指代与省略的信息；只输出一句中文，不要解释，不要加引号。\n\n"
        f"近期对话：\n{chr(10).join(recent) if recent else '（无）'}\n\n"
        f"用户最新提问：{question}"
    )
    try:
        rewritten = str(llm.invoke(prompt).content or "").strip().strip('"').strip("“”")
        if rewritten and len(rewritten) <= 300:
            debug_log(f"[tool-search] 改写 query: {question[:40]} -> {rewritten[:80]}")
            return rewritten
    except Exception as exc:  # noqa: BLE001
        debug_log(f"[tool-search] 查询改写失败：{type(exc).__name__}: {exc}")
    return question


# ── 暴露给主智能体的检索工具 ──────────────────────────────────────────

def run_search(
    query: str,
    exclude: Optional[list[str]],
    state: dict,
    llm: Any,
    candidates: list[dict[str, str]],
) -> tuple[list[str], str]:
    """执行一次检索，返回（命中的工具名，给模型的提示文本）。

    自动合并本轮已检索过的工具，避免第二次检索重复返回同一批结果。
    """
    history = [str(name) for name in (state.get("tool_search_history") or [])]
    excluded = list(dict.fromkeys([*(exclude or []), *history]))
    final_query = rewrite_query(query, list(state.get("messages") or []), llm)

    found = search_tools(final_query, candidates, exclude=excluded)
    names = [d["name"] for d in found]
    if not names:
        return [], (
            f"没有检索到可用工具（已排除 {len(excluded)} 个）。"
            "请换一个更通用的关键词重试，或直接用常驻工具完成该步骤。"
        )

    hint = f"（已排除：{excluded}）" if excluded else ""
    return names, (
        f"Available tools: {names}{hint}\n"
        "这些工具已经挂载，下一轮可以直接调用。"
        "如果仍不合适，请把上面这些工具名放进 exclude 再检索一次，不要重复检索同一批工具。"
    )


def build_search_tool(
    all_tools: list[Any],
    llm_provider: Callable[[], Any],
) -> BaseTool:
    """构造 `search_tool` 的模型侧 schema（挂载逻辑由主图的 select_tools 节点执行）。"""
    candidates = [_tool_doc(t) for t in all_tools]
    candidates = [d for d in candidates if d["name"] != SEARCH_TOOL_NAME]

    def _run(
        query: str,
        exclude: Optional[list[str]] = None,
    ) -> str:
        """按需求检索工具。第一次没找到合适的工具时，把已返回的工具名放进 exclude 再检索一次。"""
        # 该函数仅在把 search_tool 当普通工具直接调用时才会走到；
        # 主图的 select_tools 节点会直接调用 run_search 以携带完整状态。
        return "search_tool 需要在主智能体图中执行。"

    return StructuredTool.from_function(
        func=_run,
        name=SEARCH_TOOL_NAME,
        description=(
            "按任务需求检索可用工具（子智能体、MCP、A2A 等）。"
            "传入一句完整的需求描述，返回最匹配的工具名并自动挂载到下一轮；"
            "第一次结果不合适时，把已返回的工具名放进 exclude 再检索一次。"
        ),
    )


def build_candidates(
    all_tools: list[Any], exclude_names: Optional[set[str]] = None
) -> list[dict[str, str]]:
    """预计算可检索工具文档；常驻工具不参与检索，检索到也没有额外收益。"""
    skipped = set(exclude_names or ())
    return [
        d for d in (_tool_doc(t) for t in all_tools)
        if d["name"] and d["name"] != SEARCH_TOOL_NAME and d["name"] not in skipped
    ]
