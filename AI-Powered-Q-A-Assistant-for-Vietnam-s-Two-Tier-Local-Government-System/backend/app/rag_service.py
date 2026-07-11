"""
RAG Service — Legal RAG Pipeline thay thế OpenAI.

Import trực tiếp từ ChatBot/cross-encoder/generation/ pipeline:
  1. RetrievalReranker (retrieve + rerank top-K passages)
  2. GenerationPipeline (gating + LLM generation)

Singleton pattern: load models 1 lần, tái sử dụng cho mọi request.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
from pathlib import Path
from typing import Optional
from loguru import logger

# ── Add ChatBot folder to sys.path ──────────────────────────────────────────
# Structure: Folder cha / ChatBot / generation /
#            Folder cha / AI-Powered-... / backend / app / rag_service.py (this file)

_THIS_DIR = Path(__file__).resolve().parent                          # app/
_BACKEND_DIR = _THIS_DIR.parent                                      # backend/
_PROJECT_DIR = _BACKEND_DIR.parent                                   # AI-Powered-.../
_FOLDER_CHA = _PROJECT_DIR.parent                                    # Folder cha/
_CHATBOT_ROOT = _FOLDER_CHA / "ChatBot"                              # ChatBot/

# Add ChatBot/ to path so `from generation.xxx import ...` works
if str(_CHATBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHATBOT_ROOT))

# ── Lazy imports (heavy deps: torch, transformers, faiss) ─────────────────────

_pipeline = None          # GenerationPipeline singleton
_reranker = None          # Deprecated, no longer loaded to save RAM
_initialized = False

_OUT_OF_SCOPE_REPLY = (
    "Xin lỗi, tôi là trợ lý pháp lý chuyên về văn bản quy phạm pháp luật Việt Nam, "
    "đặc biệt trong lĩnh vực tổ chức và phân cấp chính quyền địa phương. Tôi chỉ có "
    "thể hỗ trợ các câu hỏi liên quan đến luật, nghị định, thông tư và quy định pháp luật. "
    "Vui lòng đặt câu hỏi phù hợp để tôi hỗ trợ bạn."
)

_LEGAL_SCOPE_KEYWORDS = {
    "luật", "pháp luật", "nghị định", "nghị quyết", "thông tư", "quy định",
    "điều", "khoản", "điểm", "chương", "thẩm quyền", "phân quyền", "phân cấp",
    "chính quyền", "địa phương", "ủy ban", "ubnd", "chủ tịch", "bộ", "sở",
    "cơ quan", "xã", "phường", "tỉnh", "huyện", "sáp nhập", "hộ tịch",
    "khai sinh", "nuôi con nuôi", "thủ tục", "hồ sơ", "cấp giấy", "y tế",
    "nội vụ", "tư pháp", "tài chính", "xây dựng", "đất đai", "thống kê",
    "công thương", "nông nghiệp", "môi trường", "chính phủ", "quốc hội",
}

_OUT_OF_SCOPE_MARKERS = {
    "con gì", "là con gì", "mấy vậy", "bằng mấy", "1+1", "2+2",
    "ăn gì", "uống gì", "thời tiết", "bóng đá",
}

_FOLLOW_UP_MARKERS = {
    "thế còn", "vậy còn", "còn", "thủ tục đó", "việc đó", "trường hợp đó",
    "cái đó", "nó", "đó", "như vậy", "mất bao lâu", "cần giấy tờ gì",
}


def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _normalize_for_scope(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_obvious_out_of_scope(query: str) -> bool:
    """
    Fast guard for clearly non-legal chatter/math/animal questions. This avoids
    spending 60-90s on retrieval + API just to return the standard refusal.
    """
    q = _normalize_for_scope(query)
    if not q:
        return True
    if any(keyword in q for keyword in _LEGAL_SCOPE_KEYWORDS):
        return False
    return any(marker in q for marker in _OUT_OF_SCOPE_MARKERS)


def _message_role(message) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def _message_content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _rewrite_follow_up_query(query: str, conversation_history=None) -> str:
    """
    Lightweight query rewrite for short context-dependent follow-ups.
    It keeps full standalone questions unchanged and only prepends the previous
    user question when the current query depends on "đó/còn/thế còn" context.
    """
    if not conversation_history:
        return query

    normalized = _normalize_for_scope(query)
    if not normalized:
        return query

    looks_contextual = any(marker in normalized for marker in _FOLLOW_UP_MARKERS)
    if not looks_contextual:
        return query

    previous_user_messages = []
    for message in conversation_history:
        if _message_role(message) != "user":
            continue
        content = _message_content(message).strip()
        if content and _normalize_for_scope(content) != normalized:
            previous_user_messages.append(content)

    if not previous_user_messages:
        return query

    previous_question = previous_user_messages[-1]
    return (
        f"Câu hỏi trước: {previous_question}\n"
        f"Câu hỏi tiếp nối cần trả lời đầy đủ: {query}"
    )


def init_rag_pipeline(
    backend: str = "huggingface",
    model: str = "auto",
    api_key: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> None:
    """
    Khởi tạo RAG pipeline (gọi 1 lần lúc startup).

    Load:
      - LegalRetriever (bi-encoder + cross-encoder + FAISS + BM25 hybrid)
      - GenerationPipeline (LLM client + gating + context builder)
    """
    global _pipeline, _initialized

    if _initialized:
        logger.info("RAG pipeline already initialized, skipping")
        return

    logger.info("🔧 Initializing RAG pipeline...")
    t0 = time.time()

    # ── Import generation modules ──
    from generation.run_generation import GenerationPipeline, LegalRetriever, PipelineLogger
    from generation.llm_client import (
        LLMClient, LLMConfig, LLMBackend, LLMMode, select_local_model_for_vram
    )
    from generation.run_generation import GEN_EVAL_DIR

    # ── 1. Setup PyTorch Threading and Device ──
    import torch
    torch.set_num_threads(1)
    
    # Check GPU VRAM to avoid slow paging bottlenecks.
    # Default is 3.5GB because 4GB laptop GPUs are often reported as slightly
    # below 4.0GiB by CUDA.
    retriever_device_override = os.environ.get("RAG_RETRIEVER_DEVICE", os.environ.get("RAG_DEVICE", "auto")).strip().lower()
    min_retriever_vram_gb = _get_float_env("RAG_RETRIEVER_MIN_VRAM_GB", 3.5)
    device = "cpu"
    total_vram = 0.0
    if torch.cuda.is_available():
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if retriever_device_override == "cpu":
                logger.info("💾 RAG_RETRIEVER_DEVICE=cpu. Loading retriever on CPU.")
            elif retriever_device_override == "cuda" or total_vram >= min_retriever_vram_gb:
                device = "cuda"
                logger.info(
                    f"💾 GPU has {total_vram:.1f}GB VRAM. Loading retriever on CUDA "
                    f"(min={min_retriever_vram_gb:.1f}GB)."
                )
            else:
                logger.warning(
                    f"💾 GPU has only {total_vram:.1f}GB VRAM "
                    f"(< {min_retriever_vram_gb:.1f}GB). Loading retriever on CPU."
                )
        except Exception as e:
            logger.warning(f"⚠️ Failed to inspect GPU properties: {e}. Defaulting to CPU.")
    else:
        logger.info("💾 No CUDA GPU detected. Defaulting to CPU.")

    # ── 2. Build LLM Client ──
    backend_map = {
        "huggingface": LLMBackend.HUGGINGFACE,
        "qwen": LLMBackend.QWEN,
        "gemini": LLMBackend.GEMINI,
        "openai": LLMBackend.OPENAI,
        "openrouter": LLMBackend.OPENROUTER,
        "llama_cpp": LLMBackend.LLAMA_CPP,
        "placeholder": LLMBackend.PLACEHOLDER,
    }

    llm_backend = backend_map.get(backend, LLMBackend.HUGGINGFACE)

    # Auto-select model name
    model_name = model
    model_path = None

    if backend == "huggingface":
        if model_name == "auto" or not model_name:
            model_name = select_local_model_for_vram(reserved_vram_gb=1.0)
        model_path = model_name
    elif backend == "qwen":
        if not model_name or model_name == "auto":
            model_name = "qwen/qwen3-30b-a3b:free"
    elif backend == "gemini":
        if not model_name or model_name == "auto":
            model_name = "gemini-2.5-flash"
    elif backend == "openai":
        if not model_name or model_name == "auto":
            model_name = "gpt-4o-mini"

    # API key from env if not provided
    if not api_key:
        env_map = {
            "qwen": "OPENROUTER_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        api_key = os.environ.get(env_map.get(backend, ""), "")

    # ── 2a. Build local fallback client (Qwen3 on GPU) ──
    # Khi Gemini/API hết quota → tự động fallback sang model local
    fallback_client = None
    if llm_backend in (LLMBackend.GEMINI, LLMBackend.OPENAI, LLMBackend.OPENROUTER, LLMBackend.QWEN):
        if device == "cuda" and 0 < total_vram < 6.0:
            logger.warning(
                f"⚠️ GPU chỉ có {total_vram:.1f}GB VRAM; bỏ local Qwen fallback để dành VRAM cho retriever CUDA."
            )
        else:
            logger.info("🔧 Building local Qwen3 fallback client (GPU)...")
            try:
                # Chọn model Qwen3 phù hợp với VRAM còn lại
                # Reserved 3.0GB cho bi-encoder + cross-encoder + overhead (buộc chọn Qwen3-0.6B để tránh OOM GPU)
                local_model = select_local_model_for_vram(reserved_vram_gb=3.0)
                local_config = LLMConfig(
                    backend=LLMBackend.HUGGINGFACE,
                    mode=LLMMode.DEV,
                    model_name=local_model,
                    model_path=local_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    context_length=4096,
                )
                fallback_client = LLMClient(local_config)
                logger.info(f"✅ Local fallback ready: {local_model}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load local fallback: {e}")
                logger.warning("   Pipeline sẽ chỉ dùng API, không có fallback.")

    llm_config = LLMConfig(
        backend=llm_backend,
        mode=LLMMode.DEV,
        model_name=model_name,
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        context_length=4096,
        api_key=api_key or None,
    )

    client = LLMClient(llm_config, fallback_client=fallback_client)
    local_client_for_pipeline = fallback_client or client

    # ── 3. Build Pipeline ──
    GEN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_logger = PipelineLogger(log_dir=str(GEN_EVAL_DIR / "logs"), append=True)

    # Retriever dùng LegalRetriever (đã load FAISS + metadata)
    retriever = LegalRetriever(device=device)

    _pipeline = GenerationPipeline(
        retriever=retriever,
        local_client=local_client_for_pipeline,
        api_client=client,
        logger=pipeline_logger,
        mode=LLMMode.DEV,
        local_first=bool(fallback_client),
        ce_sigmoid_mode=False,
    )

    _initialized = True
    elapsed = time.time() - t0
    logger.info(f"✅ RAG pipeline initialized in {elapsed:.1f}s")
    logger.info(f"   Backend: {backend} | Model: {model_name} | Fallback: {'local Qwen3 (GPU)' if fallback_client else 'none'}")


def generate_response(user_message: str, conversation_history=None) -> dict:
    """
    Xử lý câu hỏi qua RAG pipeline, trả về answer text + retrieval context.

    Luồng:
      1. GenerationPipeline.generate(query) -> Retrieval (FAISS + BM25) + Rerank + Gating + LLM
      2. Format answer + citations → return dict

    Args:
        user_message: câu hỏi từ user
        conversation_history: lịch sử DB messages, dùng để rewrite câu hỏi tiếp nối ngắn

    Returns:
        dict: {
            "answer": str,
            "retrieval_context": [
                {
                    "rank": int,
                    "text": str,
                    "score_retrieval": float,
                    "score_rerank": float,
                    "van_ban": str,
                    "dieu": str | None,
                    "khoan": str | None,
                    "diem": str | None,
                    "chunk_id": int,
                }
            ],
            "timing_ms": float,
            "tier": str,
        }
    """
    global _pipeline, _initialized

    if not _initialized or _pipeline is None:
        raise RuntimeError("RAG pipeline chưa sẵn sàng. Vui lòng thử lại sau.")

    t_start = time.time()

    effective_query = _rewrite_follow_up_query(user_message, conversation_history)
    if effective_query != user_message:
        logger.info(
            "Rewrote follow-up query for retrieval: {!r} -> {!r}",
            user_message[:80],
            effective_query[:160],
        )

    if _is_obvious_out_of_scope(user_message):
        elapsed_ms = (time.time() - t_start) * 1000
        logger.info(f"⚡ Fast out-of-scope response in {elapsed_ms:.0f}ms: {user_message[:80]}...")
        return {
            "answer": f"{_OUT_OF_SCOPE_REPLY}\n\n_⏱ {elapsed_ms:.0f}ms | Tier: NONE_",
            "retrieval_context": [],
            "timing_ms": elapsed_ms,
            "tier": "NONE",
        }

    # ── Step 1: Retrieval + Rerank + Gating + LLM Generation ──
    logger.info(f"🔍 Processing query with Hybrid RAG: {effective_query[:80]}...")
    output, metadata = _pipeline.generate(
        query=effective_query,
        verbose=False,
    )

    # Debug: log raw response and parsed output
    logger.debug(f"   [RAW] raw_response: {(output.raw_response or '')[:500]}")
    logger.debug(f"   [PARSED] answer: {(output.answer or 'NONE')[:200]}")
    logger.debug(f"   [PARSED] abstain: {output.abstain}, citations: {len(output.citations)}")
    logger.debug(f"   [PARSED] decision: {output.decision}")

    elapsed_ms = (time.time() - t_start) * 1000

    # ── Step 2: Format response ──
    if output.abstain:
        answer_text = output.reason_detail or "Không đủ căn cứ pháp lý để trả lời câu hỏi này."
        if output.clarification_question:
            answer_text += f"\n\n❓ {output.clarification_question}"
    else:
        answer_text = output.answer or "Không có câu trả lời."

        # Append citations
        if output.citations:
            citations_lines = []
            for cit in output.citations:
                citations_lines.append(f"- {cit.to_str()}")
            answer_text += "\n\n📎 **Trích dẫn:**\n" + "\n".join(citations_lines)

    tier = metadata.get("tier", "?").upper()
    answer_text += f"\n\n_⏱ {elapsed_ms:.0f}ms | Tier: {tier}_"

    logger.info(f"✅ Response generated in {elapsed_ms:.0f}ms (Tier: {tier})")

    # ── Step 3: Build retrieval context for frontend ──
    retrieval_context = []
    top_chunks = metadata.get("retrieved_chunks", [])

    for rank, chunk in enumerate(top_chunks, 1):
        retrieval_context.append({
            "rank": rank,
            "text": getattr(chunk, "text", "")[:500],
            "score_retrieval": round(float(getattr(chunk, "score_retrieval", 0.0)), 4),
            "score_rerank": round(float(getattr(chunk, "score_rerank", 0.0)), 4),
            "van_ban": getattr(chunk, "van_ban", "") or "",
            "dieu": str(chunk.dieu) if getattr(chunk, "dieu", None) else None,
            "khoan": str(chunk.khoan) if getattr(chunk, "khoan", None) else None,
            "diem": str(chunk.diem) if getattr(chunk, "diem", None) else None,
            "chunk_id": getattr(chunk, "chunk_id", -1),
        })

    return {
        "answer": answer_text,
        "retrieval_context": retrieval_context,
        "timing_ms": round(elapsed_ms, 1),
        "tier": tier,
    }
