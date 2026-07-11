"""
config.py — CẤU HÌNH PIPELINE DUY NHẤT
========================================
Chỉ cần sửa file này, rồi chạy:  python run_pipeline.py

Mọi tham số pipeline đều nằm ở đây.
"""

import os

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHẾ ĐỘ CHẠY
# ═══════════════════════════════════════════════════════════════════════════════

# "reranked"  → chạy từ file reranked JSONL (bỏ qua retrieval)
# "query"     → chạy 1 câu hỏi đơn lẻ
# "eval"      → chạy evaluation trên goldset
# "interactive" → hỏi đáp tương tác
RUN_MODE = "query"  # "reranked", "query", "eval", "interactive"

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LLM — CHỌN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

# Backend: "huggingface" (chạy local GPU), "qwen" (API OpenRouter), 
#          "gemini" (API Google), "openai" (API OpenAI), "placeholder" (test)
BACKEND = "gemini"

# Tên model — tuỳ backend:
#   huggingface : "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "auto" (tự chọn theo VRAM)
#   qwen        : "qwen/qwen3-30b-a3b:free", "qwen/qwen3-8b:free"
#   gemini      : "gemini-2.0-flash"
#   openai      : "gpt-4o-mini"
MODEL = "gemini-2.5-flash"

# API Key (chỉ cần khi dùng API backend, để "" nếu chạy local)
#   qwen    → OPENROUTER_API_KEY
#   gemini  → GEMINI_API_KEY
#   openai  → OPENAI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. THAM SỐ LLM
# ═══════════════════════════════════════════════════════════════════════════════

MAX_TOKENS = 2048           # Số token tối đa cho câu trả lời
TEMPERATURE = 0.1          # 0.0 = chính xác, 1.0 = sáng tạo (pháp luật nên dùng 0.1)
TOP_P = 0.9                # Nucleus sampling
CONTEXT_LENGTH = 8192      # Context window (Gemini 2.5 Flash hỗ trợ 1M tokens)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. INPUT — DỮ LIỆU ĐẦU VÀO
# ═══════════════════════════════════════════════════════════════════════════════

# File reranked JSONL (dùng khi RUN_MODE = "reranked")
RERANKED_INPUT = r"..\data\pipeline_results_v6.jsonl"

# Câu hỏi đơn lẻ (dùng khi RUN_MODE = "query")
QUERY = "Thẩm quyền quản lý nhà nước của Bộ Xây dựng được phân cấp cho cơ quan nào?"


# Max samples (giới hạn số queries khi chạy reranked/eval, None = chạy hết)
MAX_SAMPLES = 5

# ═══════════════════════════════════════════════════════════════════════════════
# 5. OUTPUT — KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════════════════

VERBOSE = True             # In chi tiết từng bước
LOG_APPEND = True          # True = ghi tiếp file log cũ, False = tạo file log mới
LOG_DIR = None             # Thư mục log (None = mặc định outputs/generation/logs)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE MODE
# ═══════════════════════════════════════════════════════════════════════════════

PIPELINE_MODE = "dev"      # "dev" (nhanh, tiết kiệm) hoặc "demo" (chất lượng cao)
LOCAL_FIRST = True         # True = ưu tiên chạy local, chỉ gọi API cho câu khó

# ═══════════════════════════════════════════════════════════════════════════════
# 7. RETRIEVAL + RERANK PIPELINE (dùng khi RUN_MODE = "query" hoặc "interactive")
# ═══════════════════════════════════════════════════════════════════════════════

# Paths tới model fine-tuned (từ notebook v6)
#   Để None → dùng default paths trong retrieval_rerank.py
BI_MODEL_PATH  = None      # e.g. r"..\..\outputs\models\bi_bge_m3_ft"
CE_MODEL_PATH  = None      # e.g. r"..\..\outputs\models\ce_bge_reranker_ft_v6"
FAISS_INDEX_PATH = None    # e.g. r"..\notebooks\outputs\tmp\faiss_v4.index"

# Tham số retrieval
TOP_N_RETRIEVE = 100       # Số candidates từ FAISS
TOP_K_RERANK   = 5         # Số kết quả sau rerank (top-K cho generation)
CE_BATCH_SIZE  = 64        # Batch size cho cross-encoder

# ═══════════════════════════════════════════════════════════════════════════════
# 8. TỐI ƯU RETRIEVAL RUNTIME
# ═══════════════════════════════════════════════════════════════════════════════

# Chỉ ảnh hưởng cách chạy runtime, không giảm số lượng ứng viên retrieval.
RETRIEVAL_CE_BATCH_SIZE = 32
RETRIEVAL_CACHE_MAX_SIZE = 128

# Bật cơ chế bỏ qua Cross-Encoder khi BM25/RRF + Dense đều rất chắc.
# Ngưỡng mặc định đặt bảo thủ để ưu tiên giữ chất lượng hiện tại.
ENABLE_CONDITIONAL_RERANK = True
RERANK_SKIP_MIN_DENSE_SCORE = 0.70
RERANK_SKIP_MIN_DENSE_MARGIN = 0.10
RERANK_SKIP_MIN_RRF_SCORE = 0.0320
RERANK_SKIP_MIN_RRF_MARGIN = 0.0004
RERANK_SKIP_PROXY_SCORE = 8.0
