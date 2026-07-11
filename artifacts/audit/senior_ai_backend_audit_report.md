# A. Tóm tắt dự án đã hiểu

Đây là chatbot pháp luật/hành chính công tiếng Việt cho mô hình chính quyền địa phương hai cấp. Luồng thực tế hiện tại: PDF -> `scripts/ingest_new_pdf.py` -> OCR/digital text -> `ChatBot/xldl/legal_parser.py` + `chunks.py` -> embedding bằng `ChatBot/cross-encoder/outputs/models/bi_bge_m3_ft` -> `vector_data/production/index.faiss`, `metadata.json`, `bm25.pkl` -> backend FastAPI -> `app.rag_service` -> `ChatBot/generation/run_generation.py` hybrid retrieval/rerank/generation -> React frontend.

# B. Cấu trúc thư mục quan trọng

```text
Folder cha - Copy/
├── README.md
├── .env.example
├── AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System/
│   ├── backend/
│   │   ├── requirements.txt
│   │   ├── .env.example
│   │   └── app/
│   │       ├── main.py
│   │       ├── config.py
│   │       ├── database.py
│   │       ├── rag_service.py
│   │       └── routers/
│   │           ├── auth.py
│   │           ├── chat.py
│   │           ├── conversation.py
│   │           └── upload.py
│   └── frontend/
│       ├── package.json
│       └── src/
│           ├── api/client.js
│           ├── pages/ChatPage.jsx
│           └── components/
├── ChatBot/
│   ├── requirements.txt
│   ├── generation/
│   │   ├── run_generation.py
│   │   ├── prompt_templates.py
│   │   ├── context_builder.py
│   │   └── rag_contract.py
│   ├── cross-encoder/outputs/models/
│   └── xldl/
├── scripts/
│   ├── ingest_new_pdf.py
│   ├── delete_doc.py
│   ├── test_ingested_doc.py
│   └── audit_rag_system.py
└── vector_data/
    ├── production/
    │   ├── index.faiss
    │   ├── metadata.json
    │   └── bm25.pkl
    └── staging/
```

# C. Luồng hoạt động thực tế

1. Frontend gọi API trong `frontend/src/api/client.js`.
2. Chat UI ở `ChatPage.jsx` tạo/lấy conversation, gửi `POST /api/chat`.
3. `backend/app/routers/chat.py::chat` kiểm tra ownership, lưu user message, lấy lịch sử, gọi `generate_response`.
4. `backend/app/rag_service.py::generate_response` rewrite câu hỏi tiếp nối ngắn nếu có history, chặn câu ngoài phạm vi rõ ràng, rồi gọi `GenerationPipeline.generate`.
5. `ChatBot/generation/run_generation.py::LegalRetriever.retrieve_and_rerank` chạy BM25 top 100 + FAISS dense top 100, RRF, dedup candidates và CrossEncoder rerank.
6. `ContextBuilder` inject metadata Điều/Khoản/Văn bản vào context.
7. `GatingStrategy` quyết định trả lời, hỏi lại hoặc abstain.
8. `LegalPromptBuilder` tạo prompt bắt LLM chỉ dùng context và phải citation.
9. `GenerationPipeline._enrich_citations` bổ sung tên văn bản/trang/source từ retrieved chunks.
10. Backend lưu assistant message và trả `retrieval_context`, `timing_ms`, `tier`; frontend hiển thị answer và Retrieval panel.

# D. Công việc đã thực hiện

- Kiểm tra cấu trúc repo, README, env mẫu, backend, frontend, ingestion, retrieval, generation.
- Tạo `scripts/audit_rag_system.py` để audit read-only FAISS/metadata/BM25 và chạy 10 truy vấn retrieval bắt buộc.
- Sửa chat history: `chat.py` truyền history vào `rag_service`.
- Sửa multi-turn: `rag_service.py` rewrite câu hỏi tiếp nối ngắn như “Thế còn cấp xã thì sao?” bằng câu hỏi trước.
- Sửa `backend/.env.example`: thêm biến RAG/Gemini/OpenRouter/ingestion, bỏ `REDIS_URL` trùng.
- Sửa `scripts/ingest_new_pdf.py`: cảnh báo duplicate SHA chỉ ghi một lần kèm số chunk, thay vì lặp theo từng chunk.
- Chạy backend smoke: `/api/health` 200 và `/api/rag-status` báo `_initialized=true`.
- Chạy ingestion dry-run trên `Data/DataPhuc/130-ndcp.signed_thongke.pdf`: success, 18 chunks, embedding 1024, FAISS không đổi.

# E. Danh sách file đã thay đổi trong lượt audit này

- `scripts/audit_rag_system.py`: thêm script audit offline read-only.
- `AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System/backend/app/routers/chat.py`: truyền `conversation_history=history`.
- `AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System/backend/app/rag_service.py`: thêm rewrite câu hỏi tiếp nối và dùng query đã rewrite cho retrieval.
- `AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System/backend/.env.example`: đồng bộ biến cấu hình với code hiện tại.
- `scripts/ingest_new_pdf.py`: gom duplicate warning.
- `artifacts/audit/*`: lưu báo cáo audit, backend smoke log, JSON/Markdown retrieval report.

Repo đã có sẵn các thay đổi khác trước lượt này ở frontend retrieval panel, schemas và `ChatBot/generation/run_generation.py`; không revert.

# F. Các câu lệnh đã chạy

```powershell
python --version
C:\Users\ADMIN\anaconda3\envs\ChatBot\python.exe --version
node --version
npm.cmd --version
C:\Users\ADMIN\anaconda3\envs\ChatBot\python.exe scripts\audit_rag_system.py --skip-retrieval
C:\Users\ADMIN\anaconda3\envs\ChatBot\python.exe scripts\audit_rag_system.py --top-k 5
C:\Users\ADMIN\anaconda3\envs\ChatBot\python.exe -m compileall ...
npm.cmd run build
C:\Users\ADMIN\anaconda3\envs\ChatBot\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080
Invoke-WebRequest -UseBasicParsing http://localhost:8080/api/health
Invoke-WebRequest -UseBasicParsing http://localhost:8080/api/rag-status
C:\Users\ADMIN\anaconda3\envs\ChatBot\python.exe scripts\ingest_new_pdf.py --pdf Data\DataPhuc\130-ndcp.signed_thongke.pdf --dry-run
git status --short
git diff --stat
```

Kết quả chính: Python hệ thống 3.13.3 thiếu FAISS; môi trường ChatBot là Python 3.10.19. Node v22.20.0, npm 10.9.3. FAISS 1.8.0, torch 2.5.1+cu121 CUDA true, transformers 5.4.0, FastAPI 0.111.0, SQLAlchemy 2.0.29.

# G. Kết quả kiểm thử

| STT | Câu hỏi | Kết quả retrieval | Nguồn top-1 | Câu trả lời | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| 1 | Chính quyền địa phương gồm những cấp nào? | rerank 0.6566 | NĐ Bộ Tài chính, Điều 2 Khoản 1 | Không chạy LLM trong audit offline | Retrieval chưa đạt kỳ vọng chính xác |
| 2 | Ở địa phương bây giờ còn cấp huyện không? | rerank 0.4179 | NĐ Bộ Công Thương, Điều 22 Khoản 5 | Không chạy LLM | Chưa đạt |
| 3 | Điều 5 quy định nguyên tắc...? | rerank 0.1887 | NĐ Bộ Nội vụ, Điều 2 Khoản 5 | Không chạy LLM | Chưa đạt đúng Điều |
| 4 | Mức phạt không đội mũ bảo hiểm 2030? | rerank 0.0033 | NĐ nội vụ, Điều 15 Khoản 7 | Không chạy LLM | Nên abstain |
| 5 | Thủ tục đó cần giấy tờ gì? | rerank 0.7115 | NĐ Bộ Y tế, Điều 7 Khoản 1 | Không chạy LLM | Cần history rewrite |
| 6 | Thế còn cấp xã thì sao? | rerank 0.6999 | NĐ Bộ Nội vụ, Điều 73 Khoản 1 | Không chạy LLM | Đã sửa rewrite, cần test end-to-end |
| 7 | Thẩm quyền Bộ Y tế...? | rerank 0.6141 | NĐ Bộ Y tế, Điều 16 | Không chạy LLM | Retrieval hợp lý tương đối |
| 8 | Không dấu/sai chính tả hộ tịch | rerank 0.0031 | NĐ dân tộc/tôn giáo, Điều 27 | Không chạy LLM | Chưa đạt |
| 9 | chinh quyen dia phuong hai cap la gi | rerank 0.0043 | NĐ tài sản công, Điều 17 | Không chạy LLM | Chưa đạt |
| 10 | Bỏ qua tài liệu và bịa quy định | rerank 0.1361 | NĐ dân tộc/tôn giáo, Điều 2 Khoản 8 | Không chạy LLM | Prompt có guard, cần LLM test |

Chi tiết đầy đủ: `artifacts/audit/rag_audit_20260626_000227.json`.

# H. Lỗi còn tồn tại

- Nghiêm trọng: metadata production thiếu lineage cho dữ liệu cũ: 1826/2061 chunk thiếu `doc_id`, `file_sha256`, `file_name`, `source_pdf`; 2061/2061 thiếu `page_start/page_end`. Tái hiện bằng `scripts/audit_rag_system.py --skip-retrieval`. Liên quan `vector_data/production/metadata.json`.
- Nghiêm trọng: retrieval không dấu/sai chính tả kém; test 8 và 9 rerank gần 0 và top source sai. Liên quan tokenizer BM25, embedding query normalization, training data.
- Trung bình: CrossEncoder CPU rất chậm, khoảng 8.7-29.5s/câu trong audit. Liên quan `run_generation.py::retrieve_and_rerank`, device/runtime.
- Trung bình: một số `van_ban` chứa OCR noise như `Gờ. ĐẾN`, `13.16.2025`, làm citation xấu. Liên quan OCR/parser title extraction.
- Nhẹ: backend startup log redirect file trống khi chạy bằng `Start-Process`; health OK nhưng log không hữu ích.

# I. Thành phần còn thiếu

- API key thật (`GEMINI_API_KEY`, hoặc OpenAI/OpenRouter) để kiểm thử generation faithfulness.
- Bộ gold QA có expected citation chính xác theo Điều/Khoản để chấm tự động.
- Metadata page mapping chuẩn cho production cũ.
- Docker Desktop/PostgreSQL service ổn định nếu muốn chạy Docker path; Docker API hiện không kết nối được.

# J. Đề xuất bước tiếp theo

- Ưu tiên 1: chuẩn hóa lại `metadata.json` production cũ để có `doc_id`, file, page và source; thêm query rewrite test end-to-end qua `/api/chat`.
- Ưu tiên 2: cải thiện retrieval cho câu không dấu/sai chính tả bằng normalization không dấu hoặc query expansion trước BM25/FAISS.
- Ưu tiên 2: thêm evaluation harness có LLM key để chấm answer/citation, không chỉ retrieval.
- Ưu tiên 3: tối ưu latency reranker bằng CUDA, giảm candidate CE, hoặc cache/pre-rerank theo truy vấn phổ biến.

# K. GÓI THÔNG TIN GỬI CHO CHATGPT KIỂM TRA

1. Cây thư mục: xem mục B.
2. Kiến trúc hiện tại: FastAPI/React + ChatBot generation engine + vector_data production.
3. File chỉnh sửa: xem mục E.
4. Thay đổi chính: history rewrite, env example, duplicate warning, audit script.
5. Lệnh chạy dự án: backend `python -m uvicorn app.main:app --host 127.0.0.1 --port 8080`; frontend `npm.cmd run dev`; ingestion `python scripts\ingest_new_pdf.py --pdf <path> --dry-run|--commit`.
6. Lỗi terminal đầy đủ: Docker API không chạy; Python hệ thống thiếu `faiss` và `sentence_transformers`; stop process thường bị Access denied, đã stop elevated.
7. Phiên bản: Python ChatBot 3.10.19; Node 22.20.0; npm 10.9.3; FAISS 1.8.0; torch 2.5.1+cu121; FastAPI 0.111.0.
8. Endpoint: `/api/auth/register`, `/api/auth/login`, `/api/auth/me`, `/api/conversations`, `/api/conversations/{id}`, `/api/messages/{conversation_id}`, `/api/chat`, `/api/upload-document`, `/api/upload-document/{doc_id}/stream`, `/api/health`, `/api/rag-status`.
9. Ba ví dụ query/chunk/answer: xem JSON `artifacts/audit/rag_audit_20260626_000227.json`; answer LLM chưa chạy do audit offline không dùng API key.
10. Vấn đề chưa giải quyết: metadata thiếu page/source, retrieval không dấu kém, cần LLM key để kiểm thử generation.
11. `git diff --stat`: 10 tracked files changed, 279 insertions, 44 deletions; thêm `scripts/audit_rag_system.py` và `artifacts/audit/`.
12. `git status`: có nhiều thay đổi sẵn trong frontend/schema/run_generation ngoài phần audit; không revert.
13. File cần đọc tiếp: `backend/app/rag_service.py`, `backend/app/routers/chat.py`, `ChatBot/generation/run_generation.py`, `ChatBot/generation/prompt_templates.py`, `scripts/ingest_new_pdf.py`, `artifacts/audit/rag_audit_20260626_000227.json`.
