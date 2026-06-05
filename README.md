# 🤖 TwoTierGovQA — Trợ Lý Pháp Lý AI

Chatbot RAG tra cứu văn bản pháp luật Việt Nam về tổ chức chính quyền địa phương hai cấp.

**Stack:** FastAPI · React · PostgreSQL · FAISS · BM25 · Gemini / OpenAI

## ✨ Tính Năng

- **Hybrid RAG** — BM25 (keyword) + FAISS (semantic) → RRF → Cross-Encoder rerank
- **2-Tier LLM** — Tự chọn Local (Qwen3) hoặc API (Gemini/GPT) theo độ phức tạp
- **Upload PDF** — OCR → Parse cấu trúc luật → Embed → Cập nhật index, streaming tiến trình real-time
- **JWT Auth** — Đăng ký/đăng nhập, lưu lịch sử hội thoại trong PostgreSQL

---

## 🚀 Bắt Đầu Nhanh

### 1. Clone & cấu hình

```bash
git clone https://github.com/Luphuc2005/TwoTierGovQA.git
cd TwoTierGovQA
cp .env.example .env
# Mở .env → điền GEMINI_API_KEY (hoặc OPENAI_API_KEY)
```

### 2. Tải dữ liệu từ Google Drive

📦 **[Google Drive — vector_data & models](https://drive.google.com/drive/folders/1ZjAU4mNg4YkI7ff4svhbfxHAjMns3Ztu?hl=vi)**

Tải về và đặt vào đúng vị trí:

```
vector_data/production/     ← index.faiss, metadata.json, bm25.pkl
ChatBot/cross-encoder/outputs/models/  ← bi_bge_m3_ft/, ce_bge_reranker_ft_v6/
```

### 3A. Chạy bằng Docker (đơn giản nhất)

```bash
cd AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System
docker compose up --build -d
# Frontend: http://localhost:3000
# Backend:  http://localhost:8080
```

### 3B. Chạy thủ công

**Database** — PostgreSQL trên port 5433:
```bash
docker run -d --name pg-legal \
  -e POSTGRES_DB=legal_chatbot -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -p 5433:5432 postgres:16-alpine
```

**Backend:**
```bash
cd AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System/backend
pip install -r requirements.txt
cp .env.example .env        # điền DATABASE_URL, GEMINI_API_KEY, JWT_SECRET_KEY
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

**ChatBot Engine** (môi trường riêng — cần cho RAG & ingestion):
```bash
conda create -n ChatBot python=3.11 -y && conda activate ChatBot
cd ChatBot && pip install -r requirements.txt
pip install paddlepaddle     # hoặc paddlepaddle-gpu nếu có CUDA
```

**Frontend:**
```bash
cd AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System/frontend
npm install && npm run dev   # http://localhost:5173
```

---

## 📁 Cấu Trúc Thư Mục

```
TwoTierGovQA/
├── AI-Powered-.../
│   ├── backend/            # FastAPI — auth, chat, upload routers
│   ├── frontend/           # React + Vite + TailwindCSS
│   └── docker-compose.yml
├── ChatBot/
│   ├── generation/         # RAG pipeline (retrieval + rerank + generation)
│   ├── cross-encoder/      # Fine-tuned bi-encoder & cross-encoder
│   └── xldl/               # OCR, legal parser, chunker
├── scripts/                # ingest_new_pdf.py, delete_doc.py
├── vector_data/production/ # FAISS + BM25 + metadata (tải từ Drive)
├── .env.example
└── README.md
```

---

## 🔌 API

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/api/auth/register` | Đăng ký |
| POST | `/api/auth/login` | Đăng nhập → JWT |
| GET | `/api/conversations` | Danh sách hội thoại |
| POST | `/api/chat` | Gửi câu hỏi → trả lời RAG |
| POST | `/api/upload-document` | Upload PDF |
| GET | `/api/upload-document/{id}/stream` | Tiến trình xử lý (SSE) |

Swagger UI: http://localhost:8080/docs

---

## ⚙️ Biến Môi Trường Chính

| Biến | Mô tả |
|------|-------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key (tuỳ chọn) |
| `DATABASE_URL` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | Secret key ký JWT |
| `RAG_BACKEND` | `gemini` / `openai` / `huggingface` / `placeholder` |
| `RAG_MODEL` | `auto` để tự chọn theo backend |
| `INGEST_PYTHON_EXE` | Python path của môi trường ChatBot |

Xem đầy đủ trong [.env.example](.env.example).

---

## 📄 Nạp Tài Liệu Mới

```bash
conda activate ChatBot

# Thử nghiệm:
python scripts/ingest_new_pdf.py --pdf path/to/file.pdf --dry-run

# Nạp thật vào production:
python scripts/ingest_new_pdf.py --pdf path/to/file.pdf --commit
```

Pipeline: Validate → OCR/Extract → Parse cấu trúc luật → Chunk → Embed → Commit (có backup & rollback).

---

## 📄 License

MIT — xem [LICENSE](LICENSE).
