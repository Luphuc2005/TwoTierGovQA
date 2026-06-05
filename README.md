# 🤖 Trợ Lý Pháp Lý AI — Legal Chatbot RAG

> Chatbot AI tra cứu văn bản quy phạm pháp luật Việt Nam, đặc biệt về tổ chức và phân cấp chính quyền địa phương hai cấp.

**Stack:** FastAPI · React/Vite · PostgreSQL · Redis · FAISS · BM25 · Google Gemini / OpenAI

---

## 📋 Mục Lục

- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cách 1: Chạy bằng Docker (Đơn giản nhất)](#-cách-1-chạy-bằng-docker-đơn-giản-nhất)
- [Cách 2: Chạy thủ công (Manual)](#-cách-2-chạy-thủ-công-manual)
- [Chuẩn bị dữ liệu vector (FAISS/BM25)](#-chuẩn-bị-dữ-liệu-vector-faiss--bm25)
- [API Endpoints](#-api-endpoints)
- [Biến môi trường](#-biến-môi-trường)
- [Tính năng nổi bật](#-tính-năng-nổi-bật)

---

## 🏗 Kiến Trúc Hệ Thống

```
┌─────────────────┐     REST / SSE      ┌──────────────────────────┐
│   Frontend      │ ◄─────────────────► │  Backend (FastAPI)        │
│  React + Vite   │                     │  - Auth (JWT)             │
│  TailwindCSS    │                     │  - Chat router            │
└─────────────────┘                     │  - Upload router (SSE)    │
                                        └────────────┬─────────────┘
                                                     │
                        ┌────────────────────────────┼─────────────────┐
                        │                            │                 │
                   ┌────▼────┐             ┌─────────▼──────┐   ┌─────▼──────┐
                   │PostgreSQL│             │  RAG Pipeline  │   │   Redis    │
                   │(hội thoại│             │  rag_service.py│   │(rate limit)│
                   │ & user)  │             └────────┬───────┘   └────────────┘
                   └──────────┘                      │
                                    ┌────────────────┼─────────────────┐
                                    │                │                 │
                             ┌──────▼────┐   ┌───────▼──────┐  ┌──────▼──────┐
                             │  FAISS    │   │  BM25 Index  │  │  LLM (API/  │
                             │  Dense    │   │  Keyword     │  │  Local)     │
                             │  Search   │   │  Search      │  └─────────────┘
                             └───────────┘   └──────────────┘
```

**Luồng RAG:**
1. Query → BM25 (keyword) + FAISS (dense) song song
2. RRF merge → Cross-Encoder rerank (có thể bỏ qua nếu điểm tin cậy cao)
3. Gating → phân tuyến: Local LLM / API LLM / Từ chối trả lời
4. Sinh câu trả lời kèm trích dẫn điều khoản

---

## 📁 Cấu Trúc Thư Mục

```
legal-chatbot-rag/
│
├── AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System/
│   ├── backend/                    # FastAPI server
│   │   ├── app/
│   │   │   ├── main.py             # Entry point
│   │   │   ├── config.py           # Cấu hình từ .env
│   │   │   ├── models.py           # SQLAlchemy ORM (User/Conversation/Message)
│   │   │   ├── schemas.py          # Pydantic schemas
│   │   │   ├── crud.py             # Database operations
│   │   │   ├── auth.py             # JWT authentication
│   │   │   ├── rag_service.py      # RAG pipeline singleton (entry point)
│   │   │   ├── middleware.py       # Logging + Rate limiting
│   │   │   └── routers/
│   │   │       ├── auth.py         # POST /auth/register, /auth/login
│   │   │       ├── chat.py         # POST /chat, GET /messages/{id}
│   │   │       ├── conversation.py # GET/POST/DELETE /conversations
│   │   │       └── upload.py       # POST /upload-document (SSE streaming)
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── .env.example
│   │
│   ├── frontend/                   # React + Vite + TailwindCSS
│   │   ├── src/
│   │   │   ├── api/client.js       # Axios API client
│   │   │   ├── context/AuthContext.jsx
│   │   │   ├── components/         # ChatWindow, Sidebar, UploadModal...
│   │   │   └── pages/              # LoginPage, RegisterPage, ChatPage
│   │   ├── Dockerfile
│   │   └── nginx.conf
│   │
│   ├── docker-compose.yml          # Chạy toàn bộ stack
│   └── .env.example
│
├── ChatBot/                        # RAG Engine (Python modules)
│   ├── generation/                 # Pipeline tạo câu trả lời
│   │   ├── run_generation.py       # LegalRetriever + GenerationPipeline
│   │   ├── llm_client.py           # Client cho Gemini/OpenAI/HuggingFace
│   │   ├── gating.py               # Gating strategy (Pass/Abstain/Ask-back)
│   │   ├── context_builder.py      # Xây dựng context từ chunks
│   │   └── prompt_templates.py     # Prompt templates tiếng Việt
│   ├── cross-encoder/              # Fine-tuned bi-encoder và cross-encoder
│   │   └── outputs/models/         # ← Tải model về đây (xem bên dưới)
│   ├── xldl/                       # Xử lý Tài Liệu
│   │   ├── legal_parser.py         # Parser cấu trúc văn bản luật
│   │   ├── chunks.py               # Document chunker
│   │   ├── OCR_paddle_protonX.py   # OCR với PaddleOCR
│   │   └── rules_base_protonx.py   # Merge và làm sạch OCR output
│   └── requirements.txt
│
├── scripts/
│   ├── ingest_new_pdf.py           # Nạp PDF mới vào FAISS index
│   ├── delete_doc.py               # Xoá tài liệu khỏi index
│   └── test_ingested_doc.py        # Kiểm tra tài liệu đã nạp
│
├── vector_data/
│   └── production/                 # FAISS + BM25 + metadata (KHÔNG commit)
│       ├── index.faiss             # ← Tải về hoặc build từ PDF
│       ├── metadata.json           # ← Tải về hoặc build từ PDF
│       └── bm25.pkl                # ← Tải về hoặc build từ PDF
│
├── data/
│   └── uploads/                    # PDF đã upload (KHÔNG commit)
│
├── .env.example                    # Mẫu biến môi trường
├── .gitignore
└── README.md
```

---

## 💻 Yêu Cầu Hệ Thống

| Thành phần | Phiên bản tối thiểu |
|------------|---------------------|
| Python | 3.11+ |
| Node.js | 18+ |
| PostgreSQL | 14+ |
| Docker | 24+ (nếu dùng Docker) |
| RAM | 8GB+ (16GB nếu chạy model local) |
| VRAM | 4GB+ (nếu chạy Qwen3 local) |
| Poppler | Mới nhất (cho OCR file quét) |

---

## 🐳 Cách 1: Chạy Bằng Docker (Đơn Giản Nhất)

> **Lưu ý:** Docker compose sẽ chạy PostgreSQL, Redis, Backend và Frontend. Bạn vẫn cần chuẩn bị dữ liệu vector trước (xem [phần tiếp theo](#-chuẩn-bị-dữ-liệu-vector-faiss--bm25)).

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/legal-chatbot-rag.git
cd legal-chatbot-rag

# 2. Cấu hình biến môi trường
cp .env.example .env
# Mở .env và điền API key:
#   GEMINI_API_KEY=your-key-here   (hoặc OPENAI_API_KEY)

# 3. Chuẩn bị vector data (xem phần "Chuẩn bị dữ liệu vector" bên dưới)

# 4. Build và chạy
cd AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System
docker compose up --build -d

# 5. Kiểm tra
# Frontend: http://localhost:3000
# Backend API: http://localhost:8080
# API Docs: http://localhost:8080/docs
```

Kiểm tra logs nếu có lỗi:
```bash
docker compose logs backend -f
docker compose logs postgres -f
```

---

## 🛠 Cách 2: Chạy Thủ Công (Manual)

### Bước 1: Clone & Cấu Hình

```bash
git clone https://github.com/<your-username>/legal-chatbot-rag.git
cd legal-chatbot-rag
cp .env.example .env
# Mở .env, điền GEMINI_API_KEY (hoặc key LLM khác)
```

### Bước 2: PostgreSQL

```bash
# Tạo database (đảm bảo PostgreSQL đang chạy)
psql -U postgres -c "CREATE DATABASE legal_chatbot;"
```

Hoặc dùng Docker chỉ cho DB:
```bash
docker run -d --name pg-legal \
  -e POSTGRES_DB=legal_chatbot \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5433:5432 \
  postgres:16-alpine
```

### Bước 3: Backend (FastAPI)

```bash
# Tạo môi trường Python cho Backend
cd AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System/backend

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

# Cấu hình .env
cp .env.example .env
# Điền DATABASE_URL, GEMINI_API_KEY, JWT_SECRET_KEY

# Chạy server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

API tại: http://localhost:8080  
Swagger UI: http://localhost:8080/docs

### Bước 4: ChatBot Engine (RAG + OCR — môi trường riêng)

Tính năng OCR và Embedding cần môi trường riêng với PaddleOCR, torch:

```bash
# Dùng Conda (khuyến nghị)
conda create -n ChatBot python=3.11 -y
conda activate ChatBot

cd ChatBot
pip install -r requirements.txt

# Cài PaddlePaddle (CPU):
pip install paddlepaddle

# Cài PaddlePaddle (GPU CUDA 11.8):
# pip install paddlepaddle-gpu==2.6.0.post118 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# Cài Poppler (Windows — cần cho pdf2image):
# Tải tại: https://github.com/oschwartz10612/poppler-windows/releases
# Giải nén và đặt đường dẫn vào .env: POPPLER_PATH=D:\poppler\Library\bin
```

Sau khi cài, cập nhật `INGEST_PYTHON_EXE` trong `.env` trỏ tới Python của môi trường này.

### Bước 5: Frontend (React)

```bash
cd AI-Powered-Q-A-Assistant-for-Vietnam-s-Two-Tier-Local-Government-System/frontend

npm install
npm run dev
```

Frontend tại: http://localhost:5173

---

## 🗄 Chuẩn Bị Dữ Liệu Vector (FAISS + BM25)

> ⚠️ **Thư mục `vector_data/production/` không được commit lên Git** vì các file có thể nặng hàng trăm MB.

Sau khi clone repo, thư mục `vector_data/production/` sẽ trống. Có 2 cách để có dữ liệu:

### Cách A: Tải sẵn từ Google Drive (Nhanh nhất)

Tải về 3 file và đặt vào `vector_data/production/`:

| File | Mô tả |
|------|-------|
| `index.faiss` | FAISS dense vector index |
| `metadata.json` | Metadata của từng chunk văn bản luật |
| `bm25.pkl` | BM25 keyword index |

📦 **[Tải vector data & models → Google Drive](https://drive.google.com/drive/folders/1ZjAU4mNg4YkI7ff4svhbfxHAjMns3Ztu?hl=vi)**

Sau khi tải:
```bash
# Đặt file vào đúng thư mục:
vector_data/
└── production/
    ├── index.faiss     ← copy vào đây
    ├── metadata.json   ← copy vào đây
    └── bm25.pkl        ← copy vào đây
```

> **Model weights** (bi-encoder, cross-encoder) cũng cần tải về và đặt vào `ChatBot/cross-encoder/outputs/models/`. Link trong cùng thư mục Google Drive.

### Cách B: Build lại từ PDF (Tự chuẩn bị)

1. **Chuẩn bị file PDF** văn bản luật cần đưa vào hệ thống.

2. **Kích hoạt môi trường ChatBot** (môi trường đã cài ở bước 4).

3. **Chạy script nạp tài liệu** từ root của repo:
```bash
# Thử nghiệm không ghi (dry-run):
python scripts/ingest_new_pdf.py --pdf path/to/document.pdf --dry-run

# Xác nhận ghi vào production (nạp thật):
python scripts/ingest_new_pdf.py --pdf path/to/document.pdf --commit
```

4. **Nạp nhiều PDF** bằng cách chạy lệnh trên cho từng file.

5. **Xác nhận** dữ liệu đã nạp:
```bash
python scripts/test_ingested_doc.py
```

**Lần đầu tiên** (chưa có index): Script sẽ báo lỗi vì không có production index. Bạn cần tạo index trống trước:
```bash
# Kích hoạt môi trường ChatBot, sau đó:
python -c "
import faiss, json, pickle
import numpy as np
from pathlib import Path

prod = Path('vector_data/production')
prod.mkdir(parents=True, exist_ok=True)

# Tạo FAISS index trống với dimension 1024 (khớp với bi_bge_m3_ft)
index = faiss.IndexFlatIP(1024)
faiss.write_index(index, str(prod / 'index.faiss'))

# Tạo metadata trống
with open(prod / 'metadata.json', 'w') as f:
    json.dump([], f)

# Tạo BM25 trống
from rank_bm25 import BM25Okapi
bm25_data = {'bm25': BM25Okapi([[]]), 'corpus': []}
with open(prod / 'bm25.pkl', 'wb') as f:
    pickle.dump(bm25_data, f)

print('✅ Empty production index created!')
"
```

---

## 🔌 API Endpoints

| Method | Endpoint | Auth | Mô tả |
|--------|----------|------|-------|
| POST | `/api/auth/register` | Không | Đăng ký tài khoản |
| POST | `/api/auth/login` | Không | Đăng nhập, nhận JWT |
| GET | `/api/auth/me` | JWT | Thông tin user hiện tại |
| GET | `/api/conversations` | JWT | Danh sách cuộc hội thoại |
| POST | `/api/conversations` | JWT | Tạo cuộc hội thoại mới |
| DELETE | `/api/conversations/{id}` | JWT | Xoá cuộc hội thoại |
| GET | `/api/messages/{conv_id}` | JWT | Lịch sử tin nhắn |
| POST | `/api/chat` | JWT | Gửi câu hỏi, nhận trả lời RAG |
| POST | `/api/upload-document` | JWT | Upload PDF |
| GET | `/api/upload-document/{doc_id}/stream` | JWT (query) | Theo dõi tiến trình xử lý PDF (SSE) |
| GET | `/api/health` | Không | Health check |
| GET | `/api/rag-status` | Không | Trạng thái RAG pipeline |

---

## ⚙️ Biến Môi Trường

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `GEMINI_API_KEY` | *(bắt buộc nếu dùng Gemini)* | Google Gemini API key |
| `OPENAI_API_KEY` | *(bắt buộc nếu dùng OpenAI)* | OpenAI API key |
| `DATABASE_URL` | `postgresql://...localhost:5433/...` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | *(bắt buộc)* | Secret key ký JWT — đổi trước khi deploy |
| `RAG_BACKEND` | `gemini` | Backend LLM: `gemini`, `openai`, `huggingface`, `placeholder` |
| `RAG_MODEL` | `auto` | Tên model, `auto` để tự chọn |
| `RAG_MAX_TOKENS` | `2048` | Max token câu trả lời |
| `RAG_TEMPERATURE` | `0.1` | Độ sáng tạo (0=chính xác, 1=sáng tạo) |
| `INGEST_PYTHON_EXE` | *(Windows path)* | Python của môi trường ChatBot conda |
| `POPPLER_PATH` | *(Windows path)* | Đường dẫn Poppler cho OCR file quét |
| `REDIS_URL` | `redis://localhost:6380/0` | Redis URL (tuỳ chọn) |
| `RATE_LIMIT_PER_MINUTE` | `30` | Số request/phút tối đa |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Các origin được phép |

---

## ✨ Tính Năng Nổi Bật

- 🔍 **Hybrid RAG**: Kết hợp BM25 (từ khóa) + FAISS (ngữ nghĩa) → Reciprocal Rank Fusion
- ⚡ **Conditional Rerank**: Tự động bỏ qua Cross-Encoder khi điểm tin cậy cao (giảm latency 60–90%)
- 🧠 **2-Tier LLM**: Tự động chọn Local LLM (nhanh) hoặc API LLM (chất lượng cao) theo độ phức tạp
- 🔄 **API Fallback**: Tự động chuyển về model local nếu API hết quota hoặc gặp lỗi
- 📄 **Upload & Ingest PDF**: Upload PDF → OCR → Parse cấu trúc luật → Embed → Cập nhật index, hiển thị tiến trình thời gian thực
- 💬 **Lịch sử hội thoại**: Lưu toàn bộ lịch sử chat trong PostgreSQL, có tiêu đề tự động
- 🛡️ **Phạm vi câu hỏi**: Từ chối nhanh câu hỏi ngoài phạm vi pháp luật mà không tốn tài nguyên RAG
- 🔐 **JWT Auth**: Đăng nhập/đăng ký an toàn với token hết hạn sau 24h

---

## 🧪 Kiểm Tra Nhanh

```bash
# Sau khi backend đang chạy:

# 1. Đăng ký
curl -X POST http://localhost:8080/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123", "full_name": "Test User"}'

# 2. Đăng nhập — lấy token
TOKEN=$(curl -s -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}' | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 3. Tạo cuộc hội thoại
CONV_ID=$(curl -s -X POST http://localhost:8080/api/conversations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test"}' | python -c "import sys,json; print(json.load(sys.stdin)['id'])")

# 4. Gửi câu hỏi
curl -X POST http://localhost:8080/api/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"conversation_id\": $CONV_ID, \"message\": \"UBND cấp xã có những thẩm quyền gì?\"}"
```

---

## 📄 License

MIT License — xem [LICENSE](LICENSE) để biết thêm chi tiết.
#   T w o T i e r G o v Q A  
 