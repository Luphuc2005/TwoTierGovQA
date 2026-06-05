# 🤖 Legal Chatbot — RAG-Powered Chat Application

A production-ready fullstack legal chatbot built with **FastAPI**, **React**, **PostgreSQL**, and a **RAG (Retrieval-Augmented Generation)** pipeline.

---

## 📐 System Architecture

```
┌─────────────┐     HTTP/REST      ┌──────────────────┐
│   Frontend   │ ◄───────────────► │  Backend (API)   │
│  React/Vite  │                   │    FastAPI        │
│  TailwindCSS │                   │    Python         │
└─────────────┘                    └────────┬─────────┘
                                            │
                         ┌──────────────────┼──────────────────┐
                         │                  │                  │
                    ┌────▼─────┐      ┌────▼─────┐      ┌─────▼──────────┐
                    │PostgreSQL│      │Retriever │      │  LLM Generator │
                    │   (DB)   │      │(RAG)     │      │ (OpenAI/local) │
                    └──────────┘      └──────────┘      └────────────────┘
```

### Data Flow

```
1. User types message in React frontend
2. Frontend sends POST /api/chat with JWT token
3. Backend validates JWT, saves user message to PostgreSQL
4. Backend retrieves relevant legal context from knowledge sources
5. Backend sends question + retrieved context to the generator model
6. Backend saves assistant message to PostgreSQL
7. Backend returns response to frontend
8. Frontend renders response with markdown formatting
```

### Component Diagram

```
Frontend                          Backend                         External
┌──────────────────┐              ┌──────────────────────┐       ┌──────────┐
│  LoginPage       │              │  auth router         │       │          │
│  RegisterPage    │  ──────────► │  chat router         │ ────► │   RAG    │
│  ChatPage        │  ◄────────── │  conversation router │ ◄──── │  API     │
│    ├ Sidebar     │              ├──────────────────────┤       │          │
│    ├ ChatWindow  │              │  auth.py (JWT)       │       └──────────┘
│    ├ ChatInput   │              │  openai_service.py   │
│    └ Bubbles     │              │  crud.py             │       ┌──────────┐
│                  │              │  models.py           │ ────► │PostgreSQL│
│  AuthContext     │              │  middleware.py       │       └──────────┘
│  API Client      │              └──────────────────────┘
└──────────────────┘
```

---

## 🗄️ Database Schema

```sql
┌──────────────┐       ┌──────────────────┐       ┌──────────────────┐
│    users     │       │  conversations   │       │    messages       │
├──────────────┤       ├──────────────────┤       ├──────────────────┤
│ id (PK)      │──┐    │ id (PK)          │──┐    │ id (PK)          │
│ email        │  └───►│ user_id (FK)     │  └───►│ conversation_id  │
│ password_hash│       │ title            │       │ role             │
│ full_name    │       │ created_at       │       │ content          │
│ created_at   │       │ updated_at       │       │ created_at       │
│ updated_at   │       └──────────────────┘       └──────────────────┘
└──────────────┘
```

---

## 📁 Project Structure

```
legal_chatbot_api/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Environment settings
│   │   ├── database.py          # SQLAlchemy engine & session
│   │   ├── models.py            # ORM models (User, Conversation, Message)
│   │   ├── schemas.py           # Pydantic request/response schemas
│   │   ├── crud.py              # Database operations
│   │   ├── auth.py              # JWT authentication
│   │   ├── openai_service.py    # OpenAI API integration
│   │   ├── middleware.py        # Logging & rate limiting
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── auth.py          # POST /auth/register, /auth/login
│   │       ├── chat.py          # POST /chat, GET /messages/{id}
│   │       └── conversation.py  # GET/POST/DELETE /conversations
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── client.js        # Axios API client
│   │   ├── context/
│   │   │   └── AuthContext.jsx   # Auth state management
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── ChatInput.jsx
│   │   │   ├── ConversationSidebar.jsx
│   │   │   ├── LoadingDots.jsx
│   │   │   └── ProtectedRoute.jsx
│   │   ├── pages/
│   │   │   ├── LoginPage.jsx
│   │   │   ├── RegisterPage.jsx
│   │   │   └── ChatPage.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── nginx.conf
│   ├── Dockerfile
│   └── .gitignore
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint                    | Auth | Description                    |
|--------|-----------------------------|------|--------------------------------|
| POST   | `/api/auth/register`        | No   | Register a new user            |
| POST   | `/api/auth/login`           | No   | Login and receive JWT          |
| GET    | `/api/auth/me`              | Yes  | Get current user profile       |
| GET    | `/api/conversations`        | Yes  | List user's conversations      |
| POST   | `/api/conversations`        | Yes  | Create new conversation        |
| GET    | `/api/conversations/{id}`   | Yes  | Get conversation details       |
| DELETE | `/api/conversations/{id}`   | Yes  | Delete a conversation          |
| GET    | `/api/messages/{conv_id}`   | Yes  | Get messages in conversation   |
| POST   | `/api/chat`                 | Yes  | Send message & get AI response |
| POST   | `/api/chat/stream`          | Yes  | Send message & stream response |
| GET    | `/api/health`               | No   | Health check                   |

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Redis (optional)
- LLM API key (if using a hosted model)

---

### Option 1: Docker (Recommended)

```bash
# 1. Clone and navigate to the project
cd legal_chatbot_api

# 2. Create .env from template and set your model API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here (or equivalent)

# 3. Start all services
docker compose up -d

# 4. Open the app
# Frontend: http://localhost:3000
# Backend API: http://localhost:8080
# API Docs: http://localhost:8080/docs
```

---

### Option 2: Manual Setup

#### 1. Database

```bash
# Start PostgreSQL (if not running)
# Create the database
psql -U postgres -c "CREATE DATABASE legal_chatbot;"
```

#### 2. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and DATABASE_URL

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://localhost:8080`.  
Interactive docs at `http://localhost:8080/docs`.

#### 3. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## 🔐 Authentication

The API uses **JWT Bearer tokens**.

1. Register or login via `/api/auth/register` or `/api/auth/login`
2. Receive an `access_token` in the response
3. Include it in all subsequent requests:
   ```
   Authorization: Bearer <access_token>
   ```
4. Tokens expire after 24 hours (configurable via `ACCESS_TOKEN_EXPIRE_MINUTES`)

---

## 🧪 Quick Test

After starting the backend, you can test with curl:

```bash
# Register
curl -X POST http://localhost:8080/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'

# Save the token from the response, then:

# Create a conversation
curl -X POST http://localhost:8080/api/conversations \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Chat"}'

# Send a message (use the conversation ID from above)
curl -X POST http://localhost:8080/api/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": 1, "message": "Hello, what can you help me with?"}'
```

---

## ⚙️ Environment Variables

| Variable                     | Default                           | Description                    |
|------------------------------|-----------------------------------|--------------------------------|
| `DATABASE_URL`               | `postgresql://...localhost/...`    | PostgreSQL connection string   |
| `JWT_SECRET_KEY`             | (change me)                       | Secret key for JWT signing     |
| `JWT_ALGORITHM`              | `HS256`                           | JWT signing algorithm          |
| `ACCESS_TOKEN_EXPIRE_MINUTES`| `1440`                            | Token TTL (24 hours)           |
| `OPENAI_API_KEY`             | (required)                        | API key for generator model    |
| `OPENAI_MODEL`               | `gpt-3.5-turbo`                   | Generator model identifier     |
| `OPENAI_MAX_TOKENS`          | `2048`                            | Max response tokens            |
| `OPENAI_TEMPERATURE`         | `0.7`                             | Response creativity (0-1)      |
| `REDIS_URL`                  | `redis://localhost:6379/0`        | Redis connection string        |
| `CORS_ORIGINS`               | `http://localhost:5173,...`        | Allowed CORS origins           |
| `RATE_LIMIT_PER_MINUTE`      | `30`                              | Max API requests per minute    |

---

## 🔮 Future Improvements

- [ ] **Streaming responses** — real-time token-by-token display (endpoint ready at `/api/chat/stream`)
- [ ] **Redis caching** — cache frequent queries and conversation summaries
- [ ] **File uploads** — support document analysis (PDF, DOCX)
- [ ] **Multiple generation models** — allow users to choose GPT, Claude, local LLMs, etc.
- [ ] **Admin dashboard** — usage analytics, user management
- [ ] **WebSocket support** — real-time bidirectional communication
- [ ] **Search conversations** — full-text search across message history
- [ ] **Export conversations** — download chat history as PDF/Markdown
- [ ] **OAuth2** — Google/GitHub login
- [ ] **Kubernetes deployment** — Helm charts for production scaling
- [ ] **Unit & integration tests** — pytest for backend, Vitest for frontend
- [ ] **CI/CD pipeline** — GitHub Actions for automated testing and deployment
