"""
Pydantic schemas for request / response validation.
"""

from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


# ── Auth ──────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str | None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Conversation ──────────────────────────────────────────────

class ConversationCreate(BaseModel):
    title: str = "New Conversation"


class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]


# ── Message ───────────────────────────────────────────────────

class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    messages: list[MessageResponse]


# ── Chat ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    conversation_id: int
    message: str = Field(..., min_length=1, max_length=10000)


class ChatResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
