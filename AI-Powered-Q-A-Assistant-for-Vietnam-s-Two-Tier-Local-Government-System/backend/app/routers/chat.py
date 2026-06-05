"""
Chat router — send messages, receive AI responses, get message history.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from loguru import logger

from app.database import get_db
from app.schemas import ChatRequest, ChatResponse, MessageListResponse, MessageResponse
from app.auth import get_current_user
from app.models import User
from app import crud
from app.rag_service import generate_response

router = APIRouter(tags=["Chat"])


@router.get("/messages/{conversation_id}", response_model=MessageListResponse)
def get_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all messages in a conversation."""
    # Verify the conversation belongs to the current user
    conversation = crud.get_conversation_by_id(db, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    messages = crud.get_messages_by_conversation(db, conversation_id)
    return MessageListResponse(messages=messages)


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Send a message and receive an AI response.

    Flow:
    1. Validate conversation belongs to user
    2. Save user message to database
    3. Fetch conversation history
    4. Call OpenAI API with context
    5. Save assistant response to database
    6. Auto-generate conversation title if it's the first message
    7. Return both messages
    """
    # 1. Verify conversation ownership
    conversation = crud.get_conversation_by_id(db, req.conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    # 2. Save user message
    user_msg = crud.create_message(db, conversation_id=req.conversation_id, role="user", content=req.message)
    logger.info(f"User message saved: conversation_id={req.conversation_id}")

    # 3. Fetch conversation history (excluding the message we just added — it's in context already)
    history = crud.get_messages_by_conversation(db, req.conversation_id)

    # 4. Call RAG Pipeline
    try:
        ai_content = generate_response(req.message)
    except Exception as e:
        logger.error(f"RAG pipeline call failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # 5. Save assistant response
    assistant_msg = crud.create_message(db, conversation_id=req.conversation_id, role="assistant", content=ai_content)
    logger.info(f"Assistant message saved: conversation_id={req.conversation_id}")

    # 6. Auto-title the conversation based on first message
    if conversation.title == "New Conversation":
        title = req.message[:80] + ("..." if len(req.message) > 80 else "")
        crud.update_conversation_title(db, req.conversation_id, title)

    return ChatResponse(
        user_message=MessageResponse.model_validate(user_msg),
        assistant_message=MessageResponse.model_validate(assistant_msg),
    )


