"""
Conversation router — list, create, delete conversations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import ConversationCreate, ConversationResponse, ConversationListResponse
from app.auth import get_current_user
from app.models import User
from app import crud

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("", response_model=ConversationListResponse)
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all conversations for the current user, newest first."""
    conversations = crud.get_conversations_by_user(db, current_user.id)
    return ConversationListResponse(conversations=conversations)


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    req: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new conversation."""
    conversation = crud.create_conversation(db, user_id=current_user.id, title=req.title)
    return conversation


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a single conversation by ID."""
    conversation = crud.get_conversation_by_id(db, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a conversation and all its messages."""
    conversation = crud.get_conversation_by_id(db, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    crud.delete_conversation(db, conversation_id)
