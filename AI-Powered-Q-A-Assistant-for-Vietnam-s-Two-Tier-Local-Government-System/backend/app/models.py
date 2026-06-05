"""
SQLAlchemy ORM models — User, Conversation, Message.
"""

from datetime import datetime
from sqlalchemy import (
    Column, BigInteger, String, Text, DateTime, ForeignKey, CheckConstraint
)
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User id={self.id} email={self.email}>"


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(500), nullable=False, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")

    def __repr__(self):
        return f"<Conversation id={self.id} title={self.title}>"


class Message(Base):
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(BigInteger, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        CheckConstraint(role.in_(["user", "assistant", "system"]), name="ck_message_role"),
    )

    def __repr__(self):
        return f"<Message id={self.id} role={self.role}>"
