import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, Boolean, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from app.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="user", cascade="all, delete-orphan")
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(256), default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan",
                            order_by="ChatMessage.timestamp")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=gen_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class Memory(Base):
    __tablename__ = "memories"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    memory_type = Column(String(32), nullable=False, index=True)
    # Types: "session", "episodic", "semantic", "profile"
    content = Column(Text, nullable=False)
    importance_score = Column(Float, default=0.5)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    reinforcement_count = Column(Integer, default=1)
    timestamp = Column(DateTime, default=datetime.utcnow)
    faiss_index_id = Column(Integer, nullable=True)  # ID in FAISS index
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=True)
    metadata_json = Column(JSON, default=dict)  # extra fields: tags, source, etc.

    user = relationship("User", back_populates="memories")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)
    display_name = Column(String(128), default="")
    expertise_level = Column(String(32), default="intermediate")  # beginner, intermediate, expert
    goals = Column(JSON, default=list)
    interests = Column(JSON, default=list)
    preferred_language = Column(String(16), default="en")
    personality_tags = Column(JSON, default=list)
    communication_style = Column(String(32), default="conversational")
    timezone = Column(String(64), default="UTC")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="profile")
