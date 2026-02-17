import json
import uuid
from datetime import datetime
from typing import AsyncIterator, List, Optional, Dict
import httpx
from sqlalchemy.orm import Session
from fastapi import HTTPException
from pydantic import BaseModel

from app.config import settings
from app import models
from app.memory_service import (
    get_session_context,
    process_conversation_memories,
    MEMORY_SESSION,
)
from app.retrieval import retrieve_relevant_memories, embed_and_store_memory
from app.prompt_builder import build_prompt
from app.memory_service import get_or_create_profile, upsert_memory_by_content


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    content: str


# ── Session Management ────────────────────────────────────────────────────────

def get_or_create_session(db: Session, user_id: str, session_id: Optional[str]) -> models.ChatSession:
    if session_id:
        session = db.query(models.ChatSession).filter(
            models.ChatSession.id == session_id,
            models.ChatSession.user_id == user_id,
        ).first()
        if session:
            return session

    # Create new session
    session = models.ChatSession(
        user_id=user_id,
        title="New Conversation",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def save_message(db: Session, session_id: str, role: str, content: str) -> models.ChatMessage:
    msg = models.ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def update_session_title(db: Session, session: models.ChatSession, first_message: str) -> None:
    if session.title == "New Conversation":
        title = first_message[:60].strip()
        if len(first_message) > 60:
            title += "..."
        session.title = title
        session.updated_at = datetime.utcnow()
        db.commit()


def get_user_sessions(db: Session, user_id: str) -> List[Dict]:
    sessions = (
        db.query(models.ChatSession)
        .filter(models.ChatSession.user_id == user_id)
        .order_by(models.ChatSession.updated_at.desc())
        .limit(50)
        .all()
    )
    result = []
    for s in sessions:
        msg_count = len(s.messages)
        result.append({
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat() if s.updated_at else s.created_at.isoformat(),
            "message_count": msg_count,
        })
    return result


def get_session_detail(db: Session, session_id: str, user_id: str) -> Optional[Dict]:
    session = db.query(models.ChatSession).filter(
        models.ChatSession.id == session_id,
        models.ChatSession.user_id == user_id,
    ).first()
    if not session:
        return None
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in session.messages
        ],
    }


# ── Ollama Streaming ──────────────────────────────────────────────────────────

async def stream_ollama(prompt: str) -> AsyncIterator[str]:
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["\nUser:", "\nHuman:"],
        },
    }
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    yield f"[Error: Ollama returned {response.status_code}]"
                    return
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError:
            yield "[Error: Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434]"
        except Exception as e:
            yield f"[Error: {str(e)}]"


async def call_ollama_non_stream(prompt: str) -> str:
    """Non-streaming call for internal use (memory extraction)."""
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 512, "temperature": 0.3},
    }
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "")
    except Exception:
        pass
    return ""


# ── Main Chat Processing ──────────────────────────────────────────────────────

async def process_chat_stream(
    db: Session,
    user: models.User,
    request: ChatRequest,
) -> AsyncIterator[str]:
    """Full chat pipeline with memory augmentation, returns streaming tokens."""

    # 1. Session
    session = get_or_create_session(db, user.id, request.session_id)
    update_session_title(db, session, request.message)

    # 2. Save user message
    save_message(db, session.id, "user", request.message)

    # 3. Session context (recent messages)
    session_msgs = get_session_context(db, session.id, window=settings.SESSION_WINDOW)

    # 4. Retrieve relevant memories
    retrieved = retrieve_relevant_memories(db, user.id, request.message)

    # 5. User profile
    profile = get_or_create_profile(db, user.id)

    # 6. Build prompt
    prompt = build_prompt(
        user_query=request.message,
        profile=profile,
        retrieved_memories=retrieved,
        session_messages=session_msgs[:-1],  # exclude the message we just saved
        max_tokens=settings.MAX_CONTEXT_TOKENS,
    )

    # 7. Stream from Ollama, accumulate full response
    full_response = []

    # Yield session_id as first chunk (JSON metadata)
    yield json.dumps({"type": "meta", "session_id": session.id}) + "\n"

    async for token in stream_ollama(prompt):
        full_response.append(token)
        yield json.dumps({"type": "token", "content": token}) + "\n"

    complete_response = "".join(full_response)

    # 8. Save assistant response
    saved_msg = save_message(db, session.id, "assistant", complete_response)
    yield json.dumps({"type": "done", "message_id": saved_msg.id}) + "\n"

    # 9. Post-conversation memory processing (fire-and-forget style)
    all_msgs = [{"role": "user", "content": request.message},
                {"role": "assistant", "content": complete_response}]
    process_conversation_memories(db, user.id, session.id, all_msgs)

    # 10. Embed new semantic memories
    new_semantic_mems = (
        db.query(models.Memory)
        .filter(
            models.Memory.user_id == user.id,
            models.Memory.memory_type == "semantic",
            models.Memory.faiss_index_id.is_(None),
        )
        .all()
    )
    for mem in new_semantic_mems:
        try:
            embed_and_store_memory(db, user.id, mem)
        except Exception:
            pass  # Don't fail the chat due to indexing errors
