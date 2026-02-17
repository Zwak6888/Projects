import json
import os
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, init_db
from app.auth import (
    SignupRequest, LoginRequest, TokenResponse,
    signup_user, login_user, get_current_user
)
from app import models
from app.chat_service import (
    ChatRequest, process_chat_stream,
    get_user_sessions, get_session_detail
)
from app.memory_service import (
    get_memories_by_user, delete_memory,
    update_profile, get_or_create_profile,
    upsert_memory_by_content
)
from app.retrieval import retrieve_relevant_memories, rebuild_user_index
from app.analytics import get_user_metrics


# ──────────────────────────────────────────────────────────────────────────────
# App Initialization
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
frontend_dir = os.path.join(BASE_DIR, "..", "frontend")

if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.on_event("startup")
def startup():
    init_db()


# ──────────────────────────────────────────────────────────────────────────────
# Auth Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/signup", tags=["auth"])
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    user = signup_user(request, db)
    return {
        "message": "User created successfully",
        "user_id": user.id,
        "username": user.username
    }


@app.post("/login", response_model=TokenResponse, tags=["auth"])
def login(request: LoginRequest, db: Session = Depends(get_db)):
    return login_user(request, db)


@app.get("/me", tags=["auth"])
def me(current_user: models.User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "created_at": current_user.created_at.isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Chat Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/chat", tags=["chat"])
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    async def generate():
        async for chunk in process_chat_stream(db, current_user, request):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/sessions", tags=["chat"])
def list_sessions(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_user_sessions(db, current_user.id)


@app.get("/session/{session_id}", tags=["chat"])
def get_session(
    session_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    detail = get_session_detail(db, session_id, current_user.id)
    if not detail:
        raise HTTPException(status_code=404, detail="Session not found")
    return detail


# ──────────────────────────────────────────────────────────────────────────────
# Memory Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/profile", tags=["memory"])
def get_profile(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = get_or_create_profile(db, current_user.id)
    return profile


class ProfileUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    expertise_level: Optional[str] = None
    goals: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    preferred_language: Optional[str] = None
    personality_tags: Optional[List[str]] = None
    communication_style: Optional[str] = None
    timezone: Optional[str] = None


@app.put("/profile", tags=["memory"])
def update_user_profile(
    request: ProfileUpdateRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    updates = {k: v for k, v in request.dict().items() if v is not None}
    profile = update_profile(db, current_user.id, updates)
    return {"message": "Profile updated", "profile_id": profile.id}


@app.get("/memories", tags=["memory"])
def list_memories(
    memory_type: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    mems = get_memories_by_user(db, current_user.id, memory_type, limit)
    return mems


class MemoryCreateRequest(BaseModel):
    content: str
    memory_type: str = "semantic"
    metadata: Optional[dict] = None


@app.put("/memory", tags=["memory"])
def upsert_memory(
    request: MemoryCreateRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    mem = upsert_memory_by_content(
        db, current_user.id, request.memory_type,
        request.content, metadata=request.metadata
    )
    return {"message": "Memory stored", "memory_id": mem.id}


@app.delete("/memory/{memory_id}", tags=["memory"])
def remove_memory(
    memory_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not delete_memory(db, memory_id, current_user.id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"message": "Memory deleted"}


@app.post("/memory/rebuild-index", tags=["memory"])
def rebuild_memory_index(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rebuild_user_index(db, current_user.id)
    return {"message": "FAISS index rebuilt"}


@app.get("/memory/search", tags=["memory"])
def search_memories(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, le=20),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return retrieve_relevant_memories(db, current_user.id, q, top_k)


# ──────────────────────────────────────────────────────────────────────────────
# Analytics
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/metrics", tags=["analytics"])
def metrics(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_user_metrics(db, current_user.id)


# ──────────────────────────────────────────────────────────────────────────────
# Frontend Routes (FIXED UTF-8)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    try:
        with open(os.path.join(frontend_dir, "login.html"), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>PersonaMem API is running</h1>"


@app.get("/app", response_class=HTMLResponse)
def chat_app():
    try:
        with open(os.path.join(frontend_dir, "index.html"), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    try:
        with open(os.path.join(frontend_dir, "dashboard.html"), encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")


# ──────────────────────────────────────────────────────────────────────────────
# Run Server
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001)
