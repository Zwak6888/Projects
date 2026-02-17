import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from app import models
from app.config import settings


# ── Memory Type Constants ─────────────────────────────────────────────────────
MEMORY_SESSION = "session"
MEMORY_EPISODIC = "episodic"
MEMORY_SEMANTIC = "semantic"
MEMORY_PROFILE = "profile"


# ── Importance Scoring ────────────────────────────────────────────────────────

EMOTIONAL_KEYWORDS = {
    "love", "hate", "fear", "happy", "sad", "excited", "angry", "worried",
    "important", "critical", "urgent", "emergency", "always", "never",
    "prefer", "goal", "dream", "wish", "need", "must", "remember",
}

FACTUAL_KEYWORDS = {
    "my name", "i am", "i work", "i live", "i study", "i have",
    "my job", "my career", "my hobby", "i like", "i dislike",
    "i want", "i need", "my goal", "my project",
}


def compute_importance_score(content: str, reinforcement_count: int = 1) -> float:
    score = 0.3  # baseline
    lower = content.lower()

    # Emotional weight
    emotional_hits = sum(1 for kw in EMOTIONAL_KEYWORDS if kw in lower)
    score += min(emotional_hits * 0.05, 0.25)

    # Factual/personal weight
    factual_hits = sum(1 for kw in FACTUAL_KEYWORDS if kw in lower)
    score += min(factual_hits * 0.08, 0.24)

    # Length bonus (more detailed = potentially more important)
    if len(content) > 100:
        score += 0.05
    if len(content) > 300:
        score += 0.05

    # Frequency reinforcement
    score += min((reinforcement_count - 1) * 0.02, 0.1)

    return min(round(score, 4), 1.0)


def compute_recency_score(timestamp: datetime) -> float:
    age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
    if age_hours < 1:
        return 1.0
    elif age_hours < 24:
        return 0.85
    elif age_hours < 168:  # 1 week
        return 0.65
    elif age_hours < 720:  # 1 month
        return 0.45
    elif age_hours < 8760:  # 1 year
        return 0.25
    return 0.1


# ── Fact Extraction ───────────────────────────────────────────────────────────

def extract_facts_from_conversation(messages: List[Dict[str, str]]) -> List[str]:
    facts = []
    for msg in messages:
        if msg["role"] != "user":
            continue
        content = msg["content"]
        lower = content.lower()

        # Identity facts
        for pattern in [
            r"my name is (.+?)[\.\,\!]",
            r"i(?:'m| am) (.+?)[\.\,\!]",
            r"i work (?:at|for|as) (.+?)[\.\,\!]",
            r"i live in (.+?)[\.\,\!]",
            r"i(?:'m| am) studying (.+?)[\.\,\!]",
            r"i(?:'m| am) learning (.+?)[\.\,\!]",
            r"my (?:goal|dream|aim) is (.+?)[\.\,\!]",
            r"i (?:love|hate|enjoy|prefer) (.+?)[\.\,\!]",
            r"i(?:'ve| have) (?:been|worked) (.+?)[\.\,\!]",
        ]:
            matches = re.findall(pattern, lower)
            for m in matches:
                if len(m.strip()) > 3:
                    facts.append(content.strip())
                    break

        # Preferences
        if any(kw in lower for kw in ["i prefer", "i like", "i dislike", "i want", "i need"]):
            if len(content) < 500:
                facts.append(content.strip())

    # Deduplicate
    seen = set()
    unique_facts = []
    for f in facts:
        key = f[:80].lower()
        if key not in seen:
            seen.add(key)
            unique_facts.append(f)

    return unique_facts[:10]  # limit to 10 facts per conversation


# ── Memory CRUD ───────────────────────────────────────────────────────────────

def create_memory(
    db: Session,
    user_id: str,
    memory_type: str,
    content: str,
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    faiss_index_id: Optional[int] = None,
) -> models.Memory:
    importance = compute_importance_score(content)
    mem = models.Memory(
        user_id=user_id,
        memory_type=memory_type,
        content=content,
        importance_score=importance,
        session_id=session_id,
        faiss_index_id=faiss_index_id,
        metadata_json=metadata or {},
    )
    db.add(mem)
    db.commit()
    db.refresh(mem)
    return mem


def get_memories_by_user(
    db: Session,
    user_id: str,
    memory_type: Optional[str] = None,
    limit: int = 100,
) -> List[models.Memory]:
    q = db.query(models.Memory).filter(models.Memory.user_id == user_id)
    if memory_type:
        q = q.filter(models.Memory.memory_type == memory_type)
    return q.order_by(models.Memory.timestamp.desc()).limit(limit).all()


def get_memory_by_id(db: Session, memory_id: str, user_id: str) -> Optional[models.Memory]:
    return db.query(models.Memory).filter(
        models.Memory.id == memory_id,
        models.Memory.user_id == user_id
    ).first()


def update_memory_access(db: Session, memory: models.Memory) -> None:
    memory.last_accessed = datetime.utcnow()
    memory.reinforcement_count += 1
    memory.importance_score = compute_importance_score(memory.content, memory.reinforcement_count)
    db.commit()


def delete_memory(db: Session, memory_id: str, user_id: str) -> bool:
    mem = get_memory_by_id(db, memory_id, user_id)
    if not mem:
        return False
    db.delete(mem)
    db.commit()
    return True


def upsert_memory_by_content(
    db: Session,
    user_id: str,
    memory_type: str,
    content: str,
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> models.Memory:
    """Update reinforcement if very similar memory exists, else create new."""
    existing = (
        db.query(models.Memory)
        .filter(
            models.Memory.user_id == user_id,
            models.Memory.memory_type == memory_type,
        )
        .all()
    )
    content_lower = content.lower()[:120]
    for mem in existing:
        if mem.content.lower()[:120] == content_lower:
            update_memory_access(db, mem)
            return mem

    return create_memory(db, user_id, memory_type, content, session_id, metadata)


# ── Profile Memory ────────────────────────────────────────────────────────────

def get_or_create_profile(db: Session, user_id: str) -> models.UserProfile:
    profile = db.query(models.UserProfile).filter(models.UserProfile.user_id == user_id).first()
    if not profile:
        profile = models.UserProfile(user_id=user_id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


def update_profile(db: Session, user_id: str, updates: Dict[str, Any]) -> models.UserProfile:
    profile = get_or_create_profile(db, user_id)
    allowed_fields = {
        "display_name", "expertise_level", "goals", "interests",
        "preferred_language", "personality_tags", "communication_style", "timezone"
    }
    for k, v in updates.items():
        if k in allowed_fields:
            setattr(profile, k, v)
    profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(profile)
    return profile


def auto_update_profile_from_facts(db: Session, user_id: str, facts: List[str]) -> None:
    """Heuristically extract profile info from extracted facts."""
    profile = get_or_create_profile(db, user_id)
    interests = list(profile.interests or [])
    goals = list(profile.goals or [])

    for fact in facts:
        lower = fact.lower()
        # Update expertise from hints
        if any(kw in lower for kw in ["expert", "senior", "professional", "years of experience"]):
            profile.expertise_level = "expert"
        elif any(kw in lower for kw in ["beginner", "just started", "new to", "learning"]):
            profile.expertise_level = "beginner"

        # Collect goals
        for pattern in [r"my goal is (.+?)[\.\!]", r"i want to (.+?)[\.\!]", r"i aim to (.+?)[\.\!]"]:
            matches = re.findall(pattern, lower)
            for m in matches:
                g = m.strip().capitalize()[:128]
                if g and g not in goals:
                    goals.append(g)

        # Collect interests
        for pattern in [r"i (?:love|enjoy|like) (.+?)[\.\!,]"]:
            matches = re.findall(pattern, lower)
            for m in matches:
                i = m.strip().capitalize()[:64]
                if i and i not in interests:
                    interests.append(i)

    profile.goals = goals[:20]
    profile.interests = interests[:20]
    db.commit()


# ── Session Memory Management ─────────────────────────────────────────────────

def get_session_context(db: Session, session_id: str, window: int = 6) -> List[Dict[str, str]]:
    session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
    if not session:
        return []
    messages = session.messages[-window:]
    return [{"role": m.role, "content": m.content} for m in messages]


# ── Post-conversation Processing ──────────────────────────────────────────────

def process_conversation_memories(
    db: Session,
    user_id: str,
    session_id: str,
    messages: List[Dict[str, str]],
) -> None:
    """Extract and store episodic + semantic memories after a conversation turn."""
    facts = extract_facts_from_conversation(messages)

    for fact in facts:
        # Store as episodic memory
        upsert_memory_by_content(
            db, user_id, MEMORY_EPISODIC, fact,
            session_id=session_id,
            metadata={"source": "conversation", "auto_extracted": True},
        )
        # Also store as semantic memory (will be embedded separately)
        upsert_memory_by_content(
            db, user_id, MEMORY_SEMANTIC, fact,
            session_id=session_id,
            metadata={"source": "conversation", "auto_extracted": True, "needs_embedding": True},
        )

    # Update profile
    auto_update_profile_from_facts(db, user_id, facts)
