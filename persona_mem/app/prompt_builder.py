from typing import List, Dict, Optional
from app import models


# ── Token Estimation ──────────────────────────────────────────────────────────
# Rough approximation: 1 token ≈ 4 chars for English
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ── Prompt Builder ────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are PersonaMem, an intelligent AI assistant with persistent memory about the user. 
You remember past conversations, personal details, and preferences. 
Use the provided context to give personalized, helpful responses.
Be concise, clear, and adapt your tone to the user's style.
Never mention that you are reading from stored memory — just respond naturally."""


def build_profile_summary(profile: Optional[models.UserProfile]) -> str:
    if not profile:
        return ""
    parts = []
    if profile.display_name:
        parts.append(f"Name: {profile.display_name}")
    if profile.expertise_level:
        parts.append(f"Expertise: {profile.expertise_level}")
    if profile.communication_style:
        parts.append(f"Style: {profile.communication_style}")
    if profile.interests:
        interests_str = ", ".join(profile.interests[:5])
        parts.append(f"Interests: {interests_str}")
    if profile.goals:
        goals_str = "; ".join(profile.goals[:3])
        parts.append(f"Goals: {goals_str}")
    if not parts:
        return ""
    return "USER PROFILE:\n" + "\n".join(parts)


def build_memory_context(retrieved_memories: List[Dict]) -> str:
    if not retrieved_memories:
        return ""
    lines = ["RELEVANT MEMORIES:"]
    for mem in retrieved_memories:
        tag = f"[{mem['memory_type'].upper()}]"
        lines.append(f"{tag} {mem['content']}")
    return "\n".join(lines)


def build_session_summary(session_messages: List[Dict[str, str]]) -> str:
    if not session_messages:
        return ""
    lines = ["RECENT CONVERSATION:"]
    for msg in session_messages:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300]
        lines.append(f"{role_label}: {content}")
    return "\n".join(lines)


def build_prompt(
    user_query: str,
    profile: Optional[models.UserProfile],
    retrieved_memories: List[Dict],
    session_messages: List[Dict[str, str]],
    max_tokens: int = 1800,
) -> str:
    """
    Constructs the final prompt within TinyLlama's context window.
    Order: System → Profile → Long-term Memory → Session → Query
    """
    # Fixed parts
    system_part = SYSTEM_INSTRUCTION
    query_part = f"\nUser: {user_query}\nAssistant:"

    # Dynamic parts with token budgeting
    reserved = estimate_tokens(system_part) + estimate_tokens(query_part) + 50  # safety buffer
    remaining = max_tokens - reserved

    # Profile summary (small, high priority)
    profile_text = build_profile_summary(profile)
    profile_tokens = estimate_tokens(profile_text)
    if profile_tokens > remaining // 4:
        profile_text = truncate_to_tokens(profile_text, remaining // 4)
        profile_tokens = estimate_tokens(profile_text)
    remaining -= profile_tokens

    # Long-term memory (medium priority)
    memory_text = build_memory_context(retrieved_memories)
    memory_tokens = estimate_tokens(memory_text)
    memory_budget = remaining // 2
    if memory_tokens > memory_budget:
        memory_text = truncate_to_tokens(memory_text, memory_budget)
        memory_tokens = estimate_tokens(memory_text)
    remaining -= memory_tokens

    # Session context (remaining budget)
    session_text = build_session_summary(session_messages)
    if estimate_tokens(session_text) > remaining:
        session_text = truncate_to_tokens(session_text, remaining)

    # Assemble
    sections = [system_part]
    if profile_text:
        sections.append(profile_text)
    if memory_text:
        sections.append(memory_text)
    if session_text:
        sections.append(session_text)
    sections.append(query_part)

    return "\n\n".join(sections)
