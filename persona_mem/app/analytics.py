from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from app import models


def get_user_metrics(db: Session, user_id: str) -> Dict[str, Any]:
    # Total sessions
    total_sessions = db.query(func.count(models.ChatSession.id)).filter(
        models.ChatSession.user_id == user_id
    ).scalar() or 0

    # Total messages
    total_messages = (
        db.query(func.count(models.ChatMessage.id))
        .join(models.ChatSession, models.ChatMessage.session_id == models.ChatSession.id)
        .filter(models.ChatSession.user_id == user_id)
        .scalar() or 0
    )

    # Messages in last 7 days
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_messages = (
        db.query(func.count(models.ChatMessage.id))
        .join(models.ChatSession, models.ChatMessage.session_id == models.ChatSession.id)
        .filter(
            models.ChatSession.user_id == user_id,
            models.ChatMessage.timestamp >= week_ago,
        )
        .scalar() or 0
    )

    # Memory stats
    memory_counts = {}
    for mtype in ["session", "episodic", "semantic", "profile"]:
        count = db.query(func.count(models.Memory.id)).filter(
            models.Memory.user_id == user_id,
            models.Memory.memory_type == mtype,
        ).scalar() or 0
        memory_counts[mtype] = count

    total_memories = sum(memory_counts.values())

    # Average importance score
    avg_importance = db.query(func.avg(models.Memory.importance_score)).filter(
        models.Memory.user_id == user_id
    ).scalar()
    avg_importance = round(float(avg_importance), 4) if avg_importance else 0.0

    # Most accessed memories
    top_memories = (
        db.query(models.Memory)
        .filter(models.Memory.user_id == user_id)
        .order_by(models.Memory.reinforcement_count.desc())
        .limit(5)
        .all()
    )

    # Profile completeness
    profile = db.query(models.UserProfile).filter(
        models.UserProfile.user_id == user_id
    ).first()

    profile_completeness = 0
    if profile:
        fields = [
            profile.display_name,
            profile.expertise_level != "intermediate",
            bool(profile.goals),
            bool(profile.interests),
            profile.communication_style != "conversational",
        ]
        profile_completeness = int((sum(1 for f in fields if f) / len(fields)) * 100)

    # Sessions per day (last 30 days)
    month_ago = datetime.utcnow() - timedelta(days=30)
    sessions_per_day = []
    for i in range(30):
        day_start = month_ago + timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        count = db.query(func.count(models.ChatSession.id)).filter(
            models.ChatSession.user_id == user_id,
            models.ChatSession.created_at >= day_start,
            models.ChatSession.created_at < day_end,
        ).scalar() or 0
        sessions_per_day.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "count": count,
        })

    return {
        "overview": {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "recent_messages_7d": recent_messages,
            "total_memories": total_memories,
            "avg_importance_score": avg_importance,
            "profile_completeness_pct": profile_completeness,
        },
        "memory_breakdown": memory_counts,
        "top_memories": [
            {
                "id": m.id,
                "content": m.content[:100] + ("..." if len(m.content) > 100 else ""),
                "memory_type": m.memory_type,
                "importance_score": m.importance_score,
                "reinforcement_count": m.reinforcement_count,
            }
            for m in top_memories
        ],
        "activity": {
            "sessions_per_day_30d": sessions_per_day,
        },
    }
