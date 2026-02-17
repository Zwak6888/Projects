import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.config import settings
from app import models
from app.memory_service import compute_recency_score, get_memories_by_user

# ── Embedding Model (singleton) ───────────────────────────────────────────────
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def embed_text(text: str) -> np.ndarray:
    model = get_embedding_model()
    embedding = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return embedding[0].astype(np.float32)


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


# ── FAISS Index Per-User ──────────────────────────────────────────────────────

def _index_path(user_id: str) -> str:
    return os.path.join(settings.FAISS_INDEX_DIR, f"user_{user_id}.index")


def _meta_path(user_id: str) -> str:
    return os.path.join(settings.FAISS_INDEX_DIR, f"user_{user_id}.meta")


def load_faiss_index(user_id: str) -> Tuple[Optional[faiss.Index], List[str]]:
    """Returns (index, list_of_memory_ids) or (None, []) if not found."""
    ipath = _index_path(user_id)
    mpath = _meta_path(user_id)
    if not os.path.exists(ipath) or not os.path.exists(mpath):
        return None, []
    index = faiss.read_index(ipath)
    with open(mpath, "rb") as f:
        memory_ids = pickle.load(f)
    return index, memory_ids


def save_faiss_index(user_id: str, index: faiss.Index, memory_ids: List[str]) -> None:
    faiss.write_index(index, _index_path(user_id))
    with open(_meta_path(user_id), "wb") as f:
        pickle.dump(memory_ids, f)


def add_memory_to_index(user_id: str, memory_id: str, content: str) -> int:
    """Add a single memory to FAISS. Returns the assigned FAISS ID."""
    index, memory_ids = load_faiss_index(user_id)
    embedding = embed_text(content).reshape(1, -1)
    dim = embedding.shape[1]

    if index is None:
        index = faiss.IndexFlatIP(dim)  # Inner Product (cosine after normalization)
        memory_ids = []

    index.add(embedding)
    faiss_id = len(memory_ids)
    memory_ids.append(memory_id)
    save_faiss_index(user_id, index, memory_ids)
    return faiss_id


def rebuild_user_index(db: Session, user_id: str) -> None:
    """Rebuild FAISS index from all semantic memories in DB."""
    memories = get_memories_by_user(db, user_id, memory_type="semantic")
    if not memories:
        return

    texts = [m.content for m in memories]
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    memory_ids = [m.id for m in memories]

    # Update faiss_index_id in DB
    for i, mem in enumerate(memories):
        mem.faiss_index_id = i
    db.commit()

    save_faiss_index(user_id, index, memory_ids)


# ── Retrieval & Scoring ───────────────────────────────────────────────────────

def retrieve_relevant_memories(
    db: Session,
    user_id: str,
    query: str,
    top_k: int = None,
) -> List[Dict]:
    if top_k is None:
        top_k = settings.TOP_K_MEMORIES

    index, memory_ids = load_faiss_index(user_id)
    if index is None or index.ntotal == 0:
        return []

    query_vec = embed_text(query).reshape(1, -1)
    k = min(top_k * 3, index.ntotal)  # retrieve more, then re-rank
    distances, indices = index.search(query_vec, k)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(memory_ids):
            continue
        mem_id = memory_ids[idx]
        mem = db.query(models.Memory).filter(
            models.Memory.id == mem_id,
            models.Memory.user_id == user_id,
        ).first()
        if not mem:
            continue

        semantic_sim = float(dist)  # already normalized cosine similarity

        recency = compute_recency_score(mem.timestamp)
        importance = float(mem.importance_score)
        frequency = min(mem.reinforcement_count / 20.0, 1.0)

        # Personal relevance: profile or episodic memories get bonus
        personal = 0.8 if mem.memory_type in ("profile", "episodic") else 0.5

        composite = (
            settings.WEIGHT_SEMANTIC * semantic_sim
            + settings.WEIGHT_RECENCY * recency
            + settings.WEIGHT_IMPORTANCE * importance
            + settings.WEIGHT_FREQUENCY * frequency
            + settings.WEIGHT_PERSONAL * personal
        )

        candidates.append({
            "id": mem.id,
            "content": mem.content,
            "memory_type": mem.memory_type,
            "importance_score": importance,
            "semantic_similarity": round(semantic_sim, 4),
            "recency_score": round(recency, 4),
            "composite_score": round(composite, 4),
            "timestamp": mem.timestamp.isoformat(),
            "reinforcement_count": mem.reinforcement_count,
        })

    candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    return candidates[:top_k]


def embed_and_store_memory(db: Session, user_id: str, mem: models.Memory) -> None:
    """Embed a single memory and store in FAISS + update DB."""
    faiss_id = add_memory_to_index(user_id, mem.id, mem.content)
    mem.faiss_index_id = faiss_id
    db.commit()
