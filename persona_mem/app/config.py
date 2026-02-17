import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    PROJECT_NAME: str = "PersonaMem"
    VERSION: str = "1.0.0"

    # JWT
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "persona_mem_super_secret_key_change_in_prod_2024"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/persona_mem.db"

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    # ✅ CHANGE MODEL TO PHI
    OLLAMA_MODEL: str = os.getenv(
        "OLLAMA_MODEL",
        "phi"
    )

    # Memory
    FAISS_INDEX_DIR: str = str(BASE_DIR / "faiss_indexes")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K_MEMORIES: int = 5
    MAX_CONTEXT_TOKENS: int = 1800

    # Memory scoring weights
    WEIGHT_SEMANTIC: float = 0.4
    WEIGHT_RECENCY: float = 0.2
    WEIGHT_IMPORTANCE: float = 0.2
    WEIGHT_FREQUENCY: float = 0.1
    WEIGHT_PERSONAL: float = 0.1

    # Session memory window
    SESSION_WINDOW: int = 6

settings = Settings()

# Ensure FAISS index dir exists
os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
