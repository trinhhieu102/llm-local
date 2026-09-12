"""
Cấu hình tập trung cho hệ thống Local LLM.
Tự động nạp các tham số từ biến môi trường hoặc file .env.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Tự động nạp file .env từ thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Class cấu hình bất biến (Immutable Settings) đảm bảo thread-safe và clean code."""
    
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:3b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    
    # RAG Settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "3"))
    
    # Đường dẫn lưu trữ vector database
    chroma_db_dir: Path = BASE_DIR / "chroma_data"
    data_dir: Path = BASE_DIR / "data"


# Singleton instance
settings = Settings()
