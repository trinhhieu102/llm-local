"""
FastAPI Server cung cấp REST API cho Local LLM và RAG Engine.
Tương thích hoàn toàn cho việc tích hợp Web Chatbot, Mobile App hoặc Third-party services.
"""

import sys
from pathlib import Path
from typing import List, Optional

# UTF-8 encoding support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.llm_client import OllamaClient
from src.rag_engine import RAGEngine

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Vietnamese Local LLM & RAG API",
    description="High-performance, 100% offline LLM & RAG API powered by Ollama, Qwen 2.5 and ChromaDB.",
    version="1.0.0",
)

# Cấu hình CORS để mọi Frontend (React, Vue, Next.js, HTML) có thể gọi vào
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo RAG Engine và Ollama Client
ollama_client = OllamaClient()
rag_engine = RAGEngine()


# --- Pydantic Data Models ---
class ChatMessage(BaseModel):
    role: str = Field(..., example="user")
    content: str = Field(..., example="Xin chào AI!")


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: bool = False


class RAGQueryRequest(BaseModel):
    query: str = Field(..., example="Tốc độ sinh chữ của Qwen 2.5 là bao nhiêu?")
    top_k: Optional[int] = 3


# --- API Endpoints ---
@app.get("/health", tags=["System"])
def health_check():
    """Kiểm tra trạng thái kết nối tới GPU và Ollama engine."""
    is_ready = ollama_client.is_service_ready()
    return {
        "status": "online" if is_ready else "ollama_offline",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "gpu_accelerated": True,
    }


@app.post("/api/chat", tags=["LLM Chat"])
def chat(request: ChatRequest):
    """
    Endpoint trò chuyện trực tiếp (hỗ trợ cả JSON thông thường và Server-Sent Events streaming).
    """
    raw_messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        def event_generator():
            for chunk in ollama_client.stream_chat(raw_messages, model=request.model):
                yield chunk
        return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")

    # Non-stream
    reply = ""
    for chunk in ollama_client.stream_chat(raw_messages, model=request.model):
        reply += chunk
    return {"message": {"role": "assistant", "content": reply}}


@app.post("/api/rag/query", tags=["RAG"])
def query_rag(request: RAGQueryRequest):
    """
    Hỏi đáp dựa trên tài liệu nội bộ đã được băm vector trong ChromaDB.
    """
    try:
        stream_gen, context_items = rag_engine.query(request.query)
        full_reply = "".join(list(stream_gen))
        
        citations = [
            {
                "source": item["metadata"].get("source", "Unknown"),
                "chunk_index": item["metadata"].get("chunk_index", 0),
                "snippet": item["content"][:200]
            }
            for item in context_items
        ]
        
        return {
            "query": request.query,
            "answer": full_reply,
            "citations": citations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/ingest-file", tags=["RAG"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload file tài liệu mới (.txt, .md, .pdf) để nạp tức thì vào cơ sở dữ liệu vector.
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    target_path = settings.data_dir / file.filename

    with open(target_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        num_chunks = rag_engine.ingest_document(target_path)
        return {
            "filename": file.filename,
            "chunks_created": num_chunks,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi nạp file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Khởi chạy FastAPI Server tại: http://localhost:8000")
    print("📖 Swagger API Docs xem tại: http://localhost:8000/docs\n")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
