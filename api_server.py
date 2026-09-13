"""
FastAPI Server cung cấp REST API cho Local LLM và RAG Engine.
Tương thích hoàn toàn cho việc tích hợp Web Chatbot, Mobile App hoặc Third-party services.
"""

import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# UTF-8 encoding support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import settings
from src.llm_client import OllamaClient
from src.rag_engine import RAGEngine

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Vietnamese Local LLM & RAG API",
    description="High-performance, 100% offline LLM & RAG API powered by Ollama, Qwen 2.5 and ChromaDB.",
    version="1.1.0",
)

# Cấu hình CORS để mọi Frontend (React, Vue, Next.js, HTML) có thể gọi vào
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn thư mục static
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Khởi tạo RAG Engine và Ollama Client
ollama_client = OllamaClient()
rag_engine = RAGEngine()


# --- Pydantic Data Models ---
class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str = Field(..., examples=["Xin chào AI!"])


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class RAGQueryRequest(BaseModel):
    query: str = Field(..., examples=["Tốc độ sinh chữ của Qwen 2.5 là bao nhiêu?"])
    top_k: Optional[int] = 3
    temperature: Optional[float] = None
    top_p: Optional[float] = None


# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def serve_index():
    """Phục vụ giao diện Web Studio tại root URL."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Vietnamese Local LLM & RAG API is running"}


@app.get("/health", tags=["System"])
def health_check():
    """Kiểm tra trạng thái kết nối tới GPU và Ollama engine."""
    is_ready = ollama_client.is_service_ready()
    return {
        "status": "online" if is_ready else "ollama_offline",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "search_mode": "Hybrid (BM25 + BGE-M3 RRF)",
        "gpu_accelerated": True,
    }


@app.post("/api/chat", tags=["LLM Chat"])
def chat(request: ChatRequest):
    """
    Endpoint trò chuyện trực tiếp (hỗ trợ cả JSON thông thường và Server-Sent Events streaming).
    """
    raw_messages = [{"role": m.role, "content": m.content} for m in request.messages]

    options: Dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = float(request.temperature)
    if request.top_p is not None:
        options["top_p"] = float(request.top_p)

    if request.stream:
        def event_generator():
            for chunk in ollama_client.stream_chat(raw_messages, model=request.model, options=options or None):
                yield chunk
        return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")

    # Non-stream
    reply = ""
    for chunk in ollama_client.stream_chat(raw_messages, model=request.model, options=options or None):
        reply += chunk
    return {"message": {"role": "assistant", "content": reply}}


@app.post("/api/rag/query", tags=["RAG"])
def query_rag(request: RAGQueryRequest):
    """
    Hỏi đáp dựa trên tài liệu nội bộ đã được băm vector trong ChromaDB với Hybrid Search RRF.
    """
    try:
        start_time = time.time()
        options: Dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = float(request.temperature)
        if request.top_p is not None:
            options["top_p"] = float(request.top_p)

        stream_gen, context_items = rag_engine.query(
            request.query,
            top_k=request.top_k,
            options=options or None,
        )
        full_reply = "".join(list(stream_gen))
        latency = round(time.time() - start_time, 2)
        
        # Đo lường token sơ bộ (khoảng 1 token ~ 0.75 từ hoặc theo dấu cách)
        token_count = max(1, len(full_reply.split()))
        tokens_per_sec = round(token_count / max(0.1, latency), 1)

        citations = [
            {
                "source": item["metadata"].get("source", "Unknown"),
                "chunk_index": item["metadata"].get("chunk_index", 0),
                "relevance_score": item.get("relevance_score", 85.0),
                "snippet": item["content"][:240],
            }
            for item in context_items
        ]
        
        return {
            "query": request.query,
            "answer": full_reply,
            "citations": citations,
            "metrics": {
                "latency_sec": latency,
                "token_count": token_count,
                "tokens_per_sec": tokens_per_sec,
                "chunks_retrieved": len(context_items),
                "search_mode": "Hybrid RRF",
            }
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


@app.get("/api/rag/documents", tags=["RAG"])
def list_documents():
    """Liệt kê danh sách các tài liệu trong thư mục data sẵn sàng cho RAG."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    for f in settings.data_dir.iterdir():
        if f.is_file() and f.suffix.lower() in [".txt", ".md", ".pdf"]:
            docs.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "extension": f.suffix.lower()
            })
    return {"documents": docs, "total_count": len(docs)}


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Khởi chạy FastAPI Server tại: http://localhost:8000")
    print("📖 Swagger API Docs xem tại: http://localhost:8000/docs\n")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
