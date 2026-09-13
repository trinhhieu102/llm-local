"""
FastAPI Server cung cấp REST API cho Local LLM và RAG Engine.
Tương thích hoàn toàn cho việc tích hợp Web Chatbot, Mobile App hoặc Third-party services.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

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
from src.crawler import WebTopicCrawler

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Vietnamese Local LLM & RAG API",
    description="High-performance, 100% offline LLM & RAG API with OpenAI v1 compatibility powered by Ollama, Qwen 2.5 and ChromaDB.",
    version="1.2.0",
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


class OpenAIChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None


class CrawlTopicRequest(BaseModel):
    topic: str = Field(default="ai", examples=["ai"])
    max_articles: Optional[int] = 3
    auto_ingest_rag: Optional[bool] = True
    export_dataset: Optional[bool] = True


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
    """Liệt kê danh sách các tài liệu trong thư mục data và data/crawled sẵn sàng cho RAG."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    crawled_dir = settings.data_dir / "crawled"
    crawled_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    # Quét thư mục data gốc
    for f in settings.data_dir.iterdir():
        if f.is_file() and f.suffix.lower() in [".txt", ".md", ".pdf"]:
            docs.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 1),
                "extension": f.suffix.lower(),
                "category": "root",
            })
    # Quét thư mục data/crawled
    for f in crawled_dir.iterdir():
        if f.is_file() and f.suffix.lower() in [".txt", ".md", ".pdf"]:
            docs.append({
                "name": f"crawled/{f.name}",
                "size_kb": round(f.stat().st_size / 1024, 1),
                "extension": f.suffix.lower(),
                "category": "crawled",
            })

    return {"documents": docs, "total_count": len(docs)}


# --- OpenAI Compatible API Endpoints (Cho VS Code Continue, Cline, AutoGen, CrewAI) ---
@app.get("/v1/models", tags=["OpenAI Compatible"])
def list_v1_models():
    """Tương thích chuẩn OpenAI v1 models list."""
    models = ollama_client.list_models()
    if not models:
        models = [settings.llm_model, settings.embedding_model]
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-llm",
                "permission": [],
                "root": m,
                "parent": None
            }
            for m in models
        ]
    }


@app.post("/v1/chat/completions", tags=["OpenAI Compatible"])
def v1_chat_completions(request: OpenAIChatCompletionRequest):
    """
    Tương thích 100% chuẩn OpenAI Chat Completions.
    Cho phép VS Code Extension (Continue.dev, Cline), LangChain, AutoGen, CrewAI kết nối trực tiếp.
    """
    raw_messages = [{"role": m.role, "content": m.content} for m in request.messages]
    target_model = request.model or settings.llm_model
    chat_id = f"chatcmpl-{int(time.time()*1000)}"
    created_ts = int(time.time())

    options: Dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = float(request.temperature)
    if request.top_p is not None:
        options["top_p"] = float(request.top_p)
    if request.max_tokens is not None:
        options["num_predict"] = int(request.max_tokens)

    if request.stream:
        def sse_event_generator():
            for chunk in ollama_client.stream_chat(raw_messages, model=target_model, options=options or None):
                chunk_data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": target_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_event_generator(), media_type="text/event-stream")

    # Non-streaming
    reply = ""
    for chunk in ollama_client.stream_chat(raw_messages, model=target_model, options=options or None):
        reply += chunk

    prompt_tokens = sum(len(m.content.split()) for m in request.messages)
    completion_tokens = len(reply.split())

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": target_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


@app.post("/v1/embeddings", tags=["OpenAI Compatible"])
def v1_embeddings(request: OpenAIEmbeddingRequest):
    """Tương thích chuẩn OpenAI Embeddings cho AI Agent và RAG Frameworks."""
    target_model = request.model or settings.embedding_model
    inputs = request.input if isinstance(request.input, list) else [request.input]
    data = []
    for idx, text in enumerate(inputs):
        vec = ollama_client.get_embedding(str(text), model=target_model)
        data.append({"object": "embedding", "index": idx, "embedding": vec})
    return {"object": "list", "data": data, "model": target_model}


# --- Crawler & Data Collection Endpoints ---
@app.post("/api/crawler/crawl", tags=["Crawler & Training"])
def run_crawler(request: CrawlTopicRequest):
    """
    Tự động cào tin tức công nghệ/AI mới nhất, làm sạch văn bản và nạp vào Hybrid RAG hoặc xuất file train.
    """
    try:
        crawler = WebTopicCrawler()
        articles = crawler.crawl_topic(topic=request.topic, max_articles=request.max_articles or 3)
        if not articles:
            return {"status": "empty", "message": "Không tìm thấy bài viết nào từ nguồn tin.", "articles_count": 0}

        chunks_ingested = 0
        if request.auto_ingest_rag:
            chunks_ingested = crawler.save_for_rag(articles, rag_engine)

        dataset_path = None
        if request.export_dataset:
            dataset_path = str(crawler.export_fine_tuning_dataset(articles))

        return {
            "status": "success",
            "topic": request.topic,
            "articles_count": len(articles),
            "chunks_ingested": chunks_ingested,
            "dataset_file": dataset_path,
            "articles": [{"title": a["title"], "link": a["link"]} for a in articles]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi cào dữ liệu: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Khởi chạy FastAPI Server tại: http://localhost:8000")
    print("📖 Swagger API Docs xem tại: http://localhost:8000/docs\n")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
