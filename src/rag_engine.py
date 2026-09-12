"""
Module RAG (Retrieval-Augmented Generation) Engine.
Chịu trách nhiệm nạp tài liệu (PDF, TXT, MD), phân đoạn (chunking),
lưu trữ vector vào ChromaDB và truy xuất ngữ cảnh trả lời câu hỏi.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from pypdf import PdfReader

from src.config import settings
from src.llm_client import OllamaClient


class DocumentLoader:
    """Xử lý đọc và trích xuất nội dung từ các định dạng file khác nhau."""

    @staticmethod
    def load_file(file_path: Path) -> str:
        """Đọc văn bản từ file .txt, .md hoặc .pdf."""
        suffix = file_path.suffix.lower()
        if suffix in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif suffix == ".pdf":
            reader = PdfReader(str(file_path))
            text_pages = []
            for idx, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text_pages.append(extracted)
            return "\n".join(text_pages)
        else:
            raise ValueError(f"Định dạng {suffix} chưa được hỗ trợ. Vui lòng dùng .txt, .md hoặc .pdf.")


class TextSplitter:
    """Phân đoạn văn bản dài thành các chunk nhỏ có overlap để bảo toàn ngữ cảnh."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Tách chuỗi thành các đoạn nhỏ với độ dài phù hợp."""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # Dịch chuyển start với bước nhảy trừ đi phần overlap
            start += max(1, self.chunk_size - self.chunk_overlap)
            
        return chunks


class RAGEngine:
    """Lớp điều phối toàn bộ luồng RAG."""

    def __init__(self, collection_name: str = "local_knowledge_base"):
        self.client_llm = OllamaClient()
        self.collection_name = collection_name
        
        # Khởi tạo thư mục lưu trữ ChromaDB
        settings.chroma_db_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.chroma_db_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        self.splitter = TextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

    def ingest_document(self, file_path: Path) -> int:
        """
        Nạp một tài liệu vào cơ sở tri thức (Vector DB).
        
        Returns:
            Số lượng chunk đã được nạp.
        """
        text = DocumentLoader.load_file(file_path)
        chunks = self.splitter.split_text(text)
        
        if not chunks:
            return 0

        # Tạo embedding cho từng chunk và lưu vào ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        filename = file_path.name
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            emb = self.client_llm.get_embedding(chunk)
            
            ids.append(chunk_id)
            embeddings.append(emb)
            metadatas.append({"source": filename, "chunk_index": i})
            documents.append(chunk)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        return len(chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Truy xuất các đoạn văn bản có độ tương đồng cao nhất với câu hỏi.
        """
        k = top_k or settings.top_k
        query_emb = self.client_llm.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
        )
        
        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0]
            for doc, meta in zip(docs, metas):
                retrieved.append({"content": doc, "metadata": meta})
                
        return retrieved

    def query(self, user_query: str) -> Any:
        """
        Thực hiện RAG: Truy xuất ngữ cảnh -> Ghép prompt -> Gọi Ollama trả lời dạng streaming.
        """
        # 1. Lấy ngữ cảnh liên quan
        context_items = self.retrieve(user_query)
        
        if not context_items:
            context_text = "Không có tài liệu tham khảo nào trong cơ sở dữ liệu."
        else:
            context_text = "\n\n---\n\n".join(
                f"[Tài liệu: {item['metadata'].get('source', 'Unknown')}]:\n{item['content']}"
                for item in context_items
            )

        # 2. Xây dựng prompt RAG chuẩn mực
        system_prompt = (
            "Bạn là một trợ lý AI thông minh, trung thực và hữu ích. "
            "Hãy trả lời câu hỏi của người dùng bằng tiếng Việt, dựa CHÍNH XÁC vào phần ngữ cảnh được cung cấp dưới đây. "
            "Nếu thông tin không có trong ngữ cảnh, hãy thừa nhận rõ ràng rằng bạn không tìm thấy trong tài liệu, "
            "tuyệt đối không bịa đặt thông tin."
        )

        user_content = (
            f"NGỮ CẢNH TÀI LIỆU:\n{context_text}\n\n"
            f"CÂU HỎI:\n{user_query}"
        )

        messages = [{"role": "user", "content": user_content}]

        # 3. Stream phản hồi
        return self.client_llm.stream_chat(
            messages=messages,
            system_prompt=system_prompt
        ), context_items
