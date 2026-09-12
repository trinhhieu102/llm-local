"""
Module giao tiếp với dịch vụ Ollama.
Cung cấp các hàm tương tác: Chat streaming, hỏi đáp đơn lẻ, kiểm tra trạng thái và tạo Embeddings.
"""

from typing import Generator, List, Dict, Any, Optional
import ollama
from src.config import settings


class OllamaClient:
    """Wrapper quản lý giao tiếp với Ollama API."""

    def __init__(self, host: Optional[str] = None):
        self.host = host or settings.ollama_host
        self.client = ollama.Client(host=self.host)

    def is_service_ready(self) -> bool:
        """Kiểm tra dịch vụ Ollama có đang hoạt động hay không."""
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Lấy danh sách các model đã được tải về máy."""
        try:
            response = self.client.list()
            models = response.get("models", [])
            return [m.get("name", "") for m in models if "name" in m]
        except Exception as e:
            return []

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Chat ở chế độ Streaming (trả về từng token như ChatGPT).
        
        Args:
            messages: Danh sách tin nhắn dạng [{"role": "user", "content": "..."}]
            model: Tên model sử dụng (mặc định lấy từ cấu hình settings.llm_model)
            system_prompt: Lời nhắc định hình vai trò cho AI
        """
        target_model = model or settings.llm_model
        chat_messages = []
        
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
            
        chat_messages.extend(messages)

        try:
            stream = self.client.chat(
                model=target_model,
                messages=chat_messages,
                stream=True,
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            yield f"\n[Lỗi kết nối Ollama]: {str(e)}"

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Tạo vector nhúng (embedding) cho đoạn văn bản phục vụ RAG.
        
        Args:
            text: Nội dung cần tạo vector
            model: Model embedding (mặc định settings.embedding_model)
        """
        target_model = model or settings.embedding_model
        try:
            response = self.client.embeddings(model=target_model, prompt=text)
            return response.get("embedding", [])
        except Exception as e:
            raise RuntimeError(f"Lỗi tạo embedding với model {target_model}: {str(e)}")
