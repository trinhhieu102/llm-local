"""
Kiểm thử tự động quy trình RAG:
1. Nạp tài liệu mẫu vào ChromaDB
2. Truy xuất ngữ cảnh bằng vector embedding (bge-m3)
3. Sinh câu trả lời chính xác bằng Qwen 2.5 3B
"""

import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.rag_engine import RAGEngine
from src.config import settings

def test_rag_pipeline():
    print("==================================================")
    print("  KIỂM THỬ TỰ ĐỘNG QUY TRÌNH RAG VÀ VECTOR SEARCH")
    print("==================================================")
    
    engine = RAGEngine()
    sample_file = settings.data_dir / "tai_lieu_mau.txt"
    
    print(f"\n1. Nạp tài liệu: {sample_file.name}")
    start = time.time()
    num_chunks = engine.ingest_document(sample_file)
    print(f"✓ Đã băm và nạp thành công {num_chunks} chunks vào ChromaDB ({time.time() - start:.2f}s).")
    
    query = "Tốc độ sinh chữ của mô hình Qwen 2.5 3B trên máy này đạt bao nhiêu token mỗi giây?"
    print(f"\n2. Đặt câu hỏi RAG: '{query}'")
    
    start_q = time.time()
    stream_gen, context_items = engine.query(query)
    
    print(f"\n3. Ngữ cảnh tìm thấy ({len(context_items)} đoạn liên quan):")
    for idx, item in enumerate(context_items, 1):
        src = item['metadata'].get('source', 'Unknown')
        print(f"   [{idx}] Nguồn: {src} -> {item['content'][:80]}...")
        
    print("\n4. Câu trả lời của AI:")
    print("AI > ", end="", flush=True)
    for token in stream_gen:
        print(token, end="", flush=True)
    print("\n")
    print(f"✓ Hoàn tất trong {time.time() - start_q:.2f} giây.")
    print("==================================================")

if __name__ == "__main__":
    test_rag_pipeline()
