"""
Chương trình chính điều khiển Local LLM & RAG Engine.
Giao diện dòng lệnh trực quan với menu chức năng tiện dụng.
"""

import sys
from pathlib import Path

# Đảm bảo hiển thị tiếng Việt UTF-8 chuẩn trên console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo import được module trong src
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import settings
from src.llm_client import OllamaClient
from src.rag_engine import RAGEngine

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    console = Console()
except ImportError:
    # Fallback cơ bản nếu chưa cài rich
    class SimpleConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = SimpleConsole()


def print_banner():
    """Hiển thị tiêu đề ứng dụng."""
    text = (
        "[bold cyan]HỆ THỐNG LOCAL LLM & RAG TIẾNG VIỆT[/bold cyan]\n"
        f"[dim]Lõi: Ollama ({settings.ollama_host}) | Model LLM: {settings.llm_model} | Embedding: {settings.embedding_model}[/dim]"
    )
    if hasattr(console, "print"):
        console.print(Panel(text, expand=False, border_style="green"))
    else:
        print("=== HỆ THỐNG LOCAL LLM & RAG TIẾNG VIỆT ===")


def check_status():
    """Kiểm tra kết nối tới Ollama và danh sách các model."""
    client = OllamaClient()
    if not client.is_service_ready():
        console.print("[bold red]❌ Không thể kết nối tới dịch vụ Ollama![/bold red]")
        console.print("[yellow]Gợi ý: Hãy kiểm tra xem ứng dụng Ollama đã được khởi động chưa.[/yellow]\n")
        return False
    
    models = client.list_models()
    console.print("[bold green]✅ Kết nối Ollama thành công![/bold green]")
    console.print(f"Các model hiện có trong máy ({len(models)}):")
    for m in models:
        console.print(f"  • [cyan]{m}[/cyan]")
    console.print("")
    return True


def chat_mode():
    """Chế độ trò chuyện trực tiếp (General Chat / Code Assistance)."""
    client = OllamaClient()
    console.print("\n[bold green]=== CHẾ ĐỘ TRÒ CHUYỆN TRỰC TIẾP ===[/bold green]")
    console.print("[dim]Nhập câu hỏi hoặc yêu cầu code. Nhập 'exit' hoặc 'q' để quay lại menu chính.[/dim]\n")
    
    history = []
    
    while True:
        try:
            user_input = input("\nBạn > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "q", "quit"]:
                break
                
            history.append({"role": "user", "content": user_input})
            print("\nAI > ", end="", flush=True)
            
            full_reply = ""
            for token in client.stream_chat(history):
                print(token, end="", flush=True)
                full_reply += token
            print("\n")
            
            history.append({"role": "assistant", "content": full_reply})
            
            # Giới hạn lịch sử hội thoại 10 lượt gần nhất để tiết kiệm bộ nhớ
            if len(history) > 10:
                history = history[-10:]
                
        except KeyboardInterrupt:
            print("\n")
            break


def ingest_mode(engine: RAGEngine):
    """Chế độ nạp tài liệu vào Vector DB."""
    console.print("\n[bold green]=== NẠP TÀI LIỆU VÀO CƠ SỞ DỮ LIỆU RAG ===[/bold green]")
    console.print(f"Thư mục tài liệu mặc định: [cyan]{settings.data_dir}[/cyan]")
    
    # Liệt kê các file trong thư mục data
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in settings.data_dir.iterdir() if f.is_file() and f.suffix.lower() in [".txt", ".md", ".pdf"]]
    
    if not files:
        console.print("[yellow]Chưa có tài liệu nào (.txt, .md, .pdf) trong thư mục data/[/yellow]")
        return
        
    console.print("\nDanh sách tài liệu sẵn sàng nạp:")
    for idx, f in enumerate(files, 1):
        console.print(f"  {idx}. {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        
    choice = input("\nNhập số thứ tự file cần nạp (hoặc 'all' để nạp tất cả, 'q' để huỷ): ").strip().lower()
    if choice in ["q", "quit", "exit"]:
        return
        
    to_ingest = []
    if choice == "all":
        to_ingest = files
    else:
        try:
            num = int(choice)
            if 1 <= num <= len(files):
                to_ingest = [files[num - 1]]
            else:
                console.print("[red]Lựa chọn không hợp lệ.[/red]")
                return
        except ValueError:
            console.print("[red]Vui lòng nhập số hợp lệ.[/red]")
            return
            
    for f in to_ingest:
        console.print(f"Đang xử lý nạp file [cyan]{f.name}[/cyan]...")
        try:
            num_chunks = engine.ingest_document(f)
            console.print(f"[green]✓ Thành công:[/green] Đã phân tách và lưu {num_chunks} đoạn vector vào ChromaDB.")
        except Exception as e:
            console.print(f"[red]✗ Lỗi khi nạp file {f.name}:[/red] {e}")


def rag_query_mode(engine: RAGEngine):
    """Chế độ hỏi đáp dựa trên tài liệu đã nạp (RAG Q&A)."""
    console.print("\n[bold green]=== HỎI ĐÁP DỰA TRÊN TÀI LIỆU (RAG) ===[/bold green]")
    console.print("[dim]Hệ thống sẽ tìm kiếm thông tin từ cơ sở dữ liệu đã nạp để trả lời. Nhập 'q' để thoát.[/dim]\n")
    
    while True:
        try:
            query = input("\nCâu hỏi tài liệu > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "q", "quit"]:
                break
                
            stream_gen, context_items = engine.query(query)
            
            if context_items:
                console.print("\n[dim]🔍 Đã tìm thấy các đoạn tham chiếu liên quan:[/dim]")
                for i, item in enumerate(context_items, 1):
                    src = item['metadata'].get('source', 'Unknown')
                    console.print(f"  [dim]• [Nguồn: {src}][/dim]")
            else:
                console.print("[dim]⚠️ Không tìm thấy đoạn tài liệu tương ứng trong cơ sở dữ liệu.[/dim]")
                
            print("\nAI > ", end="", flush=True)
            for token in stream_gen:
                print(token, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\n")
            break


def main():
    """Hàm chạy chính của ứng dụng."""
    print_banner()
    
    if not check_status():
        print("Vui lòng khởi động Ollama trước khi sử dụng chương trình.")
        return

    rag_engine = RAGEngine()

    while True:
        console.print("\n[bold cyan]DANH MỤC CHỨC NĂNG:[/bold cyan]")
        console.print("1. Trò chuyện trực tiếp & Hỗ trợ lập trình (Chat Mode)")
        console.print("2. Nạp tài liệu vào cơ sở tri thức (Ingest Documents)")
        console.print("3. Hỏi đáp trên tài liệu nội bộ (RAG Mode)")
        console.print("4. Kiểm tra trạng thái hệ thống & Model")
        console.print("5. Thoát")
        
        choice = input("\nChọn chức năng (1-5): ").strip()
        
        if choice == "1":
            chat_mode()
        elif choice == "2":
            ingest_mode(rag_engine)
        elif choice == "3":
            rag_query_mode(rag_engine)
        elif choice == "4":
            check_status()
        elif choice in ["5", "q", "exit"]:
            console.print("[green]Tạm biệt![/green]")
            break
        else:
            console.print("[yellow]Lựa chọn không hợp lệ, vui lòng thử lại.[/yellow]")


if __name__ == "__main__":
    main()
