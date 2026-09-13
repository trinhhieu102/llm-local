# Hướng Dẫn Tích Hợp Local AI Vào VS Code & Xây Dựng AI Agent

Tài liệu này hướng dẫn cách kết nối hệ thống **Local LLM & RAG Engine** vào **Visual Studio Code** để làm trợ lý lập trình (Copilot) và gọi API xây dựng **AI Agent / Chatbot**.

---

## 💻 PHẦN 1: DÙNG NHƯ EXTENSION TRỢ LÝ CODE TRONG VS CODE

Extension mã nguồn mở số 1 để dùng Local LLM trong VS Code là **Continue.dev** (hoặc **Cline** / **Roo Code**).

### Bước 1: Cài đặt Extension trên VS Code
1. Mở VS Code, bấm tổ hợp phím `Ctrl + Shift + X` (mở tab Extensions).
2. Tìm kiếm từ khóa: **`Continue`** (của nhà phát triển *Continue*).
3. Bấm **Install**.

---

### Bước 2: Cấu hình kết nối tới Local LLM của bạn

1. Sau khi cài xong, bấm vào biểu tượng bánh răng ⚙️ ở góc dưới thanh bên của Continue, hoặc mở file:
   - **Windows:** `%USERPROFILE%\.continue\config.json`
2. Dán nội dung cấu hình dưới đây vào file `config.json`:

```json
{
  "models": [
    {
      "title": "Local Qwen 2.5 3B (FastAPI Gateway)",
      "provider": "openai",
      "model": "qwen2.5:3b",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "local-key-not-needed"
    },
    {
      "title": "Local Qwen 2.5 3B (Ollama Direct)",
      "provider": "ollama",
      "model": "qwen2.5:3b",
      "apiBase": "http://localhost:11434"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Local Autocomplete",
    "provider": "ollama",
    "model": "qwen2.5:3b",
    "apiBase": "http://localhost:11434"
  },
  "embeddingsProvider": {
    "provider": "openai",
    "model": "bge-m3",
    "apiBase": "http://localhost:8000/v1"
  }
}
```

---

### Bước 3: Sử dụng các tính năng trong VS Code

| Tính năng | Phím tắt | Mô tả cách dùng |
| :--- | :--- | :--- |
| **Chat Assistant** | `Ctrl + L` | Mở khung chat, hỏi đáp giải thích code, tìm lỗi trong dự án |
| **Inline Code Edit** | `Ctrl + I` | Bôi đen một đoạn code, gõ yêu cầu (vd: *"Refactor hàm này sang async và thêm docstring"*) |
| **Autocomplete** | `Tab` | Tự động gợi ý dòng code tiếp theo khi bạn đang gõ phím |
| **Context @Codebase** | Gõ `@Codebase` | Cho phép AI đọc toàn bộ dự án hiện tại để trả lời chuẩn xác |

---

## 🤖 PHẦN 2: GỌI API ĐỂ XÂY DỰNG CHATBOT & AI AGENT

Vì `api_server.py` đã chuẩn hóa tương thích 100% với **OpenAI API Specification**, bạn có thể dùng bất kỳ thư viện phổ biến nào để xây dựng AI Agent.

### Ví dụ 1: Sử dụng thư viện Python `openai` chính thức

```python
from openai import OpenAI

# Trỏ base_url về FastAPI Server Local của bạn
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Chạy local không cần API key
)

# 1. Trò chuyện dạng Streaming
response = client.chat.completions.create(
    model="qwen2.5:3b",
    messages=[
        {"role": "system", "content": "Bạn là kỹ sư phần mềm cao cấp."},
        {"role": "user", "content": "Viết một hàm Python kết nối PostgreSQL bằng SQLAlchemy."}
    ],
    temperature=0.3,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
print()
```

---

### Ví dụ 2: Xây dựng AI Agent tự động bằng `LangChain`

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Khởi tạo mô hình qua Local Gateway
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="qwen2.5:3b",
    temperature=0.2
)

# Định nghĩa Agent Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là chuyên gia phân tích dữ liệu AI. Hãy phân tích ngắn gọn."),
    ("user", "Hãy đánh giá tác động của chip GPU NVIDIA đối với cuộc cách mạng GenAI.")
])

agent_chain = prompt | llm | StrOutputParser()

# Kích hoạt Agent
result = agent_chain.invoke({})
print(result)
```

---

### Ví dụ 3: Xây dựng Đội Ngũ Đa Tác Nhân (Multi-Agent) bằng `CrewAI`

```python
from crewai import Agent, Task, Crew, LLM

# Khai báo Local LLM
local_llm = LLM(
    model="openai/qwen2.5:3b",
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Tạo Agent Chuyên gia Nghiên Cứu
researcher = Agent(
    role="Chuyên viên Nghiên cứu Công nghệ",
    goal="Thu thập và đúc kết thông tin mới nhất về thị trường AI",
    backstory="Bạn là chuyên gia phân tích xu hướng công nghệ tương lai.",
    llm=local_llm,
    verbose=True
)

# Tạo Nhiệm Vụ
task1 = Task(
    description="Tóm tắt 3 lợi ích cốt lõi của việc tự triển khai Local LLM On-Premise cho doanh nghiệp.",
    expected_output="Báo cáo 3 gạch đầu dòng rõ ràng, súc tích.",
    agent=researcher
)

crew = Crew(agents=[researcher], tasks=[task1])
result = crew.kickoff()
print(result)
```

---

## 🕷️ PHẦN 3: CÀO DỮ LIỆU HOT ĐỂ NẠP RAG VÀ TẠO DATASET TRAIN

Hệ thống cung cấp sẵn công cụ cào dữ liệu trong [src/crawler.py](file:///c:/Users/ADMIN/Downloads/local%20llm/src/crawler.py).

### Cách 1: Gọi qua API
```powershell
curl -X POST "http://localhost:8000/api/crawler/crawl" `
  -H "Content-Type: application/json" `
  -d '{"topic": "ai", "max_articles": 5, "auto_ingest_rag": true, "export_dataset": true}'
```

### Cách 2: Chạy trực tiếp script
```powershell
python src/crawler.py
```

### Kết quả thu được:
1. **RAG Knowledge Base:** Các bài viết được lưu tự động vào `data/crawled/` và lập chỉ mục ngay lập tức vào ChromaDB + BM25 để hỏi đáp trên Web Studio hoặc VS Code.
2. **Dataset Fine-Tuning:** Dữ liệu tự động được chuyển đổi thành cặp câu hỏi - câu trả lời lưu tại `data/train_dataset.jsonl` chuẩn format ChatML để mang đi Fine-tune LoRA/QLoRA bằng Unsloth hoặc Axolotl.
