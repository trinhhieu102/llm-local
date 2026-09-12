"""
Script kiểm thử nhanh hiệu năng mô hình Qwen 2.5 3B trên card đồ họa NVIDIA RTX 2050.
"""

import sys
import time

# Thiết lập mã hóa UTF-8 cho Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import ollama

def main():
    print("==================================================")
    print("  KIỂM THỬ MÔ HÌNH QWEN 2.5 3B TRÊN LOCAL GPU")
    print("==================================================")
    
    prompt = "Xin chào! Hãy giới thiệu bạn là ai và bạn có thể làm được những gì bằng tiếng Việt trong khoảng 3 câu."
    print(f"\n[Câu hỏi]: {prompt}\n")
    print("[AI trả lời]: ", end="", flush=True)

    start_time = time.time()
    token_count = 0

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in response:
        content = chunk.get("message", {}).get("content", "")
        if content:
            print(content, end="", flush=True)
            token_count += 1

    elapsed = time.time() - start_time
    print("\n\n--------------------------------------------------")
    print(f"Thời gian phản hồi: {elapsed:.2f} giây")
    if elapsed > 0:
        print(f"Tốc độ ước tính: ~{token_count / elapsed:.1f} tokens/giây")
    print("==================================================")

if __name__ == "__main__":
    main()
