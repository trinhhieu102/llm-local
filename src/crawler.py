"""
Module Web Crawler & Dataset Generator phục vụ RAG và Fine-tuning LLM.
Tự động thu thập tin tức, bài báo chuyên sâu về các lĩnh vực hot (AI, GPU, Bán dẫn, Tài chính),
làm sạch văn bản và nạp trực tiếp vào ChromaDB hoặc xuất dataset JSONL.
"""

import os
import sys
import re
import json
import time
import html
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo import được module trong src khi chạy trực tiếp
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.config import settings
from src.rag_engine import RAGEngine


# Các nguồn tin tức RSS mở, uy tín và cập nhật liên tục về các chủ đề hot
HOT_TOPIC_FEEDS = {
    "ai": [
        {"name": "ArXiv AI Papers", "url": "http://export.arxiv.org/rss/cs.AI"},
        {"name": "MIT Technology Review AI", "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed"},
    ],
    "hardware": [
        {"name": "AnandTech Hardware", "url": "https://www.anandtech.com/rss/"},
        {"name": "Tom's Hardware News", "url": "https://www.tomshardware.com/feeds/all"},
    ],
    "tech_finance": [
        {"name": "TechCrunch Enterprise", "url": "https://techcrunch.com/category/enterprise/feed/"},
    ]
}


class DataCleaner:
    """Làm sạch văn bản cào từ web/RSS."""

    @staticmethod
    def strip_html(raw_html: str) -> str:
        """Loại bỏ thẻ HTML, style, scripts và giải mã HTML entities."""
        if not raw_html:
            return ""
        # Xóa scripts và styles
        clean = re.sub(r"<(script|style)[^>]*>[\s\S]*?</\1>", "", raw_html, flags=re.IGNORECASE)
        # Xóa các thẻ HTML
        clean = re.sub(r"<[^>]+>", " ", clean)
        # Giải mã các ký tự HTML entity như &amp;, &quot;
        clean = html.unescape(clean)
        # Chuẩn hóa khoảng trắng
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean


class WebTopicCrawler:
    """Crawler thu thập dữ liệu chuyên sâu theo chủ đề."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (settings.data_dir / "crawled")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_rss_feed(self, feed_url: str, max_items: int = 5) -> List[Dict[str, Any]]:
        """Đọc và trích xuất danh sách bài viết từ một RSS feed."""
        articles = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LocalLLM-Crawler/1.0"
        }
        
        try:
            req = urllib.request.Request(feed_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
                
            root = ET.fromstring(xml_data)
            # Hỗ trợ cả chuẩn RSS 2.0 (item), Atom (entry) và RDF
            items = []
            for tag in [".//item", ".//{http://www.w3.org/2005/Atom}entry", ".//{http://purl.org/rss/1.0/}item"]:
                found = root.findall(tag)
                if found:
                    items = found
                    break

            for item in items[:max_items]:
                # Title
                title_elem = None
                for t_tag in ["title", "{http://www.w3.org/2005/Atom}title", "{http://purl.org/rss/1.0/}title"]:
                    cand = item.find(t_tag)
                    if cand is not None:
                        title_elem = cand
                        break
                title = DataCleaner.strip_html(title_elem.text if title_elem is not None and title_elem.text else "Tin tức AI")

                # Link
                link = ""
                for l_tag in ["link", "{http://www.w3.org/2005/Atom}link", "{http://purl.org/rss/1.0/}link"]:
                    l_elem = item.find(l_tag)
                    if l_elem is not None:
                        link = l_elem.text or l_elem.attrib.get("href", "")
                        if link:
                            break

                # Content / Description
                content = ""
                for d_tag in ["description", "{http://www.w3.org/2005/Atom}summary", "{http://www.w3.org/2005/Atom}content", "{http://purl.org/rss/1.0/}description"]:
                    d_elem = item.find(d_tag)
                    if d_elem is not None and d_elem.text:
                        content = DataCleaner.strip_html(d_elem.text)
                        if len(content) > 30:
                            break

                if title and content and len(content) > 30:
                    articles.append({
                        "title": title,
                        "link": link,
                        "content": content,
                        "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })

        except Exception as e:
            print(f"[Crawler Warning] Không thể cào từ feed {feed_url}: {e}")

        return articles

    def crawl_topic(self, topic: str = "ai", max_articles: int = 5) -> List[Dict[str, Any]]:
        """Thu thập bài viết mới nhất theo danh mục chủ đề."""
        feeds = HOT_TOPIC_FEEDS.get(topic.lower(), HOT_TOPIC_FEEDS["ai"])
        collected = []

        for feed_info in feeds:
            if len(collected) >= max_articles:
                break
            items_needed = max_articles - len(collected)
            feed_items = self.fetch_rss_feed(feed_info["url"], max_items=items_needed)
            for item in feed_items:
                item["source_feed"] = feed_info["name"]
                item["topic"] = topic
            collected.extend(feed_items)

        return collected[:max_articles]

    def save_for_rag(self, articles: List[Dict[str, Any]], rag_engine: Optional[RAGEngine] = None) -> int:
        """
        Lưu các bài viết thành file văn bản trong data/crawled/
        và nạp trực tiếp vào cơ sở dữ liệu vector ChromaDB + BM25.
        
        Returns:
            Số lượng chunks đã nạp thành công vào RAG.
        """
        if not articles:
            return 0

        engine = rag_engine or RAGEngine()
        total_chunks = 0

        for idx, art in enumerate(articles, 1):
            clean_title = re.sub(r'[\\/*?:"<>|]', "", art["title"])[:50].strip().replace(" ", "_")
            file_name = f"crawled_{art.get('topic', 'news')}_{int(time.time())}_{idx}_{clean_title}.txt"
            file_path = self.output_dir / file_name

            # Format văn bản chuẩn có metadata đầu bài
            text_content = (
                f"TIÊU ĐỀ: {art['title']}\n"
                f"NGUỒN: {art.get('source_feed', 'Web Crawler')}\n"
                f"LIÊN KẾT: {art.get('link', '')}\n"
                f"THỜI GIAN THU THẬP: {art.get('crawled_at', '')}\n\n"
                f"NỘI DUNG CHI TIẾT:\n{art['content']}\n"
            )

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            # Nạp vào RAG
            chunks = engine.ingest_document(file_path)
            total_chunks += chunks

        return total_chunks

    def export_fine_tuning_dataset(
        self,
        articles: List[Dict[str, Any]],
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Xuất các bài viết thành dataset JSONL chuẩn Alpaca/ChatML phục vụ Fine-tuning LoRA/QLoRA:
        {"instruction": "...", "input": "...", "output": "..."}
        """
        out_path = output_file or (settings.data_dir / "train_dataset.jsonl")
        samples = []

        for art in articles:
            # Dạng 1: Tóm tắt thông tin bài viết
            samples.append({
                "instruction": "Tóm tắt các điểm quan trọng và xu hướng công nghệ trong bài viết sau:",
                "input": f"Tiêu đề: {art['title']}\nNội dung: {art['content']}",
                "output": f"Bài viết '{art['title']}' trình bày về các diễn biến mới nhất: {art['content'][:250]}..."
            })
            # Dạng 2: Hỏi đáp dữ kiện
            samples.append({
                "instruction": f"Chủ đề công nghệ chính được đề cập trong bài viết '{art['title']}' là gì?",
                "input": art["content"],
                "output": f"Chủ đề trọng tâm là về {art.get('topic', 'công nghệ')} với các phân tích: {art['content'][:180]}."
            })

        with open(out_path, "a", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        return out_path


# Helper chạy nhanh độc lập
if __name__ == "__main__":
    crawler = WebTopicCrawler()
    print("🕷️ Bắt đầu cào 3 bài viết về chủ đề AI...")
    articles = crawler.crawl_topic(topic="ai", max_articles=3)
    print(f"✓ Đã thu thập {len(articles)} bài viết.")
    
    rag = RAGEngine()
    chunks = crawler.save_for_rag(articles, rag)
    print(f"✓ Đã băm và nạp thành công {chunks} chunks vào Hybrid RAG.")

    dataset_path = crawler.export_fine_tuning_dataset(articles)
    print(f"✓ Đã xuất dataset fine-tuning tại: {dataset_path}")
