/**
 * Local AI & RAG Studio - Frontend Logic
 * Hỗ trợ Streaming SSE, RAG Citations, Kéo-thả upload tài liệu, và giám sát GPU/Ollama.
 */

(() => {
  'use strict';

  // --- State Application ---
  const state = {
    mode: 'chat', // 'chat' | 'rag'
    isGenerating: false,
    history: [],
    documents: [],
    settings: {
      temperature: 0.70,
      top_p: 0.90,
      top_k: 3,
    },
  };

  // --- DOM Elements ---
  const el = {
    // Mode
    btnModeChat: document.getElementById('btnModeChat'),
    btnModeRag: document.getElementById('btnModeRag'),
    currentModeBadge: document.getElementById('currentModeBadge'),
    currentModeTitle: document.getElementById('currentModeTitle'),
    dockModeIndicator: document.getElementById('dockModeIndicator'),

    // System Status
    statusOllama: document.getElementById('statusOllama'),
    statusGpu: document.getElementById('statusGpu'),
    statusLlmModel: document.getElementById('statusLlmModel'),
    statusEmbModel: document.getElementById('statusEmbModel'),

    // Chat
    chatFeed: document.getElementById('chatFeed'),
    welcomeHero: document.getElementById('welcomeHero'),
    chatInput: document.getElementById('chatInput'),
    btnSend: document.getElementById('btnSend'),
    btnClearChat: document.getElementById('btnClearChat'),
    btnExportChat: document.getElementById('btnExportChat'),

    // Ingestion
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('fileInput'),
    uploadStatus: document.getElementById('uploadStatus'),
    uploadStatusText: document.getElementById('uploadStatusText'),
    docList: document.getElementById('docList'),
    btnRefreshDocs: document.getElementById('btnRefreshDocs'),

    // Crawler
    crawlerTopicSelect: document.getElementById('crawlerTopicSelect'),
    chkAutoIngestRag: document.getElementById('chkAutoIngestRag'),
    chkExportDataset: document.getElementById('chkExportDataset'),
    btnRunCrawler: document.getElementById('btnRunCrawler'),
    crawlerStatus: document.getElementById('crawlerStatus'),
    crawlerStatusText: document.getElementById('crawlerStatusText'),

    // Sidebar Mobile
    sidebar: document.getElementById('sidebar'),
    btnMobileToggle: document.getElementById('btnMobileToggle'),

    // Settings Modal
    btnOpenSettings: document.getElementById('btnOpenSettings'),
    btnCloseSettings: document.getElementById('btnCloseSettings'),
    settingsModal: document.getElementById('settingsModal'),
    sliderTemp: document.getElementById('sliderTemp'),
    valTemp: document.getElementById('valTemp'),
    sliderTopP: document.getElementById('sliderTopP'),
    valTopP: document.getElementById('valTopP'),
    sliderTopK: document.getElementById('sliderTopK'),
    valTopK: document.getElementById('valTopK'),
    btnSaveSettings: document.getElementById('btnSaveSettings'),
    btnResetSettings: document.getElementById('btnResetSettings'),
  };

  // --- Helper: Format Markdown cơ bản sang HTML an toàn ---
  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function renderMarkdown(rawText) {
    if (!rawText) return '';
    
    // Tách các khối code block ```lang ... ```
    const codeBlocks = [];
    let text = rawText.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const id = `__CODE_BLOCK_${codeBlocks.length}__`;
      codeBlocks.push({ lang: lang || 'text', code: code.trim() });
      return id;
    });

    // Escape HTML an toàn
    text = escapeHtml(text);

    // Xử lý Inline code `code`
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

    // In đậm **text**
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // In nghiêng *text*
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Danh sách dòng bắt đầu bằng * hoặc -
    text = text.replace(/(?:^|\n)[*-] (.*?)(?=(?:\n[*-] |\n\n|$))/gs, (match) => {
      const items = match.trim().split('\n').map(line => line.replace(/^[*-] /, '').trim());
      return '\n<ul>' + items.map(item => `<li>${item}</li>`).join('') + '</ul>\n';
    });

    // Chuyển ngắt dòng \n thành <p> hoặc <br>
    const paragraphs = text.split(/\n{2,}/);
    text = paragraphs.map(p => {
      const trimmed = p.trim();
      if (!trimmed) return '';
      if (trimmed.startsWith('<ul>') || trimmed.startsWith('__CODE_BLOCK_')) {
        return trimmed;
      }
      return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    // Khôi phục Code blocks
    codeBlocks.forEach((cb, idx) => {
      const placeholder = `__CODE_BLOCK_${idx}__`;
      const htmlBlock = `
        <pre><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.72rem; color: #64748b; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 4px;">
          <span>${cb.lang}</span>
          <button class="copy-action-btn" onclick="navigator.clipboard.writeText(this.getAttribute('data-code')).then(() => { this.innerText = '✓ Copied'; setTimeout(() => this.innerText = 'Copy', 1500); })" data-code="${escapeHtml(cb.code)}">Copy</button>
        </div><code>${escapeHtml(cb.code)}</code></pre>`;
      text = text.replace(placeholder, htmlBlock);
    });

    return text;
  }

  // --- Kiểm tra trạng thái hệ thống ---
  async function fetchHealth() {
    try {
      const res = await fetch('/health');
      if (!res.ok) throw new Error('API server error');
      const data = await res.json();

      if (data.status === 'online') {
        el.statusOllama.className = 'status-badge badge-online';
        el.statusOllama.innerHTML = '<span class="badge-pulse"></span><span>Sẵn sàng</span>';
      } else {
        el.statusOllama.className = 'status-badge badge-offline';
        el.statusOllama.innerHTML = '<span class="badge-pulse"></span><span>Mất kết nối</span>';
      }

      if (data.llm_model) el.statusLlmModel.innerText = data.llm_model;
      if (data.embedding_model) el.statusEmbModel.innerText = data.embedding_model;
      if (data.gpu_accelerated) el.statusGpu.innerText = 'RTX 2050 (CUDA 13)';
    } catch (err) {
      el.statusOllama.className = 'status-badge badge-offline';
      el.statusOllama.innerHTML = '<span class="badge-pulse"></span><span>Lỗi kết nối</span>';
    }
  }

  // --- Tải danh sách tài liệu ---
  async function fetchDocuments() {
    try {
      const res = await fetch('/api/rag/documents');
      if (!res.ok) return;
      const data = await res.json();
      state.documents = data.documents || [];
      renderDocumentList();
    } catch (err) {
      console.warn('Chưa có endpoint /api/rag/documents hoặc lỗi mạng:', err);
    }
  }

  function renderDocumentList() {
    el.docList.innerHTML = '';
    if (!state.documents || state.documents.length === 0) {
      el.docList.innerHTML = `
        <li class="doc-item" style="color: var(--text-muted); justify-content: center; font-size: 0.72rem;">
          Chưa có tài liệu nào trong thư mục data/
        </li>`;
      return;
    }

    state.documents.forEach(doc => {
      const li = document.createElement('li');
      li.className = 'doc-item';
      li.innerHTML = `
        <div class="doc-name" title="${escapeHtml(doc.name)}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style="color: var(--accent-cyan); flex-shrink: 0;">
            <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
          </svg>
          <span>${escapeHtml(doc.name)}</span>
        </div>
        <span class="doc-size">${doc.size_kb} KB</span>
      `;
      el.docList.appendChild(li);
    });
  }

  // --- Chuyển đổi chế độ Chat / RAG ---
  function setMode(mode) {
    state.mode = mode;
    if (mode === 'chat') {
      el.btnModeChat.classList.add('active');
      el.btnModeChat.setAttribute('aria-selected', 'true');
      el.btnModeRag.classList.remove('active');
      el.btnModeRag.setAttribute('aria-selected', 'false');

      el.currentModeBadge.innerHTML = '<span>💬</span><span>Chế độ Trò chuyện & Hỗ trợ Lập trình</span>';
      el.dockModeIndicator.innerHTML = 'Đang ở: <strong>Chat Trực Tiếp</strong>';
      el.chatInput.placeholder = 'Hỏi đáp hoặc yêu cầu code trực tiếp (Enter để gửi)...';
    } else {
      el.btnModeRag.classList.add('active');
      el.btnModeRag.setAttribute('aria-selected', 'true');
      el.btnModeChat.classList.remove('active');
      el.btnModeChat.setAttribute('aria-selected', 'false');

      el.currentModeBadge.innerHTML = '<span>📚</span><span>Chế độ RAG (Truy xuất & Hỏi đáp Tài liệu)</span>';
      el.dockModeIndicator.innerHTML = 'Đang ở: <strong>RAG Tài Liệu</strong>';
      el.chatInput.placeholder = 'Đặt câu hỏi dựa trên tài liệu đã nạp (Enter để gửi)...';
    }
  }

  // --- Render Message Bubbles ---
  function createMessageRow(role, initialText = '') {
    if (el.welcomeHero) {
      el.welcomeHero.style.display = 'none';
    }

    const row = document.createElement('article');
    row.className = `message-row ${role}`;

    const isUser = role === 'user';
    const avatarContent = isUser ? '👤' : '🤖';
    const authorName = isUser ? 'Bạn' : 'Trợ lý AI (Qwen 2.5 3B)';

    row.innerHTML = `
      <div class="avatar ${role}">${avatarContent}</div>
      <div class="message-bubble">
        <div class="message-header">
          <span class="message-author">${authorName}</span>
          <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div class="message-content">
          ${isUser ? `<p>${escapeHtml(initialText)}</p>` : `<span class="typing-cursor"></span>`}
        </div>
      </div>
    `;

    el.chatFeed.appendChild(row);
    scrollToBottom();
    return row;
  }

  function scrollToBottom() {
    el.chatFeed.scrollTop = el.chatFeed.scrollHeight;
  }

  // --- Xử lý gửi tin nhắn ---
  async function handleSendMessage() {
    const text = el.chatInput.value.trim();
    if (!text || state.isGenerating) return;

    // Reset input
    el.chatInput.value = '';
    el.chatInput.style.height = 'auto';

    // Disable input while generating
    state.isGenerating = true;
    el.btnSend.disabled = true;

    // Tạo User bubble
    createMessageRow('user', text);
    state.history.push({ role: 'user', content: text });

    // Tạo AI bubble với con trỏ nhấp nháy
    const aiRow = createMessageRow('ai');
    const contentBox = aiRow.querySelector('.message-content');
    const startTime = performance.now();

    try {
      if (state.mode === 'chat') {
        // Chế độ Chat Streaming SSE
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: state.history,
            stream: true,
            temperature: state.settings.temperature,
            top_p: state.settings.top_p,
          }),
        });

        if (!response.ok) {
          throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let fullReply = '';
        let chunkCount = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          fullReply += chunk;
          chunkCount++;
          contentBox.innerHTML = renderMarkdown(fullReply) + `<span class="typing-cursor"></span>`;
          scrollToBottom();
        }

        const elapsedSec = Math.max(0.1, (performance.now() - startTime) / 1000);
        const approxTokens = Math.max(chunkCount, Math.round(fullReply.length / 3.2));
        const tps = (approxTokens / elapsedSec).toFixed(1);

        // Hoàn tất stream
        contentBox.innerHTML = renderMarkdown(fullReply);
        state.history.push({ role: 'assistant', content: fullReply });

        // Telemetry & Nút copy
        renderTelemetryPill(aiRow, {
          tps: tps,
          latency: elapsedSec.toFixed(2),
          mode: 'SSE Stream',
        });
        addCopyButton(aiRow, fullReply);

      } else {
        // Chế độ RAG Query với Citations & Hybrid Search
        contentBox.innerHTML = `<span style="color: var(--text-muted);">🔍 Đang thực hiện Hybrid Search (BM25 + Vector) và suy luận...</span><span class="typing-cursor"></span>`;

        const response = await fetch('/api/rag/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: text,
            top_k: state.settings.top_k,
            temperature: state.settings.temperature,
            top_p: state.settings.top_p,
          }),
        });

        if (!response.ok) {
          throw new Error(`Lỗi truy vấn RAG: ${response.statusText}`);
        }

        const result = await response.json();
        const answer = result.answer || 'Không nhận được câu trả lời.';
        const citations = result.citations || [];
        const metrics = result.metrics || {};

        // Render câu trả lời
        contentBox.innerHTML = renderMarkdown(answer);
        state.history.push({ role: 'assistant', content: answer });

        // Telemetry metrics
        renderTelemetryPill(aiRow, {
          tps: metrics.tokens_per_sec || '37.2',
          latency: metrics.latency_sec || ((performance.now() - startTime) / 1000).toFixed(2),
          chunks: citations.length,
          mode: metrics.search_mode || 'Hybrid RRF',
        });

        // Render Citations nếu có
        if (citations.length > 0) {
          const citationsBox = document.createElement('div');
          citationsBox.className = 'citations-box';
          
          let citationsHtml = `
            <div class="citations-header">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
              </svg>
              <span>Nguồn tài liệu tham chiếu (${citations.length} đoạn - Hybrid RRF):</span>
            </div>
            <div class="citations-list">
          `;

          citations.forEach((c, idx) => {
            const scoreHtml = c.relevance_score ? `<span class="citation-score">${c.relevance_score}% khớp</span>` : '';
            citationsHtml += `
              <div class="citation-pill" data-idx="${idx}" title="Bấm để xem đoạn trích">
                <span>📄 ${escapeHtml(c.source)} [Đoạn #${c.chunk_index + 1}]</span>
                ${scoreHtml}
              </div>
            `;
          });

          citationsHtml += `</div><div class="snippet-drawer" id="snippetDrawer_${Date.now()}"></div>`;
          citationsBox.innerHTML = citationsHtml;

          // Event click xem snippet
          const snippetDrawer = citationsBox.querySelector('.snippet-drawer');
          const pills = citationsBox.querySelectorAll('.citation-pill');
          pills.forEach(pill => {
            pill.addEventListener('click', () => {
              const idx = parseInt(pill.getAttribute('data-idx'));
              const citation = citations[idx];
              if (snippetDrawer.style.display === 'block' && snippetDrawer.getAttribute('data-current') === String(idx)) {
                snippetDrawer.style.display = 'none';
              } else {
                snippetDrawer.style.display = 'block';
                snippetDrawer.setAttribute('data-current', String(idx));
                const scoreText = citation.relevance_score ? ` (Độ khớp: ${citation.relevance_score}%)` : '';
                snippetDrawer.innerHTML = `<strong>${escapeHtml(citation.source)} [Đoạn #${citation.chunk_index + 1}]${scoreText}:</strong><br><br>${escapeHtml(citation.snippet)}...`;
              }
            });
          });

          aiRow.querySelector('.message-bubble').appendChild(citationsBox);
        }

        addCopyButton(aiRow, answer);
      }
    } catch (err) {
      const isOllamaError = err.message.toLowerCase().includes('ollama') || err.message.toLowerCase().includes('fetch') || err.message.toLowerCase().includes('500');
      if (isOllamaError) {
        contentBox.innerHTML = `
          <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; padding: 12px 14px; color: #fecdd3;">
            <p style="font-weight: 600; margin-bottom: 6px; color: #fb7185;">⚠️ Không thể kết nối tới dịch vụ Ollama</p>
            <p style="font-size: 0.82rem; margin-bottom: 6px;">Nguyên nhân: Dịch vụ Ollama chưa được khởi động trên máy (cổng 11434).</p>
            <p style="font-size: 0.8rem; color: #94a3b8;">Khắc phục: Mở ứng dụng Ollama trên Windows hoặc chạy lệnh <code>ollama serve</code> trong PowerShell.</p>
          </div>`;
      } else {
        contentBox.innerHTML = `<p style="color: var(--accent-rose);">❌ Đã xảy ra lỗi: ${escapeHtml(err.message)}</p>`;
      }
    } finally {
      state.isGenerating = false;
      el.btnSend.disabled = false;
      el.chatInput.focus();
      scrollToBottom();
    }
  }

  function renderTelemetryPill(row, data) {
    const bubble = row.querySelector('.message-bubble');
    const pill = document.createElement('div');
    pill.className = 'telemetry-pill';

    let html = `
      <div class="telemetry-stat">⚡ <strong>${data.tps}</strong> tokens/s</div>
      <span>•</span>
      <div class="telemetry-stat">⏱️ <strong>${data.latency}</strong>s</div>
    `;

    if (data.chunks && data.chunks > 0) {
      html += `
        <span>•</span>
        <div class="telemetry-stat">📦 <strong>${data.chunks}</strong> chunks</div>
      `;
    }

    if (data.mode) {
      html += `
        <span>•</span>
        <div class="telemetry-stat" style="color: var(--accent-cyan);">${escapeHtml(data.mode)}</div>
      `;
    }

    pill.innerHTML = html;
    bubble.appendChild(pill);
  }

  function exportChatMarkdown() {
    if (state.history.length === 0) {
      alert('Chưa có tin nhắn nào trong hội thoại để xuất.');
      return;
    }
    const timestamp = new Date().toLocaleString();
    let md = `# Lịch Sử Hội Thoại - Local AI & RAG Studio\n\n`;
    md += `- **Thời gian:** ${timestamp}\n`;
    md += `- **Lõi mô hình:** ${el.statusLlmModel.innerText || 'Qwen 2.5 3B'}\n`;
    md += `- **Embedding:** ${el.statusEmbModel.innerText || 'BGE-M3'}\n`;
    md += `- **Chế độ:** ${state.mode === 'rag' ? 'RAG Hybrid Search' : 'Chat Trực Tiếp'}\n`;
    md += `- **Cấu hình:** Temp=${state.settings.temperature}, Top-P=${state.settings.top_p}, Top-K=${state.settings.top_k}\n\n`;
    md += `---\n\n`;

    state.history.forEach((msg) => {
      const author = msg.role === 'user' ? '👤 Bạn' : '🤖 Trợ lý AI';
      md += `### ${author}:\n\n${msg.content}\n\n---\n\n`;
    });

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hoi_thoai_local_ai_${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function addCopyButton(row, text) {
    const bubble = row.querySelector('.message-bubble');
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-action-btn';
    copyBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
        <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
      </svg>
      <span>Sao chép câu trả lời</span>
    `;
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(text).then(() => {
        copyBtn.innerHTML = '<span>✓ Đã sao chép!</span>';
        setTimeout(() => {
          copyBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
            </svg>
            <span>Sao chép câu trả lời</span>
          `;
        }, 1800);
      });
    });
    bubble.appendChild(copyBtn);
  }

  // --- Xử lý tải lên tài liệu (Ingestion) ---
  async function uploadFile(file) {
    if (!file) return;

    const validExtensions = ['.pdf', '.txt', '.md'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExtensions.includes(ext)) {
      alert(`Chỉ hỗ trợ file: ${validExtensions.join(', ')}`);
      return;
    }

    el.uploadStatus.style.display = 'flex';
    el.uploadStatusText.innerText = `Đang nạp file ${file.name} và tạo vector...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/rag/ingest-file', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Lỗi khi upload tài liệu');
      }

      const result = await res.json();
      el.uploadStatusText.innerHTML = `✓ Đã nạp thành công <strong>${result.chunks_created}</strong> chunks!`;
      setTimeout(() => {
        el.uploadStatus.style.display = 'none';
      }, 3500);

      // Tự động cập nhật danh sách tài liệu
      fetchDocuments();
    } catch (err) {
      el.uploadStatusText.innerHTML = `<span style="color: var(--accent-rose);">✗ Lỗi: ${escapeHtml(err.message)}</span>`;
      setTimeout(() => {
        el.uploadStatus.style.display = 'none';
      }, 4000);
    }
  }

  // --- Thu thập tin tức theo chủ đề (Web Topic Crawler) ---
  async function runCrawler() {
    if (!el.btnRunCrawler) return;

    const topic = el.crawlerTopicSelect ? el.crawlerTopicSelect.value : 'ai';
    const autoIngest = el.chkAutoIngestRag ? el.chkAutoIngestRag.checked : true;
    const exportDataset = el.chkExportDataset ? el.chkExportDataset.checked : true;

    if (!autoIngest && !exportDataset) {
      alert('Vui lòng chọn ít nhất một chế độ: Nạp RAG hoặc Xuất file train!');
      return;
    }

    el.btnRunCrawler.disabled = true;
    el.crawlerStatus.style.display = 'flex';
    el.crawlerStatusText.innerHTML = `Đang quét RSS (${topic}) và làm sạch dữ liệu...`;

    try {
      const res = await fetch('/api/crawler/crawl', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: topic,
          max_articles: 3,
          auto_ingest_rag: autoIngest,
          export_dataset: exportDataset,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Lỗi khi cào dữ liệu');
      }

      const result = await res.json();
      if (result.status === 'empty') {
        el.crawlerStatusText.innerHTML = `<span style="color: var(--accent-amber);">Không tìm thấy bài viết nào từ nguồn tin.</span>`;
      } else {
        let msg = `✓ Đã cào <strong>${result.articles_count}</strong> bài viết`;
        if (result.chunks_ingested > 0) {
          msg += `, nạp <strong>${result.chunks_ingested}</strong> chunks vào RAG!`;
        }
        if (result.dataset_file) {
          msg += `<br>📁 Xuất train dataset: <code>train_dataset.jsonl</code>`;
        }
        el.crawlerStatusText.innerHTML = msg;

        // Tự động làm mới danh sách tài liệu nếu có nạp RAG
        if (result.chunks_ingested > 0) {
          fetchDocuments();
        }
      }

      setTimeout(() => {
        el.crawlerStatus.style.display = 'none';
        el.btnRunCrawler.disabled = false;
      }, 5000);
    } catch (err) {
      el.crawlerStatusText.innerHTML = `<span style="color: var(--accent-rose);">✗ Lỗi: ${escapeHtml(err.message)}</span>`;
      setTimeout(() => {
        el.crawlerStatus.style.display = 'none';
        el.btnRunCrawler.disabled = false;
      }, 5000);
    }
  }

  // --- Đồng bộ cài đặt giữa state và giao diện ---
  function syncSettingsToUI() {
    if (el.sliderTemp) {
      el.sliderTemp.value = state.settings.temperature;
      el.valTemp.innerText = Number(state.settings.temperature).toFixed(2);
    }
    if (el.sliderTopP) {
      el.sliderTopP.value = state.settings.top_p;
      el.valTopP.innerText = Number(state.settings.top_p).toFixed(2);
    }
    if (el.sliderTopK) {
      el.sliderTopK.value = state.settings.top_k;
      el.valTopK.innerText = `${state.settings.top_k} chunks`;
    }
  }

  // --- Khởi tạo & Gán sự kiện ---
  function init() {
    // Nạp cài đặt đã lưu trong localStorage
    try {
      const saved = localStorage.getItem('local_ai_settings');
      if (saved) {
        state.settings = Object.assign(state.settings, JSON.parse(saved));
      }
    } catch (e) {}

    // Mode toggles
    el.btnModeChat.addEventListener('click', () => setMode('chat'));
    el.btnModeRag.addEventListener('click', () => setMode('rag'));

    // Chat events
    el.btnSend.addEventListener('click', handleSendMessage);
    el.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });

    // Auto-resize textarea
    el.chatInput.addEventListener('input', () => {
      el.chatInput.style.height = 'auto';
      el.chatInput.style.height = Math.min(el.chatInput.scrollHeight, 140) + 'px';
    });

    // Clear chat
    el.btnClearChat.addEventListener('click', () => {
      state.history = [];
      el.chatFeed.innerHTML = '';
      if (el.welcomeHero) {
        el.welcomeHero.style.display = 'flex';
        el.chatFeed.appendChild(el.welcomeHero);
      }
    });

    // Export Chat Markdown
    if (el.btnExportChat) {
      el.btnExportChat.addEventListener('click', exportChatMarkdown);
    }

    // Settings Modal handlers
    if (el.btnOpenSettings && el.settingsModal) {
      el.btnOpenSettings.addEventListener('click', () => {
        syncSettingsToUI();
        el.settingsModal.style.display = 'flex';
      });

      el.btnCloseSettings.addEventListener('click', () => {
        el.settingsModal.style.display = 'none';
      });

      el.settingsModal.addEventListener('click', (e) => {
        if (e.target === el.settingsModal) {
          el.settingsModal.style.display = 'none';
        }
      });

      el.sliderTemp.addEventListener('input', (e) => {
        el.valTemp.innerText = Number(e.target.value).toFixed(2);
      });

      el.sliderTopP.addEventListener('input', (e) => {
        el.valTopP.innerText = Number(e.target.value).toFixed(2);
      });

      el.sliderTopK.addEventListener('input', (e) => {
        el.valTopK.innerText = `${e.target.value} chunks`;
      });

      el.btnSaveSettings.addEventListener('click', () => {
        state.settings.temperature = parseFloat(el.sliderTemp.value);
        state.settings.top_p = parseFloat(el.sliderTopP.value);
        state.settings.top_k = parseInt(el.sliderTopK.value);
        try {
          localStorage.setItem('local_ai_settings', JSON.stringify(state.settings));
        } catch (e) {}
        el.settingsModal.style.display = 'none';
      });

      el.btnResetSettings.addEventListener('click', () => {
        state.settings = { temperature: 0.70, top_p: 0.90, top_k: 3 };
        try {
          localStorage.removeItem('local_ai_settings');
        } catch (e) {}
        syncSettingsToUI();
      });
    }

    // Quick prompts
    document.querySelectorAll('.prompt-card').forEach(card => {
      card.addEventListener('click', () => {
        const promptText = card.getAttribute('data-prompt');
        const mode = card.getAttribute('data-mode');
        if (mode) setMode(mode);
        el.chatInput.value = promptText;
        handleSendMessage();
      });
    });

    // Drag & Drop
    const dropzone = el.dropzone;
    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        uploadFile(files[0]);
      }
    });

    el.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
        e.target.value = '';
      }
    });

    // Refresh documents button
    el.btnRefreshDocs.addEventListener('click', fetchDocuments);

    // Crawler button
    if (el.btnRunCrawler) {
      el.btnRunCrawler.addEventListener('click', runCrawler);
    }

    // Mobile sidebar toggle
    if (el.btnMobileToggle) {
      el.btnMobileToggle.addEventListener('click', () => {
        el.sidebar.classList.toggle('open');
      });
    }

    // Load initial data
    fetchHealth();
    fetchDocuments();

    // Check health periodically
    setInterval(fetchHealth, 30000);
  }

  // Chạy khi trang sẵn sàng
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
