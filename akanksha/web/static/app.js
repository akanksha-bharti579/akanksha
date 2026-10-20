/**
 * Akanksha AI — Second Brain Frontend Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const statDocs = document.getElementById("stat-docs");
  const statChunks = document.getElementById("stat-chunks");
  const docList = document.getElementById("doc-list");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const uploadProgress = document.getElementById("upload-progress");
  const uploadStatusText = document.getElementById("upload-status-text");
  const refreshDocsBtn = document.getElementById("refresh-docs-btn");

  const chatContainer = document.getElementById("chat-container");
  const welcomeHero = document.getElementById("welcome-hero");
  const messageStream = document.getElementById("message-stream");
  const chatTextarea = document.getElementById("chat-textarea");
  const sendBtn = document.getElementById("send-btn");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  const quickAttachBtn = document.getElementById("quick-attach-btn");

  const activeProviderLabel = document.getElementById("active-provider-label");
  const openSettingsBtn = document.getElementById("open-settings-btn");
  const settingsModal = document.getElementById("settings-modal");
  const closeSettingsBtn = document.getElementById("close-settings-btn");
  const cancelSettingsBtn = document.getElementById("cancel-settings-btn");
  const saveSettingsBtn = document.getElementById("save-settings-btn");
  const providerSelect = document.getElementById("provider-select");
  const apiKeyGroup = document.getElementById("api-key-group");
  const apiKeyInput = document.getElementById("api-key-input");
  const apiKeyLabel = document.getElementById("api-key-label");
  const modelNameGroup = document.getElementById("model-name-group");
  const modelNameInput = document.getElementById("model-name-input");
  const resetIndexBtn = document.getElementById("reset-index-btn");

  // State
  let config = {
    provider: localStorage.getItem("akanksha_provider") || "offline",
    apiKey: localStorage.getItem("akanksha_api_key") || "",
    model: localStorage.getItem("akanksha_model") || "",
  };

  // Sync settings UI with state
  function updateSettingsUI() {
    providerSelect.value = config.provider;
    activeProviderLabel.textContent = getProviderDisplayName(config.provider);

    if (config.provider === "gemini") {
      apiKeyGroup.style.display = "flex";
      apiKeyLabel.textContent = "Google Gemini API Key";
      apiKeyInput.value = config.apiKey;
      modelNameGroup.style.display = "flex";
      modelNameInput.placeholder = "gemini-1.5-flash";
      modelNameInput.value = config.model || "gemini-1.5-flash";
    } else if (config.provider === "openai") {
      apiKeyGroup.style.display = "flex";
      apiKeyLabel.textContent = "OpenAI / API Key";
      apiKeyInput.value = config.apiKey;
      modelNameGroup.style.display = "flex";
      modelNameInput.placeholder = "gpt-4o-mini";
      modelNameInput.value = config.model || "gpt-4o-mini";
    } else if (config.provider === "ollama") {
      apiKeyGroup.style.display = "none";
      modelNameGroup.style.display = "flex";
      modelNameInput.placeholder = "llama3";
      modelNameInput.value = config.model || "llama3";
    } else {
      apiKeyGroup.style.display = "none";
      modelNameGroup.style.display = "none";
    }
  }

  function getProviderDisplayName(p) {
    switch (p) {
      case "gemini": return "Google Gemini";
      case "openai": return "OpenAI / Compatible";
      case "ollama": return "Local Ollama";
      default: return "Offline Extractive";
    }
  }

  updateSettingsUI();

  // Load Status and Documents
  async function loadStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) return;
      const data = await res.json();
      statDocs.textContent = data.documents_count || 0;
      statChunks.textContent = data.total_chunks || 0;
    } catch (err) {
      console.warn("Status fetch error:", err);
    }
  }

  async function loadDocuments() {
    try {
      const res = await fetch("/api/documents");
      if (!res.ok) return;
      const data = await res.json();
      renderDocumentList(data.documents || []);
    } catch (err) {
      console.warn("Docs fetch error:", err);
    }
  }

  function renderDocumentList(docs) {
    if (!docs.length) {
      docList.innerHTML = `
        <div class="empty-docs">
          <p>No documents indexed yet.</p>
          <span>Upload research papers or notes to start querying.</span>
        </div>`;
      return;
    }

    docList.innerHTML = "";
    docs.forEach(doc => {
      const item = document.createElement("div");
      item.className = "doc-item";
      item.innerHTML = `
        <div class="doc-item-left">
          <span class="doc-type-badge">${doc.file_type.slice(0, 4)}</span>
          <div class="doc-item-info">
            <span class="doc-item-name" title="${doc.title}">${doc.title}</span>
            <span class="doc-item-meta">${doc.total_chunks} chunks • ${doc.character_count.toLocaleString()} chars</span>
          </div>
        </div>
        <div class="doc-item-actions">
          <button class="doc-action-btn summarize-btn" title="Summarize document">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
            </svg>
          </button>
          <button class="doc-action-btn delete-btn" title="Delete from index">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
              <polyline points="3 6 5 6 21 6"/>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
          </button>
        </div>
      `;

      // Summarize trigger
      item.querySelector(".summarize-btn").addEventListener("click", () => {
        askQuestion(`Provide an executive summary and key findings of the document "${doc.title}".`);
      });

      // Delete trigger
      item.querySelector(".delete-btn").addEventListener("click", async () => {
        if (!confirm(`Remove "${doc.title}" from your Second Brain?`)) return;
        try {
          await fetch(`/api/documents/${encodeURIComponent(doc.title)}`, { method: "DELETE" });
          loadStatus();
          loadDocuments();
        } catch (err) {
          alert("Error deleting: " + err.message);
        }
      });

      docList.appendChild(item);
    });
  }

  // File Upload Handlers
  browseBtn.addEventListener("click", () => fileInput.click());
  quickAttachBtn.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
      handleFiles(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) {
      handleFiles(e.target.files);
      e.target.value = "";
    }
  });

  async function handleFiles(files) {
    uploadProgress.style.display = "flex";
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      uploadStatusText.textContent = `Indexing ${file.name}...`;

      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (data.error) {
          alert(`Error uploading ${file.name}: ${data.error}`);
        }
      } catch (err) {
        alert(`Failed to upload ${file.name}: ${err.message}`);
      }
    }

    uploadProgress.style.display = "none";
    await loadStatus();
    await loadDocuments();
  }

  refreshDocsBtn.addEventListener("click", () => {
    loadStatus();
    loadDocuments();
  });

  // Prompt Cards
  document.querySelectorAll(".prompt-card").forEach(card => {
    card.addEventListener("click", () => {
      const promptText = card.getAttribute("data-prompt");
      if (promptText) {
        askQuestion(promptText);
      }
    });
  });

  // Chat Execution
  async function askQuestion(text) {
    if (!text.trim()) return;

    // Switch view to message stream
    welcomeHero.style.display = "none";
    messageStream.style.display = "flex";

    // Append User Message
    appendMessage("user", text);
    chatTextarea.value = "";
    chatTextarea.style.height = "auto";

    // Show Assistant Loading Bubble
    const loadingRow = appendMessage("assistant", "Thinking and searching your knowledge base...");
    const loadingBubble = loadingRow.querySelector(".message-bubble");
    loadingBubble.classList.add("thinking");

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          provider: config.provider,
          api_key: config.apiKey,
          model: config.model,
        }),
      });

      const data = await res.json();
      loadingBubble.classList.remove("thinking");

      if (data.error) {
        loadingBubble.innerHTML = `<p style="color: var(--danger)">Error: ${data.error}</p>`;
        return;
      }

      // Render Markdown response
      loadingBubble.innerHTML = marked.parse(data.answer);

      // Render Source Citations if present
      if (data.sources && data.sources.length) {
        const citationsBox = document.createElement("div");
        citationsBox.className = "citations-box";
        citationsBox.innerHTML = `
          <span class="citations-header">Sources & Citations (${data.latency_ms}ms)</span>
          <div class="citations-list"></div>
        `;
        const listEl = citationsBox.querySelector(".citations-list");

        data.sources.forEach((src) => {
          const chip = document.createElement("div");
          chip.className = "citation-chip";
          const pageStr = src.page_number ? ` • p.${src.page_number}` : "";
          chip.innerHTML = `
            <span>📄 ${src.title}${pageStr}</span>
            <span class="citation-score">${(src.similarity_score * 100).toFixed(0)}%</span>
          `;
          chip.title = `"${src.snippet}"`;
          listEl.appendChild(chip);
        });

        loadingRow.querySelector(".message-body").appendChild(citationsBox);
      }
    } catch (err) {
      loadingBubble.classList.remove("thinking");
      loadingBubble.innerHTML = `<p style="color: var(--danger)">Network error: ${err.message}</p>`;
    }

    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function appendMessage(sender, content) {
    const row = document.createElement("div");
    row.className = `message-row ${sender}`;
    const avatar = sender === "user" ? "You" : "🧠";
    const senderName = sender === "user" ? "You" : "Akanksha AI";
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    row.innerHTML = `
      <div class="message-avatar">${avatar}</div>
      <div class="message-body">
        <div class="message-header">
          <span class="sender-name">${senderName}</span>
          <span class="message-time">${timeStr}</span>
        </div>
        <div class="message-bubble">${typeof marked !== "undefined" ? marked.parse(content) : content}</div>
      </div>
    `;

    messageStream.appendChild(row);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return row;
  }

  // Chat Input Handling
  chatTextarea.addEventListener("input", () => {
    chatTextarea.style.height = "auto";
    chatTextarea.style.height = Math.min(chatTextarea.scrollHeight, 160) + "px";
  });

  chatTextarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const val = chatTextarea.value.trim();
      if (val) askQuestion(val);
    }
  });

  sendBtn.addEventListener("click", () => {
    const val = chatTextarea.value.trim();
    if (val) askQuestion(val);
  });

  clearChatBtn.addEventListener("click", () => {
    messageStream.innerHTML = "";
    messageStream.style.display = "none";
    welcomeHero.style.display = "block";
  });

  // Settings Modal Handlers
  openSettingsBtn.addEventListener("click", () => {
    updateSettingsUI();
    settingsModal.style.display = "flex";
  });

  function closeSettings() {
    settingsModal.style.display = "none";
  }

  closeSettingsBtn.addEventListener("click", closeSettings);
  cancelSettingsBtn.addEventListener("click", closeSettings);

  providerSelect.addEventListener("change", (e) => {
    config.provider = e.target.value;
    updateSettingsUI();
  });

  saveSettingsBtn.addEventListener("click", () => {
    config.provider = providerSelect.value;
    config.apiKey = apiKeyInput.value.trim();
    config.model = modelNameInput.value.trim();

    localStorage.setItem("akanksha_provider", config.provider);
    localStorage.setItem("akanksha_api_key", config.apiKey);
    localStorage.setItem("akanksha_model", config.model);

    updateSettingsUI();
    closeSettings();
  });

  resetIndexBtn.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to wipe all indexed documents and vector chunks?")) return;
    try {
      const res = await fetch("/api/documents/ALL", { method: "DELETE" });
      loadStatus();
      loadDocuments();
      closeSettings();
    } catch (err) {
      alert("Error resetting: " + err.message);
    }
  });

  // Initial Load
  loadStatus();
  loadDocuments();
});
