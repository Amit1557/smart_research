/**
 * SmartDoc AI - Multi-Document Intelligence & 1-Click Executive Summary (v2.1)
 */

document.addEventListener("DOMContentLoaded", () => {
    // ─────────────────────────────────────────────────────────────────────────
    // DOM ELEMENTS
    // ─────────────────────────────────────────────────────────────────────────
    // Topbar Controls
    const btnTopbarSummary = document.getElementById("btn-topbar-summary");
    const btnThemeToggle = document.getElementById("btn-theme-toggle");
    const btnFullscreenToggle = document.getElementById("btn-fullscreen-toggle");
    const btnClearChat = document.getElementById("btn-clear-chat");
    const btnSettingsToggle = document.getElementById("btn-settings-toggle");
    const topbarDocPill = document.getElementById("topbar-doc-pill");
    const topbarDocName = document.getElementById("topbar-doc-name");
    const topbarDocBadge = document.getElementById("topbar-doc-badge");

    // File Upload Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const btnBrowseFile = document.getElementById("btn-browse-file");
    const btnAttachFile = document.getElementById("btn-attach-file");
    const uploadProgressCard = document.getElementById("upload-progress-card");
    const uploadStatusText = document.getElementById("upload-status-text");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const progressStepName = document.getElementById("progress-step-name");

    // Document Library Elements
    const docLibraryList = document.getElementById("doc-library-list");
    const libCountBadge = document.getElementById("lib-count-badge");
    const btnLibSummarizeAll = document.getElementById("btn-lib-summarize-all");
    const emptyLibraryMsg = document.getElementById("empty-library-msg");

    // Accordion & Manual Paste Elements
    const btnTogglePaste = document.getElementById("btn-toggle-paste");
    const pasteSection = document.getElementById("paste-section");
    const manualDocTitle = document.getElementById("manual-doc-title");
    const manualDocText = document.getElementById("manual-doc-text");
    const btnManualIndex = document.getElementById("btn-manual-index");
    const presetPolicy = document.getElementById("preset-policy");
    const presetResearch = document.getElementById("preset-research");
    const presetWarranty = document.getElementById("preset-warranty");

    // Chat Workspace Elements
    const suggestedContainer = document.getElementById("suggested-container");
    const suggestedChipsList = document.getElementById("suggested-chips-list");
    const chipQuickSummary = document.getElementById("chip-quick-summary");
    const messagesContainer = document.getElementById("messages-container");
    const welcomeHero = document.getElementById("welcome-hero");
    const btnHeroSample = document.getElementById("btn-hero-sample");
    const heroCardUpload = document.getElementById("hero-card-upload");
    const heroCardSummary = document.getElementById("hero-card-summary");
    const chatTypingIndicator = document.getElementById("chat-typing-indicator");
    const typingStatusText = document.getElementById("typing-status-text");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const btnSendMessage = document.getElementById("btn-send-message");

    // Settings Modal Elements
    const settingsModal = document.getElementById("settings-modal");
    const btnCloseSettings = document.getElementById("btn-close-settings");
    const settingApiKey = document.getElementById("setting-api-key");
    const btnToggleKeyVisibility = document.getElementById("btn-toggle-key-visibility");
    const apiKeyStatusBadge = document.getElementById("api-key-status-badge");
    const settingModelSelect = document.getElementById("setting-model-select");
    const settingTopK = document.getElementById("setting-top-k");
    const topKVal = document.getElementById("top-k-val");
    const btnModalResetDb = document.getElementById("btn-modal-reset-db");
    const btnSaveSettings = document.getElementById("btn-save-settings");

    // Toast Container
    const toastContainer = document.getElementById("toast-container");

    // ─────────────────────────────────────────────────────────────────────────
    // STATE
    // ─────────────────────────────────────────────────────────────────────────
    let documentsList = [];
    let isProcessing = false;

    const SAMPLES = {
        policy: {
            title: "NexusTech Employee Handbook 2026",
            text: `NexusTech Enterprise Human Resources Policy & Perks Guide (2026 Edition)

1. Annual Leave & Work From Home (WFH):
All full-time employees receive 25 paid annual leaves (vacation days) per calendar year. In addition, employees are entitled to 10 sick leaves and 12 public holidays. NexusTech operates on a hybrid-first model: team members may work remotely up to 3 days per week with manager coordination. Core collaboration hours are 10:00 AM to 4:00 PM local time.

2. Comprehensive Health & Wellness Benefits:
NexusTech provides 100% employer-sponsored health, dental, and vision insurance for employees and 80% coverage for eligible dependents. Every employee receives an annual $1,200 Wellness & Mental Health stipend, which can be applied toward gym memberships, ergonomic home equipment, or licensed therapy sessions.

3. Professional Learning & Upskilling Stipend:
Each team member has access to an annual $2,000 professional education budget for approved technical certifications (e.g. AWS, GCP, PyTorch), books, and conferences. Courses must align with the employee's career progression plan.

4. Notice Period & Resignation Protocol:
Employees in individual contributor roles must provide a minimum of 30 days written notice before their final working day. Engineering leads, managers, and executives are subject to a 60-day notice period to ensure seamless knowledge transfer and project handover. During the notice period, all company hardware and access tokens must be surrendered to IT.`
        },
        research: {
            title: "Semantic Vector Geometry & Transformer Models",
            text: `High-Dimensional Semantic Geometry in Transformer Language Models (Research Abstract)

Modern transformer architectures employ multi-head self-attention mechanisms to map discrete language sequences into continuous mathematical vector manifolds. In this evaluation, we assess dense sentence representations generated by MiniLM architectures (384 dimensions) compared to classical bag-of-words and lexical search.

Key Research Findings:
1. Cosine similarity computed over normalized 384-dimensional dense vectors consistently outperforms lexical matching on semantic paraphrase benchmarks, achieving a 24.8% increase in Top-3 retrieval recall.
2. Embedding normalization constrains the inner dot product strictly between [-1.0, 1.0], enabling high-performance sub-millisecond similarity indexing when utilizing inverted vector indexes like HNSW in Qdrant.
3. When retrieved vector chunks are provided as grounded reference context to Google Gemini, factual hallucinations decrease by 87.4% compared to standard ungrounded prompting across complex technical queries.`
        },
        warranty: {
            title: "AeroSound Pro Headphone Warranty & Support",
            text: `AeroSound Pro Wireless ANC Headphones - Official Warranty & Return Guidelines

Warranty Duration & Coverage:
The AeroSound Pro comes with a 2-Year Limited Manufacturer Warranty covering manufacturing defects, internal component failure, and battery degradation exceeding 30% within 24 months. Accidental liquid submersion beyond IPX4 rating and third-party modifications are not covered.

Replacement & RMA Process:
To file an RMA claim, submit proof of purchase and serial number to support@aerosound.com. If verified, a prepaid return shipping label will be provided within 48 hours. Replacement units are dispatched within 3-5 business days upon receiving the defective unit.

30-Day Money-Back Guarantee:
Customers who purchase directly through the official website may return the product within 30 days for a full refund, provided all original packaging, cables, and ear-cushion accessories are intact.`
        }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // THEME MANAGEMENT (LIGHT / DARK)
    // ─────────────────────────────────────────────────────────────────────────
    function initTheme() {
        const savedTheme = localStorage.getItem("smartdoc_theme") || "dark";
        document.documentElement.setAttribute("data-theme", savedTheme);
    }

    btnThemeToggle.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("smartdoc_theme", newTheme);
        showToast(`Switched to ${newTheme.toUpperCase()} theme`, "info");
    });

    initTheme();

    // ─────────────────────────────────────────────────────────────────────────
    // FULLSCREEN TOGGLE
    // ─────────────────────────────────────────────────────────────────────────
    btnFullscreenToggle.addEventListener("click", () => {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(() => {});
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen().catch(() => {});
            }
        }
    });

    // ─────────────────────────────────────────────────────────────────────────
    // TOAST NOTIFICATIONS
    // ─────────────────────────────────────────────────────────────────────────
    function showToast(message, type = "success") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        
        let icon = "✓";
        if (type === "error") icon = "✕";
        if (type === "info") icon = "✦";

        toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(10px) scale(0.95)";
            setTimeout(() => toast.remove(), 250);
        }, 3200);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MULTI-DOCUMENT LIBRARY SYNC & RENDERING
    // ─────────────────────────────────────────────────────────────────────────
    async function loadDocumentsLibrary() {
        try {
            const res = await fetch("/api/documents");
            if (res.ok) {
                documentsList = await res.json();
                renderLibraryUI();
            }
        } catch (e) {
            console.error("Error loading documents:", e);
        }
    }

    function renderLibraryUI() {
        libCountBadge.textContent = documentsList.length;

        // Update Topbar Status Pill
        if (documentsList.length === 0) {
            topbarDocPill.classList.remove("active");
            topbarDocPill.classList.add("empty");
            topbarDocName.textContent = "No documents loaded";
            topbarDocBadge.classList.add("hidden");
            btnLibSummarizeAll.classList.add("hidden");
            suggestedContainer.classList.add("hidden");
            docLibraryList.innerHTML = `
                <div id="empty-library-msg" class="empty-library-msg">
                    <span class="empty-lib-icon">📁</span>
                    <p>No documents uploaded yet.<br>Drop files above to build your research library.</p>
                </div>
            `;
            return;
        }

        const totalChunks = documentsList.reduce((acc, d) => acc + (d.chunks_count || 0), 0);
        topbarDocPill.classList.remove("empty");
        topbarDocPill.classList.add("active");
        topbarDocName.textContent = documentsList.length === 1 ? documentsList[0].filename : `${documentsList.length} Documents Active`;
        topbarDocBadge.textContent = `${totalChunks} chunks`;
        topbarDocBadge.classList.remove("hidden");
        btnLibSummarizeAll.classList.remove("hidden");

        // Render Cards in Sidebar
        docLibraryList.innerHTML = "";
        documentsList.forEach((doc) => {
            let icon = "📄";
            const type = (doc.filetype || "").toUpperCase();
            if (type.includes("PDF")) icon = "📕";
            else if (type.includes("DOC")) icon = "📘";
            else if (type.includes("TXT") || type.includes("MD")) icon = "📝";

            const card = document.createElement("div");
            card.className = "doc-item-card";
            card.innerHTML = `
                <div class="doc-item-top">
                    <span class="doc-item-icon">${icon}</span>
                    <div class="doc-item-info">
                        <div class="doc-item-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                        <div class="doc-item-meta">${doc.filesize_formatted} • ${doc.chunks_count} chunks • ${doc.page_count} pgs</div>
                    </div>
                </div>
                <div class="doc-item-actions">
                    <button type="button" class="btn-card-summary" title="Generate Executive Summary for this document">
                        <span>⚡</span>
                        <span>Summary</span>
                    </button>
                    <button type="button" class="btn-card-delete" title="Delete document from vector memory">
                        🗑️
                    </button>
                </div>
            `;

            // Action: Summarize specific document
            card.querySelector(".btn-card-summary").addEventListener("click", () => {
                triggerExecutiveSummary(doc.doc_id, doc.filename);
            });

            // Action: Delete document
            card.querySelector(".btn-card-delete").addEventListener("click", async () => {
                if (confirm(`Remove "${doc.filename}" from your research library?`)) {
                    await deleteDocument(doc.doc_id, doc.filename);
                }
            });

            docLibraryList.appendChild(card);
        });

        // Update suggested questions chips from all documents
        updateSuggestedChips();
    }

    function updateSuggestedChips() {
        const allQuestions = [];
        documentsList.forEach(d => {
            if (d.suggested_questions) {
                d.suggested_questions.forEach(q => {
                    if (!allQuestions.includes(q)) allQuestions.push(q);
                });
            }
        });

        suggestedChipsList.innerHTML = "";
        allQuestions.slice(0, 4).forEach(q => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "chip-question";
            btn.textContent = q;
            btn.addEventListener("click", () => {
                userInput.value = q;
                handleUserQuestion(q);
            });
            suggestedChipsList.appendChild(btn);
        });

        suggestedContainer.classList.remove("hidden");
    }

    async function deleteDocument(docId, filename) {
        try {
            const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
            if (res.ok) {
                showToast(`Removed "${filename}"`, "info");
                await loadDocumentsLibrary();
            } else {
                showToast("Failed to delete document", "error");
            }
        } catch (e) {
            showToast(e.message, "error");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LAPTOP FILE UPLOAD (MULTI-FILE SUPPORT)
    // ─────────────────────────────────────────────────────────────────────────
    btnBrowseFile.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    btnAttachFile.addEventListener("click", () => {
        fileInput.click();
    });

    heroCardUpload.addEventListener("click", () => {
        fileInput.click();
    });

    dropZone.addEventListener("click", () => fileInput.click());

    ["dragenter", "dragover"].forEach(evtName => {
        dropZone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evtName => {
        dropZone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleMultipleFilesUpload(Array.from(files));
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleMultipleFilesUpload(Array.from(fileInput.files));
            fileInput.value = "";
        }
    });

    async function handleMultipleFilesUpload(files) {
        if (!files || files.length === 0) return;

        uploadProgressCard.classList.remove("hidden");
        let successful = 0;

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const percent = Math.round(((i + 0.3) / files.length) * 100);
            progressBarFill.style.width = `${percent}%`;
            uploadStatusText.textContent = `Uploading ${file.name} (${i + 1}/${files.length})...`;
            progressStepName.textContent = "Extracting text & PyTorch 384-D Vectorizing";

            const formData = new FormData();
            formData.append("file", file);
            formData.append("chunk_size", localStorage.getItem("smartdoc_chunk_size") || "100");
            formData.append("chunk_overlap", "20");

            try {
                const response = await fetch("/api/upload", {
                    method: "POST",
                    body: formData,
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.detail || `Failed to index ${file.name}`);
                }

                successful++;
            } catch (err) {
                showToast(err.message, "error");
            }
        }

        progressBarFill.style.width = "100%";
        uploadStatusText.textContent = `Completed indexing ${successful} file(s)!`;
        progressStepName.textContent = "Qdrant Vector Database Updated";

        setTimeout(async () => {
            uploadProgressCard.classList.add("hidden");
            await loadDocumentsLibrary();
            showToast(`Added ${successful} document(s) to research library!`, "success");
            
            // If first upload, display friendly assistant greeting
            if (messagesContainer.querySelectorAll(".chat-row").length === 0) {
                welcomeHero.classList.add("hidden");
                appendAIMessage({
                    answer: `### 📚 Research Knowledge Base Updated\n\nI have indexed **${successful} file(s)** into the **Qdrant Vector Database** using **384-dimensional PyTorch embeddings**.\n\nYou can now:\n* Click **⚡ 1-Click Summary** to generate an executive overview.\n* Ask questions comparing or extracting details across your documents.\n* Select any suggested question pill above!`,
                    model: "SmartDoc AI",
                    ml_metrics: { latency_ms: 0, highest_similarity_score: 100 },
                    retrieved_sources: [],
                });
            }
        }, 400);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ⚡ 1-CLICK EXECUTIVE SUMMARY & KEY TAKEAWAYS GENERATOR
    // ─────────────────────────────────────────────────────────────────────────
    btnTopbarSummary.addEventListener("click", () => triggerExecutiveSummary(null, "All Documents"));
    btnLibSummarizeAll.addEventListener("click", () => triggerExecutiveSummary(null, "All Documents"));
    chipQuickSummary.addEventListener("click", () => triggerExecutiveSummary(null, "All Documents"));
    heroCardSummary.addEventListener("click", () => triggerExecutiveSummary(null, "All Documents"));

    async function triggerExecutiveSummary(docId = null, targetName = "Library Documents") {
        if (isProcessing) return;
        if (documentsList.length === 0) {
            showToast("Please upload at least one document from your laptop first!", "error");
            dropZone.classList.add("dragover");
            setTimeout(() => dropZone.classList.remove("dragover"), 800);
            return;
        }

        welcomeHero.classList.add("hidden");

        // 1. Append user request prompt
        const promptLabel = docId ? `⚡ Generate Executive Summary & Key Takeaways for "${targetName}"` : "⚡ Generate Executive Summary & Key Takeaways for All Documents";
        appendUserMessage(promptLabel);

        // 2. Show typing indicator
        isProcessing = true;
        btnSendMessage.disabled = true;
        chatTypingIndicator.classList.remove("hidden");
        typingStatusText.textContent = `Synthesizing Executive Summary with Google Gemini for ${targetName}...`;
        scrollToBottom();

        try {
            const chosenModel = localStorage.getItem("smartdoc_model") || null;
            const customKey = localStorage.getItem("smartdoc_api_key") || null;

            const res = await fetch("/api/summarize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    doc_id: docId,
                    model: chosenModel,
                    api_key: customKey,
                })
            });

            const data = await res.json();
            chatTypingIndicator.classList.add("hidden");
            isProcessing = false;
            btnSendMessage.disabled = false;

            if (!res.ok) {
                throw new Error(data.detail || "Failed to generate summary.");
            }

            // 3. Append rich summary card in chat
            appendAIMessage({
                answer: data.summary,
                model: `${chosenModel || "Gemini 2.5 Flash"} • Executive Summary`,
                ml_metrics: {
                    latency_ms: data.latency_ms || 0,
                    highest_similarity_score: 100,
                },
                retrieved_sources: [],
                isSummary: true,
            });

        } catch (error) {
            chatTypingIndicator.classList.add("hidden");
            isProcessing = false;
            btnSendMessage.disabled = false;
            appendAIMessage({
                answer: `⚠️ **Executive Summary Error**\n\n${error.message}\n\n*Tip: Ensure your Gemini API Key is configured in Settings (⚙️).*`,
                model: "Error",
                ml_metrics: { latency_ms: 0, highest_similarity_score: 0 },
                retrieved_sources: [],
            });
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MANUAL RAW TEXT / PRESETS
    // ─────────────────────────────────────────────────────────────────────────
    btnTogglePaste.addEventListener("click", () => {
        pasteSection.classList.toggle("hidden");
        btnTogglePaste.parentElement.classList.toggle("open");
    });

    presetPolicy.addEventListener("click", () => {
        manualDocTitle.value = SAMPLES.policy.title;
        manualDocText.value = SAMPLES.policy.text;
    });

    presetResearch.addEventListener("click", () => {
        manualDocTitle.value = SAMPLES.research.title;
        manualDocText.value = SAMPLES.research.text;
    });

    presetWarranty.addEventListener("click", () => {
        manualDocTitle.value = SAMPLES.warranty.title;
        manualDocText.value = SAMPLES.warranty.text;
    });

    btnHeroSample.addEventListener("click", () => {
        loadPresetSample("research");
    });

    async function loadPresetSample(sampleKey) {
        const sample = SAMPLES[sampleKey] || SAMPLES.research;
        try {
            showToast("Indexing sample research document...", "info");
            const res = await fetch("/api/index", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    document_title: sample.title,
                    text: sample.text,
                    chunk_size: 90,
                    chunk_overlap: 20
                })
            });
            const data = await res.json();
            if (res.ok) {
                await loadDocumentsLibrary();
                showToast(`Loaded "${sample.title}"`, "success");
            } else {
                showToast(data.detail || "Error loading preset", "error");
            }
        } catch (e) {
            showToast(e.message, "error");
        }
    }

    btnManualIndex.addEventListener("click", async () => {
        const title = manualDocTitle.value.trim() || "Pasted Document";
        const text = manualDocText.value.trim();
        if (!text) {
            showToast("Please paste some text content first.", "error");
            return;
        }

        try {
            btnManualIndex.disabled = true;
            btnManualIndex.textContent = "Embedding & Indexing...";
            const res = await fetch("/api/index", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    document_title: title,
                    text: text,
                    chunk_size: 90,
                    chunk_overlap: 20
                })
            });
            const data = await res.json();
            btnManualIndex.disabled = false;
            btnManualIndex.innerHTML = "<span>Embed & Add to Library</span>";

            if (res.ok) {
                pasteSection.classList.add("hidden");
                btnTogglePaste.parentElement.classList.remove("open");
                await loadDocumentsLibrary();
                showToast(`Added "${title}" to library!`, "success");
            } else {
                showToast(data.detail || "Failed to index text.", "error");
            }
        } catch (e) {
            btnManualIndex.disabled = false;
            btnManualIndex.innerHTML = "<span>Embed & Add to Library</span>";
            showToast(e.message, "error");
        }
    });

    // ─────────────────────────────────────────────────────────────────────────
    // CHAT & MULTI-DOCUMENT QUESTION ANSWERING
    // ─────────────────────────────────────────────────────────────────────────
    userInput.addEventListener("input", () => {
        userInput.style.height = "auto";
        userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
    });

    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;
        handleUserQuestion(query);
    });

    async function handleUserQuestion(question) {
        if (isProcessing) return;

        if (documentsList.length === 0) {
            showToast("Please upload a document from your laptop first!", "error");
            dropZone.classList.add("dragover");
            setTimeout(() => dropZone.classList.remove("dragover"), 800);
            return;
        }

        userInput.value = "";
        userInput.style.height = "auto";
        welcomeHero.classList.add("hidden");

        appendUserMessage(question);

        isProcessing = true;
        btnSendMessage.disabled = true;
        chatTypingIndicator.classList.remove("hidden");
        typingStatusText.textContent = "Searching Qdrant across all documents ➔ Grounding with Gemini...";
        scrollToBottom();

        try {
            const topK = parseInt(localStorage.getItem("smartdoc_top_k") || "4", 10);
            const chosenModel = localStorage.getItem("smartdoc_model") || null;
            const customKey = localStorage.getItem("smartdoc_api_key") || null;

            const response = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    top_k: topK,
                    model: chosenModel,
                    api_key: customKey,
                })
            });

            const data = await response.json();
            chatTypingIndicator.classList.add("hidden");
            isProcessing = false;
            btnSendMessage.disabled = false;

            if (!response.ok) {
                throw new Error(data.detail || "Error querying documents.");
            }

            appendAIMessage({
                answer: data.answer,
                model: chosenModel || "Google Gemini",
                ml_metrics: data.ml_metrics,
                retrieved_sources: data.retrieved_sources || [],
            });

        } catch (error) {
            chatTypingIndicator.classList.add("hidden");
            isProcessing = false;
            btnSendMessage.disabled = false;
            appendAIMessage({
                answer: `⚠️ **Error Processing Query**\n\n${error.message}\n\n*Tip: Check if your Gemini API key is configured properly in Settings (⚙️).*`,
                model: "Error",
                ml_metrics: { latency_ms: 0, highest_similarity_score: 0 },
                retrieved_sources: [],
            });
        }
    }

    function appendUserMessage(text) {
        const row = document.createElement("div");
        row.className = "chat-row user-row";
        row.innerHTML = `
            <div class="chat-bubble">
                <div class="user-text">${escapeHtml(text)}</div>
            </div>
            <div class="chat-avatar user-avatar">👤</div>
        `;
        messagesContainer.appendChild(row);
        scrollToBottom();
    }

    function appendAIMessage({ answer, model, ml_metrics, retrieved_sources, isSummary = false }) {
        const row = document.createElement("div");
        row.className = "chat-row ai-row";

        let formattedHtml = answer;
        if (window.marked && typeof window.marked.parse === "function") {
            formattedHtml = window.marked.parse(answer);
        }

        const matchScore = ml_metrics && ml_metrics.highest_similarity_score ? `${ml_metrics.highest_similarity_score}%` : "N/A";
        const latency = ml_metrics && ml_metrics.latency_ms ? `${ml_metrics.latency_ms} ms` : "0 ms";

        let citationsHtml = "";
        if (retrieved_sources && retrieved_sources.length > 0) {
            const sourcesCards = retrieved_sources.map(src => {
                const pageLabel = src.page ? `Page ${src.page}` : `Chunk #${src.chunk_id}`;
                return `
                    <div class="citation-card">
                        <div class="citation-card-header">
                            <span class="citation-source-name">[Source ${src.rank}] ${escapeHtml(src.document)} (${pageLabel})</span>
                            <span class="citation-score">Cosine Match: ${src.similarity_score}%</span>
                        </div>
                        <div class="citation-card-body">${escapeHtml(src.text)}</div>
                    </div>
                `;
            }).join("");

            citationsHtml = `
                <div class="citations-tray">
                    <button type="button" class="citations-toggle">
                        <div class="citations-toggle-left">
                            <span>📚</span>
                            <span>${retrieved_sources.length} Cross-Document Sources</span>
                            <span class="citations-match-badge">Match: ${matchScore}</span>
                        </div>
                        <span class="accordion-arrow">▼</span>
                    </button>
                    <div class="citations-list hidden">
                        ${sourcesCards}
                    </div>
                </div>
            `;
        }

        row.innerHTML = `
            <div class="chat-avatar ai-avatar ${isSummary ? 'summary-mode' : ''}">✦</div>
            <div class="chat-bubble">
                <div class="ai-meta-header">
                    <div class="ai-meta-left">
                        <span class="ai-model-tag ${isSummary ? 'summary-tag' : ''}">${escapeHtml(model)}</span>
                        <span class="latency-badge">${latency}</span>
                    </div>
                    <div class="ai-meta-right">
                        <button type="button" class="btn-copy-answer" title="Copy response to clipboard">
                            <span>📋</span>
                            <span>Copy</span>
                        </button>
                    </div>
                </div>
                <div class="markdown-body">${formattedHtml}</div>
                ${citationsHtml}
            </div>
        `;

        const toggleBtn = row.querySelector(".citations-toggle");
        if (toggleBtn) {
            toggleBtn.addEventListener("click", () => {
                const list = row.querySelector(".citations-list");
                const arrow = toggleBtn.querySelector(".accordion-arrow");
                list.classList.toggle("hidden");
                arrow.style.transform = list.classList.contains("hidden") ? "rotate(0deg)" : "rotate(180deg)";
            });
        }

        const copyBtn = row.querySelector(".btn-copy-answer");
        if (copyBtn) {
            copyBtn.addEventListener("click", () => {
                navigator.clipboard.writeText(answer).then(() => {
                    copyBtn.innerHTML = `<span>✓</span> <span>Copied!</span>`;
                    setTimeout(() => {
                        copyBtn.innerHTML = `<span>📋</span> <span>Copy</span>`;
                    }, 2000);
                }).catch(() => {});
            });
        }

        messagesContainer.appendChild(row);
        scrollToBottom();
    }

    function scrollToBottom() {
        setTimeout(() => {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }, 50);
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CLEAR CONVERSATION
    // ─────────────────────────────────────────────────────────────────────────
    btnClearChat.addEventListener("click", () => {
        messagesContainer.innerHTML = "";
        messagesContainer.appendChild(welcomeHero);
        welcomeHero.classList.remove("hidden");
        showToast("Conversation cleared", "info");
    });

    // ─────────────────────────────────────────────────────────────────────────
    // SETTINGS MODAL
    // ─────────────────────────────────────────────────────────────────────────
    btnSettingsToggle.addEventListener("click", async () => {
        settingsModal.classList.remove("hidden");
        try {
            const res = await fetch("/api/settings");
            if (res.ok) {
                const data = await res.json();
                if (data.api_key_configured) {
                    apiKeyStatusBadge.textContent = "Configured";
                    apiKeyStatusBadge.className = "badge-status-pill";
                    settingApiKey.placeholder = `Current: ${data.api_key_masked}`;
                } else {
                    apiKeyStatusBadge.textContent = "Missing";
                    apiKeyStatusBadge.className = "badge-status-pill missing";
                }
                
                const savedModel = localStorage.getItem("smartdoc_model") || data.current_model;
                if (savedModel && settingModelSelect.querySelector(`option[value="${savedModel}"]`)) {
                    settingModelSelect.value = savedModel;
                }
            }
        } catch (e) {}

        const savedTopK = localStorage.getItem("smartdoc_top_k") || "4";
        settingTopK.value = savedTopK;
        topKVal.textContent = savedTopK;
    });

    btnCloseSettings.addEventListener("click", () => {
        settingsModal.classList.add("hidden");
    });

    settingsModal.addEventListener("click", (e) => {
        if (e.target === settingsModal) {
            settingsModal.classList.add("hidden");
        }
    });

    btnToggleKeyVisibility.addEventListener("click", () => {
        if (settingApiKey.type === "password") {
            settingApiKey.type = "text";
            btnToggleKeyVisibility.textContent = "🔒";
        } else {
            settingApiKey.type = "password";
            btnToggleKeyVisibility.textContent = "👁️";
        }
    });

    settingTopK.addEventListener("input", () => {
        topKVal.textContent = settingTopK.value;
    });

    btnSaveSettings.addEventListener("click", async () => {
        const newKey = settingApiKey.value.trim();
        const selectedModel = settingModelSelect.value;
        const newTopK = settingTopK.value;

        localStorage.setItem("smartdoc_model", selectedModel);
        localStorage.setItem("smartdoc_top_k", newTopK);

        const payload = { gemini_model: selectedModel };
        if (newKey) {
            payload.gemini_api_key = newKey;
            localStorage.setItem("smartdoc_api_key", newKey);
        }

        try {
            const res = await fetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (res.ok) {
                showToast("Settings applied successfully!", "success");
                settingsModal.classList.add("hidden");
            } else {
                showToast("Failed to save settings.", "error");
            }
        } catch (e) {
            showToast("Settings saved locally.", "info");
            settingsModal.classList.add("hidden");
        }
    });

    btnModalResetDb.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to reset all document vectors in memory?")) return;
        try {
            const res = await fetch("/api/reset", { method: "POST" });
            if (res.ok) {
                documentsList = [];
                renderLibraryUI();
                messagesContainer.innerHTML = "";
                welcomeHero.classList.remove("hidden");
                messagesContainer.appendChild(welcomeHero);
                settingsModal.classList.add("hidden");
                showToast("Vector Database & Library reset.", "success");
            }
        } catch (e) {
            showToast(e.message, "error");
        }
    });

    // ─────────────────────────────────────────────────────────────────────────
    // INITIALIZATION
    // ─────────────────────────────────────────────────────────────────────────
    loadDocumentsLibrary();
});
