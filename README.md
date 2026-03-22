<div align="center">
  <img src="electron/assets/icon-512.png" width="96" alt="Smart Search logo"/>

  # Smart Search

  > A local RAG (Retrieval-Augmented Generation) desktop app that semantically searches your PDF and DOCX files and generates AI-powered answers — packaged as a native macOS app. Indexing and search run entirely on your machine — only the final AI summary is sent to Groq's cloud API.

  ![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
  ![Electron](https://img.shields.io/badge/Electron-33-47848F)
  ![Flask](https://img.shields.io/badge/Flask-backend-lightgrey)
  ![FAISS](https://img.shields.io/badge/FAISS-vector_search-orange)
  ![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20llama--3.1--8b-green)
  ![License](https://img.shields.io/badge/License-MIT-yellow)
</div>

---

## What It Does

1. You point it at folders containing PDF and DOCX files
2. It chunks and embeds the text using a local ONNX model (`all-MiniLM-L6-v2`, ~22 MB, no GPU needed)
3. Builds a FAISS vector index (semantic) **and** a BM25 keyword index
4. On each search, runs both indices and combines results via **Reciprocal Rank Fusion**
5. Sends the top-ranked chunks to Groq's free LLM and returns an AI summary + source file paths
6. All indexing and search happens locally on your Mac — only the top-matching text chunks are sent to Groq's cloud API to generate the AI summary

---

## Architecture

```
┌────────────────────────────────────────┐
│          Electron Shell (UI)           │
│  main.js → spawns Python sidecar      │
│  preload.js → native folder picker    │
└──────────────┬─────────────────────────┘
               │ HTTP (localhost)
┌──────────────▼─────────────────────────┐
│       Flask Backend (app.py)           │
│                                        │
│  embedder.py  →  fastembed ONNX        │
│  config.py    →  ~/.smartsearch/       │
│  memory_helper.py → system stats       │
│                                        │
│  FAISS + BM25 hybrid search            │
│  Groq API for LLM summaries            │
└────────────────────────────────────────┘
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.9+ | [python.org](https://python.org) or `brew install python` |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) or `brew install node` |
| Groq API key | — | [console.groq.com](https://console.groq.com) — free, no credit card |

> **macOS only.** The Electron shell and DMG packaging target macOS. The Flask backend alone works on Linux/Windows.

---

## Building from Source

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-search.git
cd smart-search
```

### 2. Set up the Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> On first index the ONNX embedding model (~22 MB) downloads automatically to `~/.cache/huggingface/` and is cached for all future runs.

### 3. Build the Python binary with PyInstaller

```bash
source venv/bin/activate
pip install pyinstaller
pyinstaller smartsearch.spec --noconfirm
```

This produces `dist/smartsearch` — a self-contained ~43 MB binary that includes Flask, FAISS, fastembed, and all dependencies.

### 4. Install Electron dependencies

```bash
cd electron
npm install
```

### 5. Build the DMG

```bash
npm run dist
```

The output is `electron/dist/Smart Search-1.0.0-arm64.dmg` (~137 MB).

---

## Installing the App

1. Open `electron/dist/Smart Search-1.0.0-arm64.dmg`
2. Drag **Smart Search** into your Applications folder
3. Launch it — it lives in your macOS menu bar (tray icon)
4. On first launch, click the tray icon → **Settings**
5. Paste your [Groq API key](https://console.groq.com) and add the folders you want to search
6. Go back to the main window and click **Index Files**

> **Gatekeeper warning:** Because the app is not notarized, macOS may block it on first open. Right-click the app → **Open** → **Open** to bypass this once.

---

## Development Mode (no DMG needed)

Run the Flask backend and Electron shell separately:

```bash
# Terminal 1 — Python backend
source venv/bin/activate
python app.py
# Starts on http://127.0.0.1:5001

# Terminal 2 — Electron shell (points at the running backend)
cd electron
npm start
```

Electron detects the already-running backend and skips spawning its own Python process.

---

## Project Structure

```
smart-search/
├── app.py                  # Flask backend — indexing, search, Groq API, settings routes
├── embedder.py             # ONNX text embeddings via fastembed (replaces torch)
├── config.py               # Persistent config at ~/.smartsearch/config.json
├── memory_helper.py        # System memory / psutil utilities
├── smartsearch.spec        # PyInstaller build spec
├── requirements.txt        # Python dependencies
├── .gitignore
├── README.md
├── templates/
│   ├── index.html          # Main search UI (sidebar, search bar, results)
│   └── settings.html       # Settings page (API key, folders, model)
└── electron/
    ├── main.js             # Electron main process — spawn Python, tray, window
    ├── preload.js          # Context bridge — folder picker, isElectron flag
    ├── package.json        # electron-builder config, DMG target
    └── assets/
        ├── icon.icns       # macOS app icon (all sizes)
        ├── icon-512.png    # Source icon (512×512)
        ├── tray-icon.png   # Menu bar tray icon (16×16)
        └── dmg-bg.png      # DMG window background
```

**Not committed (auto-generated or user-specific):**

```
venv/                       # Python virtual environment
dist/                       # PyInstaller output
build/                      # PyInstaller build cache
electron/node_modules/      # Node dependencies
electron/dist/              # DMG output
faiss_index/                # Your personal document index
~/.smartsearch/             # Your config + API key (stored outside the repo)
```

---

## How It Works

### Indexing

```
Folders you chose in Settings
        │
        ▼
Filter PDF + DOCX  (skip venv/, .git/, node_modules/)
        │
        ▼
Extract text  (PyPDF2 / python-docx)
        │
        ▼
Chunk text  (256 words, 32-word overlap)
        │
        ▼
Embed chunks  (all-MiniLM-L6-v2 ONNX, 384-dim vectors)
        │
        ▼
Save FAISS index + BM25 index + metadata to disk
```

### Search

```
Your question
        │
        ▼
Embed query  (same ONNX model, 384-dim)
        │
        ├── FAISS L2 search → top-k semantic matches
        │
        └── BM25 keyword search → top-k keyword matches
                │
                ▼
        Reciprocal Rank Fusion → unified ranked list
                │
                ▼
        Top chunks sent to Groq API (llama-3.1-8b-instant)
                │
                ▼
        AI summary + source file paths returned to UI
```

---

## Configuration

All settings are stored in `~/.smartsearch/config.json` and managed through the in-app Settings page. Nothing is hardcoded.

| Setting | Default | Description |
|---------|---------|-------------|
| `groq_api_key` | _(set in Settings)_ | Your Groq API key |
| `groq_model` | `llama-3.1-8b-instant` | Groq model for AI summaries |
| `scan_folders` | `~/Documents`, `~/Downloads` | Folders scanned for documents |
| `max_files` | `100` | Cap on documents indexed per run |
| `chunk_size` | `256` | Words per chunk |
| `overlap` | `32` | Word overlap between adjacent chunks |
| `port` | `5001` | Flask port (auto-reassigned on conflict) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main search UI |
| `GET` | `/settings` | Settings page |
| `POST` | `/settings/save` | Save config |
| `POST` | `/settings/test-api-key` | Validate a Groq key |
| `POST` | `/index` | Trigger document indexing |
| `GET` | `/progress` | Indexing progress (polled by UI) |
| `GET` | `/index-stats` | Doc count and index size |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/search` | Hybrid search + Groq LLM response |
| `GET` | `/status` | Backend health check |

---

## Dependencies

### Python

| Package | Purpose |
|---------|---------|
| `flask` | Web server / API |
| `fastembed` | ONNX text embeddings (no GPU, no PyTorch) |
| `faiss-cpu` | Vector similarity search |
| `groq` | Groq LLM API |
| `PyPDF2` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `rank-bm25` | BM25 keyword index |
| `numpy` | Numerical operations |
| `psutil` | System memory monitoring |
| `python-dotenv` | `.env` file loading |

### Node / Electron

| Package | Purpose |
|---------|---------|
| `electron` | Desktop shell |
| `electron-builder` | DMG packaging |



---

## Why fastembed instead of PyTorch?

The original version used `sentence-transformers` + PyTorch (~1.2 GB installed). This was replaced with [fastembed](https://github.com/qdrant/fastembed), which runs the same `all-MiniLM-L6-v2` model via ONNX Runtime — no GPU, no PyTorch, ~80 MB installed. The final DMG is 137 MB instead of ~1.5 GB.

---

## License

MIT
