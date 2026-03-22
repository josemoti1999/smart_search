# Smart Search

> A local RAG (Retrieval-Augmented Generation) system that semantically searches your PDF and DOCX documents and generates AI-powered answers — all running on your machine.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.1.3-lightgrey)
![FAISS](https://img.shields.io/badge/FAISS-1.13.0-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20llama--3.1--8b-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What It Does

1. Scans `~/Documents` and `~/Downloads` for PDF and DOCX files (up to 100)
2. Chunks and embeds the text using `sentence-transformers`
3. Stores vectors in a local FAISS index
4. On each search, retrieves the most similar chunks and sends them to Groq's free LLM
5. Returns an AI summary + the original matching chunks with source file paths

---

## UI Preview

### 1. First Launch — Nothing Indexed Yet

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │  Indexed Documents  │  │                                  │  │
│  │         0           │  │       🔍  Smart Search           │  │
│  │  ─────────────────  │  │                                  │  │
│  │                     │  │  ┌──────────────────────────┐    │  │
│  │   📂 No documents   │  │  │  ↻ Index Files           │    │  │
│  │      indexed yet.   │  │  └──────────────────────────┘    │  │
│  │   Click Index Files │  │                                  │  │
│  │      to start.      │  │  ┌──────────────────────────────┐│  │
│  │                     │  │  │ ⋮  Type your question...  🔍 ││  │
│  │                     │  │  └──────────────────────────────┘│  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 2. During Indexing — Progress Bar

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │  Indexed Documents  │  │       🔍  Smart Search           │  │
│  │         0           │  │                                  │  │
│  │                     │  │  ┌──────────────┐  ✦ Indexing…  │  │
│  │   📂 No documents   │  │  │ ↻ Indexing...│               │  │
│  │      indexed yet.   │  │  └──────────────┘               │  │
│  │                     │  │                                  │  │
│  │                     │  │  ████████████░░░░░░  8 / 11     │  │
│  │                     │  │  Indexing: 8 / 11 files          │  │
│  │                     │  │                                  │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

  ↑ Bar fills per file. Switches to "Encoding chunks…" phase
    at 100% while vectors are being computed.
```

---

### 3. After Indexing — Sidebar Populated

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │  Indexed Documents  │  │       🔍  Smart Search           │  │
│  │  ←          [11]    │  │                                  │  │
│  │  ─────────────────  │  │  ┌──────────────┐  ✅ 11 docs   │  │
│  │  ~/Documents/       │  │  │ ↻ Index Files│     0.8 MB    │  │
│  │  Resume_infos/      │  │  └──────────────┘               │  │
│  │  📄 resume.pdf   ✓  │  │                                  │  │
│  │  📄 doc2.pdf ✓ │  │  ┌──────────────────────────────┐│  │
│  │  📝 doc3.docx   ✓  │  │  │ ⋮  Type your question...  🔍 ││  │
│  │                     │  │  └──────────────────────────────┘│  │
│  │  ~/Documents/       │  │                                  │  │
│  │  doccs/ │  │                                  │  │
│  │  📄 forma.pdf  ✓  │  │                                  │  │
│  │  📄 formb.pdf  ✓  │  │                                  │  │
│  │  ─────────────────  │  │                                  │  │
│  │  11 docs | 0.8 MB   │  │                                  │  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

  ✓  Green tick on every document = successfully indexed
  ←  Click the arrow to collapse the sidebar
```

---

### 4. Sidebar Collapsed

```
┌─────────────────────────────────────────────────────────────────┐
│ →│         ┌──────────────────────────────────────────────────┐ │
│  │         │         🔍  Smart Search                         │ │
│  │         │                                                  │ │
│  │         │  ┌──────────────┐  ✅ 11 docs | 0.8 MB          │ │
│  │         │  │ ↻ Index Files│                                │ │
│  │         │  └──────────────┘                                │ │
│  │         │                                                  │ │
│  │         │  ┌──────────────────────────────────────────────┐│ │
│  │         │  │ ⋮  Type your question here...            🔍  ││ │
│  │         │  └──────────────────────────────────────────────┘│ │
│  │         └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

  →  Small tab on the left edge — click to re-open sidebar.
     Sidebar state is remembered across page refreshes.
```

---

### 5. Search Results

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │  Indexed Documents  │  │                                  │  │
│  │  ←          [11]    │  │  Results                         │  │
│  │  📄 resume.pdf   ✓  │  │  Time: 1.4s · Chunks: 5 · AI:0.8s│ │
│  │  📄 forma.pdf  ✓  │  │                                  │  │
│  │  ...                │  │  ┌──────────────────────────────┐│  │
│  │                     │  │  │ 💡 AI Summary                ││  │
│  │                     │  │  │  llama-3.1-8b-instant (Groq) ││  │
│  │                     │  │  │                              ││  │
│  │                     │  │  │  Based on your resume, you   ││  │
│  │                     │  │  │  have 4 years of experience  ││  │
│  │                     │  │  │  in data science at...       ││  │
│  │                     │  │  └──────────────────────────────┘│  │
│  │                     │  │                                  │  │
│  │                     │  │  MATCHING CHUNKS                 │  │
│  │                     │  │  ┌──────────────────────────────┐│  │
│  │                     │  │  │ 📄 resume_data_scientist.pdf ││  │
│  │                     │  │  │ ~/Documents/Resume_infos/... ││  │
│  │                     │  │  │                       92.3%  ││  │
│  │                     │  │  │ ─────────────────────────── ││  │
│  │                     │  │  │ "Led a team of 3 engineers   ││  │
│  │                     │  │  │  to build a real-time ML     ││  │
│  │                     │  │  │  pipeline processing 2M..."  ││  │
│  │                     │  │  └──────────────────────────────┘│  │
│  │                     │  │                                  │  │
│  │                     │  │  ┌──────────────────────────────┐│  │
│  │                     │  │  │ 📄 resume_ml.pdf             ││  │
│  │                     │  │  │ ~/Documents/Resume_infos/... ││  │
│  │                     │  │  │                       87.1%  ││  │
│  │                     │  │  │ ─────────────────────────── ││  │
│  │                     │  │  │ "Python, PyTorch, scikit-    ││  │
│  │                     │  │  │  learn, SQL, Spark, AWS..."  ││  │
│  │                     │  │  └──────────────────────────────┘│  │
│  └─────────────────────┘  └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 6. Filter by File Type

Click the **⋮** (three-dot) button inside the search bar to filter results:

```
  ┌──────────────────────────────────────┐
  │ ⋮  Type your question...        🔍  │
  └──┬───────────────────────────────────┘
     │  ┌──────────────────┐
     │  │ 📄  All Files    │  ← default
     │  │ 📝  Documents    │  ← .docx only
     │  │ 📑  PDFs         │  ← .pdf only
     │  └──────────────────┘
```

---

## Requirements

- Python 3.9+ (3.10+ recommended)
- macOS or Linux
- A free [Groq API key](https://console.groq.com)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-search.git
cd smart-search
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` embedding model (~90 MB) from HuggingFace. It is cached locally after the first download.

### 4. Configure your API key

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card required. Includes 14,400 requests/day free.

### 5. Add documents to scan

Place PDF or DOCX files anywhere inside:

| Folder | Scanned recursively |
|--------|-------------------|
| `~/Documents` | Yes |
| `~/Downloads` | Yes |

> Code files, `venv/`, `.git/`, `node_modules/`, and `__pycache__/` are automatically skipped.

### 6. Run the app

```bash
source venv/bin/activate
python app.py
```

Open **http://localhost:5001** in your browser.

---

## Usage

| Step | Action | What happens |
|------|--------|-------------|
| 1 | Click **Index Files** | Scans `~/Documents` + `~/Downloads`, extracts text, builds FAISS index |
| 2 | Watch the progress bar | Updates per file → switches to "Encoding chunks…" at 100% |
| 3 | Sidebar populates | Every indexed file appears with a green ✓ tick, grouped by folder |
| 4 | Type a question | Semantic search finds the most relevant chunks |
| 5 | See results | AI summary at top, matching chunks below with file path + match % |
| 6 | Click ← to hide sidebar | Sidebar collapses; a → tab remains to restore it |

---

## Project Structure

```
smart_search/
├── app.py               # Flask backend — indexing, search, Groq API
├── memory_helper.py     # System stats and memory utilities
├── requirements.txt     # All Python dependencies with versions
├── .env                 # Your API keys (not committed to git)
├── .env.example         # Template — copy this to .env
├── .gitignore           # Excludes venv/, faiss_index/, .env
├── README.md
├── templates/
│   └── index.html       # Full frontend — sidebar, search, results
└── faiss_index/         # Auto-generated at runtime (not committed)
    ├── index.faiss      # Binary vector index
    └── metadata.pkl     # Document chunk metadata + file paths
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the main UI |
| `POST` | `/index` | Triggers document indexing |
| `GET` | `/progress` | Returns current indexing progress (polled by UI) |
| `GET` | `/index-stats` | Returns doc count and FAISS index size in MB |
| `GET` | `/documents` | Returns list of all indexed documents |
| `POST` | `/search` | Performs semantic search + Groq LLM response |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | 3.1.3 | Web server |
| `sentence-transformers` | 5.1.2 | Text embeddings (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | 1.13.0 | Vector similarity search |
| `groq` | 1.0.0 | Groq LLM API (`llama-3.1-8b-instant`) |
| `torch` | 2.8.0 | PyTorch backend for embeddings |
| `PyPDF2` | 3.0.1 | PDF text extraction |
| `python-docx` | 1.2.0 | DOCX text extraction |
| `python-dotenv` | 1.2.1 | Load `.env` file |
| `numpy` | 2.0.2 | Numerical operations |
| `psutil` | 7.2.2 | System memory monitoring |

Full list in [`requirements.txt`](requirements.txt).

---

## Configuration

Edit these constants at the top of `app.py` to customise behaviour:

| Setting | Default | Description |
|---------|---------|-------------|
| `SCAN_FOLDERS` | `~/Documents`, `~/Downloads` | Folders scanned for documents |
| `FAISS_INDEX_PATH` | `./faiss_index` | Where the FAISS index is saved |
| Max files | `100` | Cap on documents indexed per run |
| `chunk_size` | `256` words | Size of each text chunk |
| `overlap` | `32` words | Word overlap between adjacent chunks |
| LLM model | `llama-3.1-8b-instant` | Groq model used for AI summaries |

---

## How It Works

```
 ┌─────────────────────────────────────────────────────────────┐
 │                        INDEXING                             │
 │                                                             │
 │  ~/Documents + ~/Downloads                                  │
 │         │                                                   │
 │         ▼                                                   │
 │   Filter PDF + DOCX  (skip code/env/git)                   │
 │         │                                                   │
 │         ▼                                                   │
 │   Extract text  (PyPDF2 / python-docx)                     │
 │         │                                                   │
 │         ▼                                                   │
 │   Chunk text  (256 words, 32 overlap)                      │
 │         │                                                   │
 │         ▼                                                   │
 │   Embed chunks  (all-MiniLM-L6-v2, 384-dim vectors)        │
 │         │                                                   │
 │         ▼                                                   │
 │   Save FAISS index + metadata.pkl to disk                  │
 └─────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────┐
 │                        SEARCH                               │
 │                                                             │
 │  User question: "What is my current job title?"            │
 │         │                                                   │
 │         ▼                                                   │
 │   Embed query  (same model, 384-dim vector)                │
 │         │                                                   │
 │         ▼                                                   │
 │   FAISS L2 search  → top-5 nearest chunks                  │
 │         │                                                   │
 │         ▼                                                   │
 │   Send chunks as context to Groq API                       │
 │   (llama-3.1-8b-instant)                                   │
 │         │                                                   │
 │         ▼                                                   │
 │   Return: AI summary + chunks with source paths            │
 └─────────────────────────────────────────────────────────────┘
```

---

## Environment Variables

| Variable | Required | Where to get it |
|----------|----------|----------------|
| `GROQ_API_KEY` | **Yes** | [console.groq.com](https://console.groq.com) — free, no credit card |

---

## Notes

- The FAISS index is **rebuilt from scratch** each time you click Index Files
- `faiss_index/` is excluded from git (see `.gitignore`) — each user builds their own
- Python 3.9 works but **3.10+ is recommended** — Google and other libraries are dropping 3.9 support
- On first run the embedding model downloads ~90 MB and is cached in `~/.cache/huggingface/`

---

## Roadmap

- [ ] Image search using CLIP embeddings
- [ ] ChromaDB as an alternative to FAISS (persistent, easier metadata filtering)
- [ ] Drag-and-drop file upload instead of scanning entire folders
- [ ] Page number tracking in PDF chunks
- [ ] Hybrid search (semantic + keyword BM25)

---

## License

MIT
