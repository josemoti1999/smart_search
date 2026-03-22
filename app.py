# app.py
# Main Flask application for the local RAG system.
import os
from dotenv import load_dotenv
load_dotenv()
import shutil
import time
import pickle
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
import faiss
import PyPDF2
import docx
import numpy as np
from memory_helper import print_system_stats, cleanup_variables
from groq import Groq
from rank_bm25 import BM25Okapi
from config import get_config, save_config
import embedder  # ONNX-based, no torch required

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Config helpers ---
def get_faiss_path() -> str:
    return get_config().get('faiss_index_path', os.path.join(os.path.dirname(__file__), 'faiss_index'))

def get_groq_key() -> str:
    """Groq key: only from config file. Must be set explicitly in Settings."""
    return get_config().get("groq_api_key", "").strip()

def get_groq_model() -> str:
    return get_config().get("groq_model", "llama-3.1-8b-instant")


# --- Eagerly load the embedding model at startup ---
try:
    embedder.load_model()
except Exception as e:
    logging.error(f"Failed to load embedding model: {e}")

# --- Global state ---
indexing_progress = {"current": 0, "total": 0, "status": "idle"}

# --- Flask App ---
app = Flask(__name__)


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace('\0', '')
    text = ' '.join(text.split())
    return text if text.strip() else ""


def read_pdf_file(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"Could not read PDF {file_path}: {e}")
    return text


def read_docx_file(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logging.error(f"Could not read DOCX {file_path}: {e}")
    return text


def chunk_text(text, chunk_size=256, overlap=32):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
        if start >= len(words):
            break
    return chunks


def is_env_directory(path):
    env_patterns = {
        '.env', 'env', 'venv', '.venv', 'virtualenv', 'virtual_env',
        '.conda', 'anaconda', 'node_modules', '.git',
        '__pycache__', '.pytest_cache', 'site-packages', 'dist-packages',
        'lib', 'libs', 'build', 'dist',
    }
    parts = path.lower().split(os.sep)
    return any(part in env_patterns for part in parts)


def cleanup_storage():
    faiss_path = get_faiss_path()
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
    db_path = os.path.join(os.path.dirname(__file__), 'document_store.db')
    if os.path.exists(db_path):
        os.remove(db_path)


# ──────────────────────────────────────────────
# Core: indexing
# ──────────────────────────────────────────────

def index_documents():
    global indexing_progress

    cfg = get_config()
    faiss_path = cfg['faiss_index_path']
    scan_folders = cfg['scan_folders']
    max_files = cfg.get('max_files', 100)
    chunk_size = cfg.get('chunk_size', 256)
    chunk_overlap = cfg.get('chunk_overlap', 32)

    logging.info("Cleaning up existing index...")
    cleanup_storage()
    logging.info(f"Starting indexing from: {scan_folders}")

    all_files = []
    for scan_root in scan_folders:
        expanded = os.path.expanduser(scan_root)
        if not os.path.exists(expanded):
            continue
        for root, dirs, files in os.walk(expanded):
            if is_env_directory(root):
                dirs.clear()
                continue
            dirs[:] = [d for d in dirs if not is_env_directory(os.path.join(root, d))]
            for filename in files:
                if filename.lower().endswith(('.pdf', '.docx')):
                    file_path = os.path.join(root, filename)
                    if not is_env_directory(file_path):
                        rel_path = os.path.relpath(file_path, os.path.expanduser("~"))
                        all_files.append((file_path, rel_path))

    all_files = all_files[:max_files]
    total = len(all_files)
    indexing_progress["total"] = total
    indexing_progress["current"] = 0
    indexing_progress["status"] = "indexing"

    if not all_files:
        logging.warning("No supported files found.")
        indexing_progress["status"] = "completed"
        return 0

    all_texts = []
    all_metadata = []
    supported_files = 0

    # Phase 1: Extract text, update progress per file
    for i, (file_path, rel_path) in enumerate(all_files):
        try:
            content = ""
            if file_path.lower().endswith('.pdf'):
                content = read_pdf_file(file_path)
            elif file_path.lower().endswith('.docx'):
                content = read_docx_file(file_path)
            content = clean_text(content)
            if content:
                chunks = chunk_text(content, chunk_size, chunk_overlap)
                for chunk in chunks:
                    chunk = clean_text(chunk)
                    if chunk:
                        all_texts.append(chunk)
                        all_metadata.append({
                            'source': rel_path,
                            'full_path': file_path,
                            'text': chunk,
                        })
                supported_files += 1
        except Exception as e:
            logging.error(f"Error processing {rel_path}: {e}")
        indexing_progress["current"] = i + 1

    if not all_texts:
        logging.warning("No text chunks generated.")
        indexing_progress["current"] = total
        indexing_progress["status"] = "completed"
        return 0

    indexing_progress["current"] = total
    indexing_progress["status"] = "encoding"

    try:
        # Phase 2: Batch encode all chunks (ONNX, no torch)
        logging.info(f"Encoding {len(all_texts)} chunks from {supported_files} files...")
        embeddings_np = embedder.encode(all_texts, batch_size=64)

        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)

        os.makedirs(faiss_path, exist_ok=True)
        faiss.write_index(index, os.path.join(faiss_path, 'index.faiss'))
        with open(os.path.join(faiss_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(all_metadata, f)

        # Build BM25 index
        tokenized_corpus = [text.lower().split() for text in all_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(os.path.join(faiss_path, 'bm25.pkl'), 'wb') as f:
            pickle.dump(bm25, f)

        logging.info(f"Indexed {len(all_texts)} chunks from {supported_files} files.")
        indexing_progress["status"] = "completed"
        return len(all_texts)

    except Exception as e:
        logging.error(f"Error creating FAISS index: {e}")
        indexing_progress["status"] = "error"
        return 0


# ──────────────────────────────────────────────
# Core: LLM
# ──────────────────────────────────────────────

def generate_response(context, query):
    try:
        api_key = get_groq_key()
        if not api_key:
            return "Error: Groq API key not configured. Go to Settings to add your free key from console.groq.com"
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=get_groq_model(),
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer questions based only on the provided context. Be concise and clear."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating Groq response: {e}")
        return f"Error generating response: {str(e)}"


def get_file_extension(file_type):
    return {'documents': '.docx', 'pdfs': '.pdf'}.get(file_type, '')


# ──────────────────────────────────────────────
# Routes: main UI
# ──────────────────────────────────────────────

@app.route('/')
def home():
    cfg = get_config()
    no_key = not cfg.get('groq_api_key', '').strip()
    return render_template('index.html', no_api_key=no_key)


@app.route('/progress')
def get_progress():
    return jsonify(indexing_progress)


@app.route('/documents')
def get_indexed_documents():
    faiss_path = get_faiss_path()
    meta_path = os.path.join(faiss_path, 'metadata.pkl')
    if not os.path.exists(meta_path):
        return jsonify({'documents': []})
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    seen = {}
    for m in metadata:
        source = m['source']
        if source not in seen:
            seen[source] = {
                'source': source,
                'full_path': m.get('full_path', source),
                'filename': os.path.basename(source),
                'folder': os.path.dirname(source) or '/',
                'type': 'pdf' if source.lower().endswith('.pdf') else 'docx',
            }
    return jsonify({'documents': list(seen.values())})


@app.route('/index-stats')
def get_index_stats():
    faiss_path = get_faiss_path()
    stats = {"doc_count": 0, "chunk_count": 0, "size_mb": 0.0, "indexed": False}
    meta_path = os.path.join(faiss_path, 'metadata.pkl')
    index_path = os.path.join(faiss_path, 'index.faiss')
    if os.path.exists(meta_path) and os.path.exists(index_path):
        stats["indexed"] = True
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        stats["doc_count"] = len(set(m['source'] for m in metadata))
        stats["chunk_count"] = len(metadata)
        stats["size_mb"] = round(os.path.getsize(index_path) / (1024 * 1024), 2)
    return jsonify(stats)


@app.route('/index', methods=['POST'])
def handle_index():
    try:
        num_chunks = index_documents()
        if num_chunks > 0:
            return jsonify({'status': 'success', 'message': f'Successfully indexed {num_chunks} text chunks.'})
        else:
            return jsonify({'status': 'warning', 'message': 'No documents found. Add PDF or DOCX files to your scan folders.'})
    except Exception as e:
        logging.error(f"Indexing error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/search', methods=['POST'])
def search_endpoint():
    start_time = time.time()
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    query = request.json.get('query', '')
    file_type = request.json.get('fileType', 'All Files').lower()

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    faiss_path = get_faiss_path()

    try:
        query_embedding = embedder.encode_single(query).astype('float32')

        if not os.path.exists(os.path.join(faiss_path, 'index.faiss')):
            return jsonify({'error': 'No documents indexed. Click Index Files first.'}), 400

        faiss_index = faiss.read_index(os.path.join(faiss_path, 'index.faiss'))
        with open(os.path.join(faiss_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        bm25 = None
        bm25_path = os.path.join(faiss_path, 'bm25.pkl')
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                bm25 = pickle.load(f)

        top_k = 5
        candidate_k = top_k * 3

        # FAISS semantic search
        D, I = faiss_index.search(query_embedding.reshape(1, -1), k=candidate_k)
        faiss_indices = [int(idx) for idx in I[0] if idx < len(metadata)]

        # BM25 keyword search
        bm25_indices = []
        if bm25 is not None:
            bm25_scores = bm25.get_scores(query.lower().split())
            bm25_indices = list(np.argsort(bm25_scores)[::-1][:candidate_k])

        # Reciprocal Rank Fusion
        RRF_K = 60
        rrf_scores = {}
        for rank, idx in enumerate(faiss_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + RRF_K)
        for rank, idx in enumerate(bm25_indices):
            if idx < len(metadata):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + RRF_K)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, rrf_score in ranked:
            meta = metadata[idx]
            if file_type in ('all files', 'all') or meta['source'].lower().endswith(get_file_extension(file_type)):
                results.append({
                    'text': meta['text'],
                    'source': meta['source'],
                    'full_path': meta.get('full_path', meta['source']),
                    'score': round(rrf_score, 4),
                })

        cleanup_variables(faiss_index, query_embedding, D, I, metadata)

        if not results:
            return jsonify({'error': 'No matches found for the specified file type'}), 404

        combined_context = "\n".join([r['text'] for r in results])

        ai_start = time.time()
        ai_response = generate_response(combined_context, query)
        ai_time = time.time() - ai_start

        final_results = [{
            'text': ai_response,
            'source': 'AI Assistant',
            'full_path': None,
            'score': 1.0,
        }] + results

        total_time = time.time() - start_time

        return jsonify({
            'results': final_results,
            'processing_time': f"{total_time:.2f}",
            'gemma_generation_time': f"{ai_time:.2f}",
            'model_used': f"{get_groq_model()} (Groq)",
            'search_method': 'Hybrid (FAISS + BM25)' if bm25 else 'Semantic (FAISS)',
        })

    except Exception as e:
        logging.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# Routes: settings
# ──────────────────────────────────────────────

@app.route('/settings')
def settings():
    cfg = get_config()
    # Mask API key for display
    key = get_groq_key()
    if key and len(key) > 8:
        cfg['groq_api_key_display'] = key[:4] + '•' * (len(key) - 8) + key[-4:]
    else:
        cfg['groq_api_key_display'] = ''
    return render_template('settings.html', config=cfg)


@app.route('/settings/save', methods=['POST'])
def settings_save():
    data = request.get_json()
    cfg = get_config()

    # Update scan folders (filter empty strings)
    if 'scan_folders' in data:
        folders = [f.strip() for f in data['scan_folders'] if f.strip()]
        cfg['scan_folders'] = folders

    # Update API key if provided and not a masked value
    if 'groq_api_key' in data:
        key = data['groq_api_key'].strip()
        if key and '•' not in key:
            cfg['groq_api_key'] = key

    # Update other settings
    for field in ('groq_model', 'max_files', 'chunk_size', 'chunk_overlap'):
        if field in data:
            val = data[field]
            if field in ('max_files', 'chunk_size', 'chunk_overlap'):
                try:
                    val = int(val)
                except (TypeError, ValueError):
                    continue
            cfg[field] = val

    cfg['first_run'] = False
    save_config(cfg)

    return jsonify({'status': 'saved'})


@app.route('/settings/test-api-key', methods=['POST'])
def settings_test_key():
    data = request.get_json()
    # Only test the key explicitly sent — never fall back to env or config
    key = (data or {}).get('groq_api_key', '').strip()
    if not key:
        return jsonify({'status': 'error', 'message': 'Please enter an API key first.'})
    # Reject obviously masked values (e.g. "gsk_••••YFOi")
    if '•' in key:
        return jsonify({'status': 'error', 'message': 'Please enter the actual key, not the masked value.'})
    try:
        client = Groq(api_key=key)
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        return jsonify({'status': 'ok', 'message': 'API key is valid.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/settings/config')
def settings_get_config():
    """Return current config as JSON (used by frontend)."""
    cfg = get_config()
    # Don't expose the API key
    cfg_safe = {k: v for k, v in cfg.items() if k != 'groq_api_key'}
    return jsonify(cfg_safe)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

@app.route('/status')
def status():
    return jsonify({'model_ready': embedder.is_ready()})


if __name__ == '__main__':
    import socket
    cfg = get_config()
    port = cfg.get('port', 5001)
    # Auto-find a free port if preferred one is taken
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', port)) == 0:
            # Port in use — find a free one
            with socket.socket() as s2:
                s2.bind(('127.0.0.1', 0))
                port = s2.getsockname()[1]
            logging.warning(f"Port 5001 in use, using port {port} instead")
    # Write the actual port to config so Electron can read it
    cfg['_runtime_port'] = port
    save_config(cfg)
    app.run(debug=False, port=port, host='127.0.0.1')
