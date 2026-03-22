# app.py
# Main Flask application for the local RAG system.
import os
from dotenv import load_dotenv
load_dotenv()
import shutil
import atexit
import time
import pickle
import psutil  # For CPU and RAM monitoring
import gc  # For garbage collection
from flask import Flask, render_template, request, jsonify
import torch
from sentence_transformers import SentenceTransformer, util
import faiss
import PyPDF2
import docx
import logging
import sqlite3
import numpy as np
from pathlib import Path
from memory_helper import get_system_stats, print_system_stats, check_memory_pressure, cleanup_variables
from groq import Groq
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.docstore.document import Document

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory for storing FAISS index
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss_index')
SCAN_FOLDERS = [
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Downloads"),
]

# --- Model Loading ---
# Load a pre-trained sentence-transformer model. 'all-MiniLM-L6-v2' is a good starting point
# as it's small and efficient, perfect for running locally.
try:
    logging.info("Loading Sentence Transformer model...")
    # Use torch.device to ensure the model runs on CPU if no GPU is available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    logging.info(f"Model loaded successfully on device: {device}")
except Exception as e:
    logging.error(f"Failed to load the model: {e}")
    model = None

# --- Global In-Memory Storage ---
# For simplicity, we'll store the embeddings and corresponding text chunks in memory.
# For a larger-scale application, you would use a vector database like FAISS or Chroma.
document_chunks = []
chunk_embeddings = None

# Global indexing progress
indexing_progress = {"current": 0, "total": 0, "status": "idle"}

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Core Application Logic ---

def clean_text(text):
    """Clean and validate text before encoding."""
    if not isinstance(text, str):
        return ""
    # Remove any null bytes
    text = text.replace('\0', '')
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text if text.strip() else ""

def read_text_file(file_path):
    """Reads content from a .txt file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def read_pdf_file(file_path):
    """Reads content from a .pdf file."""
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
    """Reads content from a .docx file."""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logging.error(f"Could not read DOCX {file_path}: {e}")
    return text

def chunk_text(text, chunk_size=256, overlap=32):
    """
    Splits a long text into smaller chunks with a specified overlap.
    This helps in maintaining context between chunks.
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # Move the window forward, considering the overlap
        start += chunk_size - overlap
        if start >= len(words):
            break
            
    return chunks


def is_env_directory(path):
    """Check if the path is likely a code environment directory."""
    env_patterns = {
        '.env', 'env', 'venv', '.venv', 
        'virtualenv', 'virtual_env',
        '.conda', 'anaconda', 
        'node_modules', '.git',
        '__pycache__', '.pytest_cache',
        'site-packages', 'dist-packages',
        'lib', 'libs', 'build', 'dist'
    }
    
    parts = path.lower().split(os.sep)
    return any(part in env_patterns for part in parts)

def index_documents():
    """
    Scans the documents folder, reads PDF and DOCX files, chunks text,
    and generates vector embeddings for each chunk using FAISS.
    Progress is updated per file so the UI bar moves smoothly.
    """
    global indexing_progress

    if model is None:
        logging.error("Model not available, cannot generate embeddings.")
        return 0

    logging.info("Cleaning up existing index and storage...")
    cleanup_storage()

    logging.info(f"Starting document indexing from: {SCAN_FOLDERS}")

    # Collect only PDF and DOCX files from all scan folders, skip code/env directories
    all_files = []
    for scan_root in SCAN_FOLDERS:
        if not os.path.exists(scan_root):
            continue
        for root, dirs, files in os.walk(scan_root):
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
                        logging.info(f"Found document: {rel_path}")

    # Cap at 100 documents
    all_files = all_files[:100]
    total = len(all_files)

    indexing_progress["total"] = total
    indexing_progress["current"] = 0
    indexing_progress["status"] = "indexing"

    if not all_files:
        logging.warning("No supported files found (PDF or DOCX).")
        indexing_progress["status"] = "completed"
        return 0

    all_texts = []
    all_metadata = []
    supported_files = 0

    # Phase 1: Extract text from each file, update progress per file
    for i, (file_path, rel_path) in enumerate(all_files):
        try:
            content = ""
            if file_path.lower().endswith('.pdf'):
                content = read_pdf_file(file_path)
            elif file_path.lower().endswith('.docx'):
                content = read_docx_file(file_path)

            content = clean_text(content)
            if content:
                chunks = chunk_text(content)
                for chunk in chunks:
                    chunk = clean_text(chunk)
                    if chunk:
                        all_texts.append(chunk)
                        all_metadata.append({
                            'source': rel_path,
                            'full_path': file_path,
                            'text': chunk
                        })
                supported_files += 1
                logging.info(f"Processed '{rel_path}' — {len(chunks)} chunks.")
            else:
                logging.warning(f"No content extracted from {rel_path}")
        except Exception as e:
            logging.error(f"Error processing {rel_path}: {e}")

        # Update progress after each file so the bar moves smoothly
        indexing_progress["current"] = i + 1

    if not all_texts:
        logging.warning("No text chunks were generated.")
        indexing_progress["current"] = total
        indexing_progress["status"] = "completed"
        return 0

    # Ensure progress bar shows 100% before encoding starts
    indexing_progress["current"] = total
    indexing_progress["status"] = "encoding"

    try:
        # Phase 2: Batch-encode all chunks at once (efficient)
        logging.info(f"Encoding {len(all_texts)} chunks from {supported_files} files...")
        embeddings = model.encode(
            all_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )
        embeddings_np = embeddings.cpu().numpy()

        # Build FAISS index
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)

        # Save index and metadata
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, 'index.faiss'))
        with open(os.path.join(FAISS_INDEX_PATH, 'metadata.pkl'), 'wb') as f:
            pickle.dump(all_metadata, f)

        logging.info(f"Successfully indexed {len(all_texts)} chunks from {supported_files} files.")
        indexing_progress["status"] = "completed"
        return len(all_texts)

    except Exception as e:
        logging.error(f"Error creating FAISS index: {e}")
        indexing_progress["status"] = "error"
        return 0
    


def perform_search(query, top_k=5):
    """
    Performs a semantic search using FAISS for a given query against the indexed document chunks.
    """
    print(f"\nStarting FAISS search for query: '{query}'")
    if model is None:
        print("Error: Search model not available")
        return []

    if not os.path.exists(FAISS_INDEX_PATH):
        logging.error("FAISS index not found. Please index documents first.")
        return []

    logging.info(f"Performing search for query: '{query}'")
    
    try:
        # Load the FAISS index
        import faiss
        index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, 'index.faiss'))
        
        # Use GPU if available
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # Load metadata
        import pickle
        with open(os.path.join(FAISS_INDEX_PATH, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Generate query embedding
        query_embedding = model.encode(
            [str(query)],  # Ensure query is string
            convert_to_tensor=True,
            batch_size=1
        )
        query_embedding_np = query_embedding.cpu().numpy()
        
        # Search
        distances, indices = index.search(query_embedding_np, top_k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata):  # Ensure the index is valid
                meta = metadata[idx]
                results.append({
                    'score': float(1.0 / (1.0 + dist)),  # Convert distance to similarity score
                    'source': meta['source'],
                    'text': meta['text']  # Get text from metadata
                })
        
        logging.info(f"Found {len(results)} relevant chunks.")
        return results
        
    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        return []
    

def cleanup_storage():
    """Clean up database and FAISS index."""
    # Remove FAISS index directory
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
    
    # Remove database (if you still want to keep it for other purposes)
    db_path = os.path.join(os.path.dirname(__file__), 'document_store.db')
    if os.path.exists(db_path):
        os.remove(db_path)

# Remove the cleanup on shutdown
# @atexit.register
# def cleanup():
#     cleanup_storage()

def generate_response(context, query):
    """
    Generate a response using Groq API (llama-3.1-8b-instant) based on context and query.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY environment variable not set. Get a free key at console.groq.com"

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions based only on the provided context. Be concise and clear."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            max_tokens=512,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating Groq response: {e}")
        return f"Error generating response: {str(e)}"


@app.route('/')
def home():
    """Renders the main page of the web application."""
    return render_template('index.html')

@app.route('/progress')
def get_progress():
    return jsonify(indexing_progress)

@app.route('/documents')
def get_indexed_documents():
    """Returns list of unique documents currently in the FAISS index."""
    meta_path = os.path.join(FAISS_INDEX_PATH, 'metadata.pkl')
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
                'type': 'pdf' if source.lower().endswith('.pdf') else 'docx'
            }
    return jsonify({'documents': list(seen.values())})

@app.route('/index-stats')
def get_index_stats():
    """Returns how many documents are indexed and FAISS index size in MB."""
    stats = {"doc_count": 0, "chunk_count": 0, "size_mb": 0.0, "indexed": False}
    meta_path = os.path.join(FAISS_INDEX_PATH, 'metadata.pkl')
    index_path = os.path.join(FAISS_INDEX_PATH, 'index.faiss')
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
    """Endpoint to trigger the document indexing process."""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model is not loaded. Cannot index.'}), 500
        
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            
        # Clear existing index if requested
        if data and data.get('reset_index', False):
            cleanup_storage()
            
        num_chunks = index_documents()
        if num_chunks > 0:
            return jsonify({'status': 'success', 'message': f'Successfully indexed {num_chunks} text chunks.'})
        else:
            return jsonify({'status': 'warning', 'message': 'No documents were found or processed. Please add files to the "documents" folder.'})
    except Exception as e:
        logging.error(f"An error occurred during indexing: {e}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search_endpoint():
    """
    Handles search requests by finding relevant document chunks and generating
    a response using the Gemma model with comprehensive system monitoring.
    """
    print("\n=== Starting Search Process ===")
    print(f"Request Method: {request.method}")
    print(f"Content-Type: {request.headers.get('Content-Type', 'Not specified')}")
    
    start_time = time.time()
    initial_stats = print_system_stats("Search Start")

    if model is None:
        print("Error: Model not available")
        return jsonify({'error': 'Model not available'}), 500

    # Check if the request has JSON content
    if not request.is_json:
        print("Error: Request must be JSON")
        return jsonify({'error': 'Request must be JSON'}), 400

    query = request.json.get('query', '')
    file_type = request.json.get('fileType', 'All Files').lower()
    
    print(f"Query received: '{query}'")
    print(f"File type filter: {file_type}")
    
    if not query:
        print("Error: No query provided")
        return jsonify({'error': 'No query provided'}), 400

    try:
        print("\nStep 2: Generating Query Embedding")
        embed_start = time.time()
        # Generate embedding for the query
        query_embedding = model.encode([query])[0]
        print(f"Query embedding generated in {time.time() - embed_start:.2f} seconds")
        
        print("\nStep 3: Preparing for FAISS Search")
        faiss_prep_start = time.time()
        # Convert query embedding to the format expected by FAISS
        query_embedding = query_embedding.astype('float32')
        print(f"Query embedding converted to float32 in {time.time() - faiss_prep_start:.2f} seconds")
        
        print("\nStep 4: Loading FAISS Index")
        index_load_start = time.time()

        # Check if FAISS index exists
        if not os.path.exists(os.path.join(FAISS_INDEX_PATH, 'index.faiss')):
            print("Error: FAISS index not found")
            return jsonify({'error': 'No documents indexed. Please index documents first.'}), 400

        # Load FAISS index
        import faiss
        index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, 'index.faiss'))
        print(f"FAISS index loaded in {time.time() - index_load_start:.2f} seconds")
        
        # Move to GPU if available and system has resources
        faiss_stats = get_system_stats()
        if torch.cuda.is_available() and faiss_stats['ram_percent'] < 80:
            gpu_start = time.time()
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print(f"Index moved to GPU in {time.time() - gpu_start:.2f} seconds")
            except Exception as e:
                print(f"Failed to move FAISS to GPU: {e}, using CPU")

        print("\nStep 5: Performing FAISS Search")
        search_start = time.time()
        D, I = index.search(query_embedding.reshape(1, -1), k=5)
        print(f"FAISS search completed in {time.time() - search_start:.2f} seconds")
        
        if len(I) == 0 or len(I[0]) == 0:
            print("Error: No relevant documents found")
            return jsonify({'error': 'No relevant documents found'}), 404
            
        print("\nStep 6: Processing Search Results")
        results_start = time.time()
        
        # Load metadata
        print("Loading metadata...")
        with open(os.path.join(FAISS_INDEX_PATH, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        print(f"Metadata loaded in {time.time() - results_start:.2f} seconds")
        
        # Clean up FAISS resources
        cleanup_variables(index, query_embedding)
        
        # Filter results by file type if specified
        print("Filtering and formatting results...")
        results = []
        for idx in I[0]:
            if idx < len(metadata):  # Ensure the index is valid
                meta = metadata[idx]
                if file_type == 'all files' or file_type == 'all' or meta['source'].lower().endswith(get_file_extension(file_type)):
                    results.append({
                        'text': meta['text'],
                        'source': meta['source'],
                        'full_path': meta.get('full_path', meta['source']),
                        'score': float(1.0 / (1.0 + D[0][list(I[0]).index(idx)]))
                    })
        
        # Clean up search data
        cleanup_variables(D, I, metadata)
        
        if not results:
            print("Error: No matches found for the specified file type")
            return jsonify({'error': 'No matches found for the specified file type'}), 404

        print(f"Found {len(results)} relevant documents")
        print(f"Results processing completed in {time.time() - results_start:.2f} seconds")
        search_stats = print_system_stats("After Search")

        # Combine the relevant contexts
        print("\nStep 7: Preparing Context for Gemma")
        context_start = time.time()
        combined_context = "\n".join([r['text'] for r in results])
        print(f"Context preparation completed in {time.time() - context_start:.2f} seconds")
        
        print("\nStep 8: Generating Gemma Response")
        gemma_start = time.time()
        # Generate a response using Gemini
        gemma_response = generate_response(combined_context, query)
        gemma_generation_time = time.time() - gemma_start
        print(f"Gemini response generated in {gemma_generation_time:.2f} seconds")
        
        # Clean up context data
        cleanup_variables(combined_context)
        
        print("\nStep 9: Preparing Final Response")
        response_start = time.time()
        
        # Add the Gemma response to the results
        final_results = [{
            'text': gemma_response,
            'source': 'AI Assistant',
            'full_path': None,
            'score': 1.0
        }] + results  # Add original context chunks after the AI response
        
        total_time = time.time() - start_time
        final_stats = print_system_stats("Search Complete")
        
        print(f"\n🏁 Search Summary:")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   RAM Change: {final_stats['ram_used_gb'] - initial_stats['ram_used_gb']:+.2f}GB")
        print(f"   Peak RAM Usage: {max(search_stats['ram_percent'], final_stats['ram_percent']):.1f}%")
        
        return jsonify({
            'results': final_results,
            'processing_time': f"{total_time:.2f}",
            'gemma_generation_time': f"{gemma_generation_time:.2f}",
            'model_used': 'llama-3.1-8b-instant (Groq)',
            'system_stats': {
                'ram_usage_percent': final_stats['ram_percent'],
                'peak_ram_percent': max(search_stats['ram_percent'], final_stats['ram_percent']),
                'cpu_usage_percent': final_stats['cpu_percent']
            }
        })

    except Exception as e:
        print(f"\nError during search: {str(e)}")
        logging.error(f"Search error: {e}")
        print_system_stats("Error State")
        return jsonify({'error': str(e)}), 500

def get_file_extension(file_type):
    """Convert file type to file extension."""
    file_type_mapping = {
        'documents': '.docx',
        'pdfs': '.pdf',
        'spreadsheets': '.xlsx',
        'images': '.jpg'  # Add more image extensions if needed
    }
    return file_type_mapping.get(file_type, '')

# --- Main Execution ---
if __name__ == '__main__':
    # Running in debug mode is useful for development.
    # For a production deployment, you would use a proper WSGI server like Gunicorn.
    app.run(debug=True, port=5001)
