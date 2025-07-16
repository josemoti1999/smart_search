# app.py
# Main Flask application for the local RAG system.
import os
import shutil
import atexit
from flask import Flask, render_template, request, jsonify
import torch
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import logging
import sqlite3
import numpy as np
from pathlib import Path

def setup_database():
    """Set up SQLite database for storing document chunks."""
    db_path = os.path.join(os.path.dirname(__file__), 'document_store.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chunks
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT NOT NULL,
                  source TEXT NOT NULL,
                  embedding_file TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def save_embeddings(embeddings, chunk_id):
    """Save embeddings to disk as numpy files."""
    embeddings_dir = os.path.join(os.path.dirname(__file__), 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Save the embedding as a numpy file
    embedding_file = os.path.join(embeddings_dir, f'embedding_{chunk_id}.npy')
    np.save(embedding_file, embeddings.cpu().numpy())
    return embedding_file

def load_embeddings(embedding_files):
    """Load embeddings from disk."""
    embeddings_list = []
    for file in embedding_files:
        embedding = np.load(file)
        embeddings_list.append(embedding)
    return torch.tensor(np.array(embeddings_list))

def clean_text(text):
    """Clean and validate text before encoding."""
    if not isinstance(text, str):
        return ""
    # Remove any null bytes
    text = text.replace('\0', '')
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text if text.strip() else ""

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Specify the folder where your documents are located.
# For this example, we create a 'documents' folder in the same directory as the app.
DOCUMENTS_FOLDER = r"C:\Users\josea\OneDrive\Documents" 

# Ensure the documents folder exists
# if not os.path.exists(DOCUMENTS_FOLDER):
#     os.makedirs(DOCUMENTS_FOLDER)
#     logging.info(f"Created documents folder at: {DOCUMENTS_FOLDER}")
#     # Add a sample file for demonstration
#     with open(os.path.join(DOCUMENTS_FOLDER, 'sample_document.txt'), 'w') as f:
#         f.write("The quick brown fox jumps over the lazy dog. "
#                 "This is a sample document to demonstrate the capabilities of the local RAG system. "
#                 "The system can read text files, PDFs, and Word documents.")

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

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Core Application Logic ---

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


def index_documents():
    """
    Scans the documents folder, reads files, chunks text,
    and generates vector embeddings for each chunk.
    """
    if model is None:
        logging.error("Model not available, cannot generate embeddings.")
        return 0

    # Setup database
    setup_database()
    db_path = os.path.join(os.path.dirname(__file__), 'document_store.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Clear existing data
    c.execute('DELETE FROM chunks')
    conn.commit()
    
    logging.info(f"Starting document indexing from '{DOCUMENTS_FOLDER}'...")
    
    supported_files = 0
    chunk_count = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(DOCUMENTS_FOLDER):
        for filename in files:
            try:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, DOCUMENTS_FOLDER)
                content = ""
                
                # Read content based on file type
                if filename.endswith(".txt"):
                    content = read_text_file(file_path)
                    supported_files += 1
                elif filename.endswith(".pdf"):
                    content = read_pdf_file(file_path)
                    supported_files += 1
                elif filename.endswith(".docx"):
                    content = read_docx_file(file_path)
                    supported_files += 1
                else:
                    logging.warning(f"Skipping unsupported file type: {rel_path}")
                    continue

                if content:
                    # Clean the content before chunking
                    content = clean_text(content)
                    if not content:
                        logging.warning(f"Skipping empty or invalid content in file: {rel_path}")
                        continue

                    chunks = chunk_text(content)
                    if not chunks:
                        logging.warning(f"No valid chunks generated from file: {rel_path}")
                        continue

                    # Process chunks in smaller batches to save memory
                    batch_size = 32
                    for i in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[i:i + batch_size]
                        
                        # Clean each chunk and filter out empty ones
                        valid_chunks = [clean_text(chunk) for chunk in batch_chunks]
                        valid_chunks = [chunk for chunk in valid_chunks if chunk]
                        
                        if not valid_chunks:
                            continue
                        
                        try:
                            # Generate embeddings for the batch
                            batch_embeddings = model.encode(valid_chunks, convert_to_tensor=True)
                            
                            # Save chunks and embeddings
                            for j, (chunk, embedding) in enumerate(zip(valid_chunks, batch_embeddings)):
                                # Save the embedding to disk
                                embedding_file = save_embeddings(embedding, chunk_count + j)
                                
                                # Save chunk info to database
                                c.execute('''INSERT INTO chunks (text, source, embedding_file)
                                           VALUES (?, ?, ?)''', (chunk, rel_path, embedding_file))
                            
                            chunk_count += len(valid_chunks)
                            conn.commit()
                            
                        except Exception as e:
                            logging.error(f"Error processing batch in file {rel_path}: {str(e)}")
                            continue
                            
                    logging.info(f"Processed and chunked '{rel_path}'. Found {len(chunks)} chunks.")
            
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")
                continue

    conn.commit()
    conn.close()

    if chunk_count == 0:
        logging.warning("No text chunks were generated. Check your documents folder or file contents.")
    else:
        logging.info(f"Successfully indexed {chunk_count} chunks.")
    
    return chunk_count



def search(query, top_k=5):
    """
    Performs a semantic search for a given query against the indexed document chunks.
    """
    if model is None:
        return []

    db_path = os.path.join(os.path.dirname(__file__), 'document_store.db')
    if not os.path.exists(db_path):
        logging.error("Database not found. Please index documents first.")
        return []

    logging.info(f"Performing search for query: '{query}'")
    
    print('Here')
    # 1. Generate an embedding for the user's query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # 2. Load all embeddings and chunk info from storage
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, text, source, embedding_file FROM chunks')
    rows = c.fetchall()
    
    if not rows:
        conn.close()
        return []
    
    # Load embeddings in batches to save memory
    batch_size = 1000
    all_results = []
    
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        batch_embeddings = load_embeddings([row[3] for row in batch_rows])
        
        # Compute similarities for this batch
        batch_hits = util.semantic_search(query_embedding, batch_embeddings, top_k=min(top_k, len(batch_rows)))
        
        # Format results
        for hit in batch_hits[0]:  # semantic_search returns a list of lists
            chunk_data = batch_rows[hit['corpus_id']]
            all_results.append({
                'score': float(hit['score']),
                'text': chunk_data[1],
                'source': chunk_data[2]
            })
    
    conn.close()
    
    # Sort all results by score and get top_k
    all_results.sort(key=lambda x: x['score'], reverse=True)
    results = all_results[:top_k]
    
    logging.info(f"Found {len(results)} relevant chunks.")
    return results

def cleanup_storage():
    """Clean up database and embedding files."""
    # Remove embeddings directory
    embeddings_dir = os.path.join(os.path.dirname(__file__), 'embeddings')
    if os.path.exists(embeddings_dir):
        shutil.rmtree(embeddings_dir)
    
    # Remove database
    db_path = os.path.join(os.path.dirname(__file__), 'document_store.db')
    if os.path.exists(db_path):
        os.remove(db_path)

# Remove the cleanup on shutdown
# @atexit.register
# def cleanup():
#     cleanup_storage()



# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main page of the web application."""
    return render_template('index.html')

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
def handle_search():
    """Endpoint to handle user search queries."""
    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()
        
    query = data.get('query')
    
    if not query:
        return jsonify({'status': 'error', 'message': 'Query cannot be empty.'}), 400
        
    try:
        results = search(query)
        if not results:
            return jsonify({'status': 'warning', 'message': 'No matching documents found. Try indexing documents first or try a different query.'}), 200
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        logging.error(f"An error occurred during search: {e}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500
    

# --- Main Execution ---
if __name__ == '__main__':
    # Running in debug mode is useful for development.
    # For a production deployment, you would use a proper WSGI server like Gunicorn.
    app.run(debug=True, port=5001)
