# app.py
# Main Flask application for the local RAG system.
import os
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
from transformers import AutoProcessor, Gemma3nForConditionalGeneration, BitsAndBytesConfig
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.docstore.document import Document

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory for storing FAISS index
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss_index')
DOCUMENTS_FOLDER = r"C:\Users\josea\OneDrive\Documents" 

# Directory for storing models
MODELS_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'models_cache')
GEMMA_CACHE_PATH = os.path.join(MODELS_CACHE_PATH, 'gemma-3n-e2b-it')

# Create models cache directory if it doesn't exist
os.makedirs(MODELS_CACHE_PATH, exist_ok=True)

def get_system_stats():
    """Get current system CPU and RAM statistics."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    stats = {
        'cpu_percent': cpu_percent,
        'ram_total_gb': memory.total / (1024**3),
        'ram_used_gb': memory.used / (1024**3),
        'ram_available_gb': memory.available / (1024**3),
        'ram_percent': memory.percent
    }
    return stats

def print_system_stats(step_name="System"):
    """Print current system statistics in a formatted way."""
    stats = get_system_stats()
    print(f"\n📊 {step_name} Resources:")
    print(f"   CPU Usage: {stats['cpu_percent']:.1f}%")
    print(f"   RAM: {stats['ram_used_gb']:.2f}GB / {stats['ram_total_gb']:.2f}GB ({stats['ram_percent']:.1f}%)")
    print(f"   Available RAM: {stats['ram_available_gb']:.2f}GB")
    return stats

def cleanup_variables(*variables):
    """Clean up specified variables and run garbage collection."""
    for var in variables:
        if var is not None:
            del var
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_memory_pressure():
    """Check if system is under memory pressure."""
    stats = get_system_stats()
    return stats['ram_percent'] > 85.0  # Return True if RAM usage > 85%

def check_memory_pressure():
    """Check if system is under memory pressure."""
    stats = get_system_stats()
    return stats['ram_percent'] > 85.0  # Return True if RAM usage > 85%

def load_or_download_gemma():
    """Load Gemma model from local cache or download if not available."""
    try:
        logging.info("Loading Gemma model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if model is already cached
        if os.path.exists(GEMMA_CACHE_PATH):
            logging.info("Loading Gemma model from local cache...")
            model = Gemma3nForConditionalGeneration.from_pretrained(
                GEMMA_CACHE_PATH,
                device_map=device,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            ).eval()
            processor = AutoProcessor.from_pretrained(
                GEMMA_CACHE_PATH,
                local_files_only=True
            )
        else:
            logging.info("Downloading Gemma model for the first time...")
            # Download and save the model
            model = Gemma3nForConditionalGeneration.from_pretrained(
                "google/gemma-3n-e2b-it",
                device_map=device,
                torch_dtype=torch.bfloat16
            ).eval()
            print("Model loaded")
            processor = AutoProcessor.from_pretrained("google/gemma-3n-e2b-it")
            
            # Save model and processor to local cache
            logging.info("Saving Gemma model to local cache...")
            model.save_pretrained(GEMMA_CACHE_PATH)
            processor.save_pretrained(GEMMA_CACHE_PATH)
            
        logging.info(f"Gemma model loaded successfully on device: {device}")
        return model, processor
    except Exception as e:
        logging.error(f"Failed to load Gemma model: {e}")
        return None, None

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

# Don't initialize Gemma model at startup
gemma_model, gemma_processor = None, None

# --- Global In-Memory Storage ---
# For simplicity, we'll store the embeddings and corresponding text chunks in memory.
# For a larger-scale application, you would use a vector database like FAISS or Chroma.
document_chunks = []
chunk_embeddings = None

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
    Scans the documents folder, reads files, chunks text,
    and generates vector embeddings for each chunk using FAISS.
    """
    if model is None:
        logging.error("Model not available, cannot generate embeddings.")
        return 0

    # Clean up any existing index and storage
    logging.info("Cleaning up existing index and storage...")
    cleanup_storage()
    
    logging.info(f"Starting document indexing from '{DOCUMENTS_FOLDER}'...")
    
    # Collect all files first, excluding environment directories
    all_files = []
    for root, dirs, files in os.walk(DOCUMENTS_FOLDER):
        # Skip environment directories
        if is_env_directory(root):
            dirs.clear()  # Skip subdirectories
            continue
            
        # Remove environment directories from dirs list to prevent walking into them
        dirs[:] = [d for d in dirs if not is_env_directory(os.path.join(root, d))]
        
        for filename in files:
            if filename.endswith(('.txt', '.pdf', '.docx')):
                file_path = os.path.join(root, filename)
                # Extra check to ensure we're not in an environment directory
                if not is_env_directory(file_path):
                    rel_path = os.path.relpath(file_path, DOCUMENTS_FOLDER)
                    all_files.append((file_path, rel_path))
                    logging.info(f"Found document: {rel_path}")
    
    if not all_files:
        logging.warning("No supported files found.")
        return 0

    supported_files = 0
    chunk_count = 0
    all_metadata = []
    
    try:
        # Process files in batches
        file_batch_size = 100
        
        # Initialize FAISS index
        # We'll determine the dimension from the first embedding
        index = None
        
        for i in range(0, len(all_files), file_batch_size):
            batch_files = all_files[i:i + file_batch_size]
            batch_documents = []
            batch_texts = []  # Store texts separately
            batch_metadata = []  # Store metadata separately
            
            # Process each file in the current batch
            for file_path, rel_path in batch_files:
                try:
                    content = ""
                    if file_path.endswith('.txt'):
                        content = read_text_file(file_path)
                    elif file_path.endswith('.pdf'):
                        content = read_pdf_file(file_path)
                    elif file_path.endswith('.docx'):
                        content = read_docx_file(file_path)
                    
                    if content := clean_text(content):
                        chunks = chunk_text(content)
                        for chunk in chunks:
                            if chunk := clean_text(chunk):
                                batch_texts.append(str(chunk))  # Ensure text is string
                                batch_metadata.append({
                                    'source': rel_path,
                                    'text': str(chunk)  # Store the text content with metadata
                                })
                        
                        supported_files += 1
                        logging.info(f"Processed '{rel_path}'. Found {len(chunks)} chunks.")
                except Exception as e:
                    logging.error(f"Error processing file {rel_path}: {str(e)}")
                    continue
            
            if not batch_texts:
                continue
                
            try:
                # Generate embeddings using sentence-transformers
                embeddings = model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    show_progress_bar=True,
                    batch_size=32  # Process in smaller sub-batches
                )
                embeddings_np = embeddings.cpu().numpy()
                
                # Initialize index with first batch if not done yet
                if index is None:
                    dimension = embeddings_np.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    if torch.cuda.is_available():
                        res = faiss.StandardGpuResources()
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                
                # Add to index
                index.add(embeddings_np)
                all_metadata.extend(batch_metadata)
                chunk_count += len(batch_texts)
                
                logging.info(f"Indexed batch of {len(batch_texts)} chunks. Total chunks: {chunk_count}")
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
                continue
        
        if chunk_count == 0:
            logging.warning("No text chunks were generated. Check your documents folder or file contents.")
            return 0
        
        # Save the index and metadata
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        
        # Convert GPU index back to CPU for saving if necessary
        if torch.cuda.is_available():
            index = faiss.index_gpu_to_cpu(index)
            
        faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, 'index.faiss'))
        
        # Save metadata
        import pickle
        with open(os.path.join(FAISS_INDEX_PATH, 'metadata.pkl'), 'wb') as f:
            pickle.dump(all_metadata, f)
        
        logging.info(f"Successfully indexed {chunk_count} chunks from {supported_files} files.")
        return chunk_count
        
    except Exception as e:
        logging.error(f"Error creating FAISS index: {str(e)}")
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

def generate_gemma_response(context, query):
    """
    Generate a response using the Gemma model based on context and query.
    Uses dynamic loading and memory optimization with system monitoring.
    """
    try:
        print("\n=== Starting Gemma Response Generation ===")
        start_time = time.time()
        initial_stats = print_system_stats("Initial")

        # Step 1: Aggressive Memory Cleanup
        print("\nStep 1: Initial Memory Cleanup")
        cleanup_start = time.time()
        
        if torch.cuda.is_available():
            # Print initial GPU state
            initial_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"Initial GPU Memory Allocated: {initial_allocated:.2f}GB")
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            
            # Clear all existing tensors
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        del obj
                except:
                    pass
            
            # Final cache clear
            torch.cuda.empty_cache()
            gc.collect()  # Additional cleanup
            
            # Check memory after cleanup
            after_cleanup = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"After Aggressive Cleanup GPU Memory: {after_cleanup:.2f}GB")
            
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Total Available GPU Memory: {gpu_memory:.2f}GB")
            
        print(f"Step 1 Time: {time.time() - cleanup_start:.2f} seconds")
        cleanup_stats = print_system_stats("After Cleanup")

        # Step 2: Load processor and prepare messages
        print("\nStep 2: Loading Processor and Preparing Messages")
        step2_start = time.time()
        
        processor = AutoProcessor.from_pretrained(
            GEMMA_CACHE_PATH if os.path.exists(GEMMA_CACHE_PATH) else "google/gemma-3n-e2b-it",
            local_files_only=os.path.exists(GEMMA_CACHE_PATH)
        )
        print(f"Processor loaded in {time.time() - step2_start:.2f} seconds")
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant. Use the provided context to answer questions accurately."}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"Context: {context}\n\nQuestion: {query}\n\nProvide a clear and concise answer based on the given context."
                    }
                ]
            }
        ]
        print(f"Messages prepared. Total step time: {time.time() - step2_start:.2f} seconds")
        step2_stats = print_system_stats("After Processor Load")

        # Step 3: Process input template
        print("\nStep 3: Processing Input Template")
        step3_start = time.time()
        
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_len = inputs["input_ids"].shape[-1]
        print(f"Input sequence length: {input_len} tokens")
        print(f"Input processing time: {time.time() - step3_start:.2f} seconds")
        step3_stats = print_system_stats("After Input Processing")
        
        # Clean up message variables
        cleanup_variables(messages)
        
        # Step 4: Configure model loading based on available memory
        print("\nStep 4: Memory Configuration for Model Loading")
        step4_start = time.time()
        
        # Check system memory pressure
        memory_pressure = check_memory_pressure()
        
        if memory_pressure:
            print(f"⚠️ System under memory pressure - using conservative settings")
        
        # Check actual available GPU memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            free_memory = total_memory - allocated_memory
            
            print(f"GPU Memory Analysis:")
            print(f"Total GPU Memory: {total_memory:.2f}GB")
            print(f"Currently Allocated: {allocated_memory:.2f}GB")
            print(f"Available for Model: {free_memory:.2f}GB")
            
            # Adjust memory strategy based on available RAM and GPU
            available_ram = step3_stats['ram_available_gb']
            
            if memory_pressure or available_ram < 3.0:
                print(f"🔴 Low system RAM (Available: {available_ram:.1f}GB) - using CPU mode")
                device = "cpu"
                torch_dtype = torch.float32
                use_streaming = False
                use_quantization = False
            elif total_memory <= 8.0:  # For GPUs with 8GB or less
                print(f"🧠 GPU has {total_memory:.1f}GB - using layer-by-layer streaming")
                print("🔄 This allows using GPU with minimal memory footprint")
                device = "cuda:0"
                torch_dtype = torch.float16
                use_streaming = True
                use_quantization = True
            else:
                print(f"💪 GPU has {total_memory:.1f}GB - using standard loading")
                device = "cuda:0"
                torch_dtype = torch.float16
                use_streaming = False
                use_quantization = True
        else:
            print("No GPU available, using CPU mode")
            device = "cpu"
            torch_dtype = torch.float32
            use_streaming = False
            use_quantization = False
            
        print(f"Selected device: {device}")
        print(f"Using quantization: {use_quantization}")
        print(f"Using layer streaming: {use_streaming}")
        print(f"Memory configuration time: {time.time() - step4_start:.2f} seconds")
        step4_stats = print_system_stats("After Memory Config")
            
        # Step 5: Load model with automatic resource detection
        print("\nStep 5: Loading Model")
        step5_start = time.time()
        
        print("Starting model load...")
        
        if device == "cpu":
            print("Loading Gemma model on CPU")
            
            # Monitor memory during CPU loading
            pre_load_stats = get_system_stats()
            
            try:
                model = Gemma3nForConditionalGeneration.from_pretrained(
                    GEMMA_CACHE_PATH if os.path.exists(GEMMA_CACHE_PATH) else "google/gemma-3n-e2b-it",
                    torch_dtype=torch_dtype,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    local_files_only=os.path.exists(GEMMA_CACHE_PATH)
                ).eval()
                
                # Check memory after loading
                post_load_stats = get_system_stats()
                memory_used = post_load_stats['ram_used_gb'] - pre_load_stats['ram_used_gb']
                
                print(f"✅ Model loaded successfully on CPU")
                print(f"📊 Memory used for model: {memory_used:.2f}GB")
                    
            except Exception as e:
                print(f"❌ CPU model loading failed: {e}")
                return f"Error loading model on CPU: {str(e)}"
            
        elif use_streaming:
            print("🧠 Loading Gemma model with layer-by-layer streaming for limited GPU")
            
            # Set up offload directory for layer streaming
            offload_dir = os.path.join(os.path.dirname(__file__), 'model_streaming')
            os.makedirs(offload_dir, exist_ok=True)
            
            # Calculate memory budget based on available RAM and GPU
            available_ram = step4_stats['ram_available_gb']
            gpu_memory_budget = min(total_memory * 0.5, available_ram * 0.3)  # Conservative approach
            
            print(f"Available System RAM: {available_ram:.2f}GB")
            print(f"GPU Memory Budget: {gpu_memory_budget:.2f}GB")
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_threshold=6.0,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=False,
            )
            
            # First attempt: Try with GPU + disk offloading
            try:
                print("🚀 Attempting GPU loading with conservative memory limits...")
                model = Gemma3nForConditionalGeneration.from_pretrained(
                    GEMMA_CACHE_PATH if os.path.exists(GEMMA_CACHE_PATH) else "google/gemma-3n-e2b-it",
                    torch_dtype=torch_dtype,
                    quantization_config=quantization_config,
                    device_map="auto",
                    offload_folder=offload_dir,
                    max_memory={0: f"{gpu_memory_budget:.1f}GB"},
                    low_cpu_mem_usage=True,
                    offload_state_dict=True,
                    local_files_only=os.path.exists(GEMMA_CACHE_PATH)
                ).eval()
                print("✅ Model loaded successfully with GPU + disk offloading!")
                
            except Exception as gpu_error:
                print(f"⚠️ GPU loading failed: {str(gpu_error)}")
                print("🔄 Falling back to CPU without quantization...")
                
                # Clean up before fallback
                cleanup_variables(quantization_config)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Fallback: Load on CPU without quantization to avoid type conflicts
                model = Gemma3nForConditionalGeneration.from_pretrained(
                    GEMMA_CACHE_PATH if os.path.exists(GEMMA_CACHE_PATH) else "google/gemma-3n-e2b-it",
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    local_files_only=os.path.exists(GEMMA_CACHE_PATH)
                ).eval()
                print("✅ Model loaded successfully on CPU without quantization!")
                device = "cpu"  # Update device for input handling
            
        else:
            print("Loading Gemma model on GPU with 8-bit quantization")
            # Configure quantization for GPU
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_threshold=6.0,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=False,
            )
            
            model = Gemma3nForConditionalGeneration.from_pretrained(
                GEMMA_CACHE_PATH if os.path.exists(GEMMA_CACHE_PATH) else "google/gemma-3n-e2b-it",
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory={0: "6GB"},
                low_cpu_mem_usage=True,
                local_files_only=os.path.exists(GEMMA_CACHE_PATH)
            ).eval()
            print("✅ Model loaded successfully on GPU with quantization")
        
        # Clean up configuration variables
        if 'quantization_config' in locals():
            cleanup_variables(quantization_config)
        
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU Memory after model load: {current_memory:.2f}GB")
            
        print(f"Model loading time: {time.time() - step5_start:.2f} seconds")
        step5_stats = print_system_stats("After Model Load")

        # Step 6: Move inputs to device
        print("\nStep 6: Moving Inputs to Device")
        step6_start = time.time()
        
        inputs = inputs.to(device)
        print(f"Inputs moved to {device}")
            
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU Memory after input move: {current_memory:.2f}GB")
        
        print(f"Input movement time: {time.time() - step6_start:.2f} seconds")
        step6_stats = print_system_stats("After Input Move")

        # Step 7: Generate response
        print("\nStep 7: Generating Response")
        step7_start = time.time()
        
        print("Starting generation...")
        if use_streaming and device != "cpu":
            print("🧠 Using layer streaming mode - layers will be loaded on-demand")
        elif device == "cpu":
            print("🖥️ Using CPU mode")
            
        # Monitor system before generation
        pre_gen_stats = print_system_stats("Pre-Generation")
        
        with torch.inference_mode():
            if torch.cuda.is_available():
                pre_gen_memory = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"GPU Memory before generation: {pre_gen_memory:.2f}GB")

            # declare max tokens
            max_tokens=200
            
            try:
                # Use simpler generation parameters that work with Gemma
                generation = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # Ensure generation tensor is materialized and moved to CPU
                if hasattr(generation, 'is_meta') and generation.is_meta:
                    print("Warning: Generation tensor is meta, attempting to materialize...")
                    generation = generation.detach()
                
                # Move to CPU safely and clean up GPU memory immediately
                if generation.device != torch.device('cpu'):
                    generation = generation.cpu()
                
                # Extract only the new tokens (skip the input)
                generation = generation[0][input_len:]
                
            except Exception as gen_error:
                print(f"❌ Generation failed: {gen_error}")
                cleanup_variables(model, processor, inputs)
                return f"Error during text generation: {str(gen_error)}"
            
            # Immediate cleanup of GPU tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                post_gen_memory = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"GPU Memory after generation: {post_gen_memory:.2f}GB")
                if use_streaming and device != "cpu":
                    print(f"🎯 Peak GPU usage with streaming: {post_gen_memory:.2f}GB")
        
        print(f"Generation time: {time.time() - step7_start:.2f} seconds")
        post_gen_stats = print_system_stats("Post-Generation")

        # Step 8: Decode and comprehensive cleanup
        print("\nStep 8: Decoding and Comprehensive Cleanup")
        step8_start = time.time()
        
        try:
            print("Decoding response...")
            response = processor.decode(generation, skip_special_tokens=True)
                
            # Clean up generation tensor immediately
            cleanup_variables(generation)
            
        except Exception as decode_error:
            print(f"❌ Decoding failed: {decode_error}")
            cleanup_variables(generation, model, processor, inputs)
            return f"Error during response decoding: {str(decode_error)}"
        
        print("Performing comprehensive memory cleanup...")
        pre_cleanup_stats = print_system_stats("Pre-Cleanup")
        
        if torch.cuda.is_available():
            pre_cleanup_memory = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU Memory before cleanup: {pre_cleanup_memory:.2f}GB")
        
        # Comprehensive cleanup
        cleanup_variables(model, processor, inputs)
        
        # Additional cleanup for streaming artifacts
        if use_streaming:
            offload_dir = os.path.join(os.path.dirname(__file__), 'model_streaming')
            if os.path.exists(offload_dir):
                try:
                    shutil.rmtree(offload_dir)
                    print("🗑️ Cleaned up streaming cache directory")
                except:
                    pass  # Non-critical if cleanup fails
        
        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            post_cleanup_memory = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU Memory after cleanup: {post_cleanup_memory:.2f}GB")
        
        print(f"Decode and cleanup time: {time.time() - step8_start:.2f} seconds")
        final_stats = print_system_stats("Final")
        
        # Summary statistics
        total_time = time.time() - start_time
        print(f"\n🏁 Processing Summary:")
        print(f"   Total Time: {total_time:.2f} seconds")
        print(f"   RAM Change: {final_stats['ram_used_gb'] - initial_stats['ram_used_gb']:+.2f}GB")
        print(f"   Peak RAM Usage: {max(step5_stats['ram_percent'], post_gen_stats['ram_percent']):.1f}%")
        print(f"   Final RAM Usage: {final_stats['ram_percent']:.1f}%")
        if device == "cpu":
            print(f"   CPU Processing: Successful without GPU acceleration")
        else:
            print(f"   GPU Processing: {'Streaming' if use_streaming else 'Standard'} mode")
            
        return response.strip()
        
    except Exception as e:
        print(f"❌ Error in generate_answer: {e}")
        import traceback
        traceback.print_exc()
        
        # Emergency cleanup
        print("\n🚨 Emergency cleanup procedure activated...")
        if 'model' in locals():
            cleanup_variables(model)
        if 'processor' in locals():
            cleanup_variables(processor)
        if 'inputs' in locals():
            cleanup_variables(inputs)
        if 'generation' in locals():
            cleanup_variables(generation)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
        
        # Print final system state for debugging
        error_stats = print_system_stats("After Error")
        
        logging.error(f"Error generating Gemma response: {e}")
        return f"Error generating response: {str(e)}"


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
        # Generate a response using Gemma
        gemma_response = generate_gemma_response(combined_context, query)
        print(f"Gemma response generated in {time.time() - gemma_start:.2f} seconds")
        
        # Clean up context data
        cleanup_variables(combined_context)
        
        print("\nStep 9: Preparing Final Response")
        response_start = time.time()
        
        # Add the Gemma response to the results
        final_results = [{
            'text': gemma_response,
            'source': 'AI Assistant',
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
