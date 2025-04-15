# src/rag_pipeline.py (or app/rag_utils/rag_pipeline.py)

# Core LlamaIndex imports
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings as LlamaSettings # Use alias
)
# Correct Imports for Node Parser and Transformation Base Class
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import Document
from typing import List

# FAISS integration
from llama_index.vector_stores.faiss import FaissVectorStore

# HuggingFace Embedding integration
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# FAISS library
import faiss

# Standard libraries
import os
import logging
import json
import shutil
import time

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set level for this module

# --- Default Model Names and Paths ---
DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
STORAGE_DIR = "storage" # Default persist_dir
DATA_DIR = "data/research_papers" # Default data dir

# --- Text Cleaning Function ---
def clean_text_for_utf8(text: str) -> str:
    """Removes or replaces characters that cause UTF-8 encoding errors."""
    if not isinstance(text, str):
        return ""
    return text.encode('utf-8', errors='ignore').decode('utf-8')

# --- Helper Function to Load Docs with Metadata ---
def load_documents_with_metadata(data_dir=DATA_DIR):
    """Loads PDFs, attaches metadata, and cleans text content."""
    pdf_dir = os.path.join(data_dir, "raw_pdfs")
    metadata_file = os.path.join(data_dir, "arxiv_metadata.json")

    if not os.path.isdir(pdf_dir):
        logger.error(f"PDF directory not found: {pdf_dir}")
        return []

    if not os.path.isfile(metadata_file):
        logger.warning(f"Metadata JSON file not found: {metadata_file}. Loading PDFs without detailed metadata.")
        metadata_lookup = {}
    else:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
                # Assuming local_pdf_path does NOT include the base data_dir
                metadata_lookup = {os.path.join(pdf_dir, os.path.basename(meta.get("local_pdf_path", ""))): meta
                                   for meta in all_metadata if meta.get("local_pdf_path")}
        except Exception as e:
            logger.error(f"Failed to load or parse metadata file {metadata_file}: {e}. Proceeding without detailed metadata.")
            metadata_lookup = {}

    def file_metadata_func(file_path: str):
        """Creates metadata dict and cleans string values."""
        base_meta = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
        }
        arxiv_meta_raw = metadata_lookup.get(file_path) # Use full path for lookup
        if arxiv_meta_raw:
            base_meta["title"] = clean_text_for_utf8(arxiv_meta_raw.get("title", "N/A"))
            base_meta["authors"] = clean_text_for_utf8(", ".join(arxiv_meta_raw.get("authors", [])))
            base_meta["published"] = clean_text_for_utf8(arxiv_meta_raw.get("published", "N/A"))
            base_meta["arxiv_id"] = clean_text_for_utf8(arxiv_meta_raw.get("arxiv_id", "N/A"))
            base_meta["pdf_url"] = clean_text_for_utf8(arxiv_meta_raw.get("pdf_url", "N/A"))
        else:
            base_meta["title"] = "N/A (Metadata not found)"
            logger.debug(f"Metadata not found for file: {file_path}")
        return base_meta

    try:
        logger.info(f"Loading documents from directory: {pdf_dir}")
        reader = SimpleDirectoryReader(
            pdf_dir,
            file_metadata=file_metadata_func,
            # filename_as_id=True, # Consider using filename as doc ID
        )
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents with metadata attached.")
        if len(documents) == 0:
             logger.warning(f"No documents were loaded from {pdf_dir}. Check if PDFs exist and are readable.")
    except Exception as e:
        logger.error(f"Failed during SimpleDirectoryReader.load_data(): {e}", exc_info=True)
        return []

    cleaned_documents = []
    logger.info("Cleaning document content...")
    docs_cleaned_count = 0
    for i, doc in enumerate(documents):
        # ...(Cleaning logic remains the same)...
        try:
            original_content = doc.get_content()
            if not original_content or original_content.isspace():
                logger.warning(f"Document {i} (file: {doc.metadata.get('file_name', 'N/A')}) has empty or whitespace-only content. Skipping.")
                continue # Skip empty documents
            cleaned_content = clean_text_for_utf8(original_content)
            if cleaned_content != original_content: docs_cleaned_count += 1
            doc.set_content(cleaned_content)
            cleaned_documents.append(doc)
        except Exception as e: logger.error(f"Error cleaning doc {i}... skipping."); continue # Skip doc on error

    logger.info(f"Finished cleaning. Returning {len(cleaned_documents)} non-empty documents.")
    return cleaned_documents

# --- UPDATED Check Storage Integrity Function ---
def check_storage_integrity(persist_dir=STORAGE_DIR):
    """Checks if storage directory has required files in the standard structure."""
    # Structure usually involves component stores + vector_store subdir
    required_files_dirs = [
        'docstore.json',
        'index_store.json',
        'graph_store.json', # Optional, might not exist if graph store isn't used
        'vector_store/vector_store.json', # Vector store metadata
        'vector_store/vector_store.faiss' # FAISS binary
    ]
    missing_items = []
    logger.info(f"Checking integrity of storage directory: {persist_dir}")

    for item in required_files_dirs:
        path = os.path.join(persist_dir, item)
        if not os.path.exists(path):
            # Be less strict about graph_store.json
            if item != 'graph_store.json':
                missing_items.append(item)
                logger.warning(f"Missing required file/dir: {path}")
            else:
                logger.info(f"Optional file {path} not found (this might be ok).")
        else:
            logger.info(f"Found: {path}")

    if not missing_items:
        logger.info("Storage integrity check passed (basic).")
        return True
    else:
        logger.error(f"Storage integrity check failed. Missing required items: {missing_items}")
        return False

# --- UPDATED Index Building Function ---
def build_faiss_index(window_size=3):
    persist_dir = STORAGE_DIR
    logger.info(f"Starting FAISS index build. Embed Model: {DEFAULT_EMBED_MODEL}")
    logger.info(f"Index will be persisted to: '{persist_dir}'")

    # --- Backup Logic (Same as previous modification) ---
    try:
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            backup_dir_name = f"{os.path.basename(persist_dir)}_backup_{int(time.time())}"
            backup_path = os.path.join(os.path.dirname(persist_dir) or '.', backup_dir_name)
            logger.info(f"Backing up existing storage from '{persist_dir}' to '{backup_path}'")
            shutil.copytree(persist_dir, backup_path, dirs_exist_ok=True)
            logger.info("Backup copy successful. Clearing original storage directory...")
            for item_name in os.listdir(persist_dir):
                item_path = os.path.join(persist_dir, item_name)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path): os.unlink(item_path)
                    elif os.path.isdir(item_path): shutil.rmtree(item_path)
                except Exception as e: logger.error(f"Failed to delete {item_path}: {e}")
            logger.info(f"Original storage directory '{persist_dir}' cleared.")
        else:
             logger.info(f"Storage directory '{persist_dir}' is empty or does not exist. Skipping backup/clear.")
    except Exception as e:
        logger.error(f"Error during backup/clear: {e}", exc_info=True)
        logger.error("Stopping index build due to backup/clear failure.")
        return
    # --- End Backup Logic ---

    os.makedirs(persist_dir, exist_ok=True)

    # 1. Initialize Embedding Model
    try:
        logger.info(f"Configuring LlamaSettings.embed_model: {DEFAULT_EMBED_MODEL}")
        LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL)
        from transformers import AutoConfig
        try: embed_config = AutoConfig.from_pretrained(DEFAULT_EMBED_MODEL); embed_dim = embed_config.hidden_size
        except Exception: logger.warning(f"Could not auto-detect embed dim. Assuming 768."); embed_dim = 768
        logger.info(f"Embedding model set globally. Dimension: {embed_dim}")
    except Exception as e: logger.error(f"Failed to load embed model: {e}", exc_info=True); return

    # 2. Define Node Parser
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # 3. Load AND CLEAN Documents
    documents = load_documents_with_metadata()
    if not documents:
        logger.error("No documents loaded after cleaning. Aborting index build.")
        return

    # 4. Create FAISS Index and Vector Store
    try:
        logger.info(f"Initializing FAISS index (IndexFlatL2) with dimension: {embed_dim}")
        # Initialize the FAISS index object directly
        faiss_index = faiss.IndexFlatL2(embed_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # 5. Create Storage Context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 6. Build VectorStoreIndex FROM documents
        logger.info("Building VectorStoreIndex (parsing, embedding, indexing)...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser],
            show_progress=True,
        )
        # --- Log node count AFTER building ---
        node_count = len(index.docstore.docs)
        logger.info(f"Index build completed. Final node count in docstore: {node_count}")
        if node_count == 0:
            logger.error("Index build resulted in 0 nodes. Check document loading, parsing, and content.")
            logger.warning("Persistence might fail or result in an unusable index.")
            # Optionally return here if 0 nodes is critical error
            # return

        # 7. Persist index components (RELY ON THIS)
        logger.info(f"Persisting index components to '{persist_dir}'")
        index.storage_context.persist(persist_dir=persist_dir)
        logger.info("Persistence call finished.")

        # 8. REMOVED Explicit FAISS binary save - handled by vector_store persistence

        # 9. Verify storage integrity (using updated function)
        if check_storage_integrity(persist_dir):
            logger.info("Index built and persisted successfully.")
        else:
            # This might still happen if 0 nodes were indexed
            logger.error("Index build finished, but storage integrity check failed. Check logs and node count.")

    except Exception as e:
        logger.error(f"Error during index building/persistence: {e}", exc_info=True)


# --- UPDATED Index Loading Function ---
def load_faiss_index(persist_dir=STORAGE_DIR, embed_model_name=DEFAULT_EMBED_MODEL):
    logger.info(f"Attempting to load index from '{persist_dir}' (Embed Model: {embed_model_name})")

    # Check storage integrity first (using updated function)
    if not check_storage_integrity(persist_dir):
        logger.error("Cannot load index: Storage integrity check failed.")
        return None

    # Configure embedding model
    try:
        logger.info(f"Configuring LlamaSettings.embed_model: '{embed_model_name}'")
        LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    except Exception as e: logger.error(f"Failed to load embed model: {e}", exc_info=True); return None

    # Load the index from storage (RELY ON THIS)
    try:
        logger.info("Loading LlamaIndex storage context and index from disk...")
        # Let LlamaIndex load all components from the directory structure
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        # load_index_from_storage uses the loaded storage context
        index = load_index_from_storage(storage_context)
        # Verify vector store loaded correctly
        if not isinstance(index.vector_store, FaissVectorStore):
            logger.error("Loaded index does not contain a FaissVectorStore!")
            return None
        if index.vector_store._faiss_index is None: # Check internal attribute if needed
            logger.error("Loaded FaissVectorStore does not have an internal FAISS index instance!")
            return None

        logger.info(f"FAISS index loaded successfully via LlamaIndex. Vector store type: {type(index.vector_store)}")
        return index

    except Exception as e:
        logger.error(f"Failed to load index from '{persist_dir}': {e}", exc_info=True)
        return None

# --- Main execution block ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Executing rag_pipeline.py script directly (for index building).")

    # --- Install pypdf if SimpleDirectoryReader needs it ---
    try:
        import pypdf
    except ImportError:
        logger.warning("`pypdf` package not found. Attempting to install...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
            logger.info("`pypdf` installed successfully.")
        except Exception as install_err:
            logger.error(f"Failed to install `pypdf`: {install_err}. PDF parsing might fail.")
    # -----------------------------------------------------

    build_faiss_index()

    logger.info("\nDirect build process complete.")

