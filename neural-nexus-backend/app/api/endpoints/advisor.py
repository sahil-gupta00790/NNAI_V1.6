# app/api/endpoints/advisor.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.models.advisor import AdvisorQuery, AdvisorResponse
from app.core.config import settings
import logging
import os
import torch
import time

# --- LlamaIndex Imports ---
from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex, PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine, CondenseQuestionChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine # For type hint
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# --- Globals for RAG System ---
rag_index: VectorStoreIndex | None = None
rag_chat_engine: BaseChatEngine | None = None # Use Base type hint
rag_reranker: SentenceTransformerRerank | None = None
rag_initialization_status = {"status": "pending", "message": "Not initialized yet."}

logger = logging.getLogger(__name__)

# --- IMPROVED System Prompt ---
system_prompt_text = """
You are an expert AI assistant specializing in neural networks, AI research, and related topics. Your goal is to provide accurate, comprehensive, and well-formatted answers.

Follow these guidelines:
1.  **Prioritize Context:** Base your answer primarily on the provided 'Context information'.
2.  **Use General Knowledge:** If the context is insufficient or doesn't fully answer the query, supplement your answer using your general knowledge, but clearly indicate when you are doing so (e.g., "Based on the context..., and adding some general information...").
3.  **Handle Missing Context:** If the context is entirely irrelevant or missing for the query, state that clearly (e.g., "The provided documents don't contain information on this specific topic.") and then provide a general answer based on your knowledge.
4.  **Cite Sources:** When using information derived *directly* from the context, cite the source document(s) by mentioning their title or filename (e.g., 'According to "Attention Is All You Need"...' or 'The document xyz.pdf states...'). Use the 'title' or 'file_name' metadata provided with the context chunks. Do not cite when using only general knowledge.
5.  **Format Clearly:** Structure complex answers using Markdown lists (e.g., using '-' or numbered lists with proper line breaks). Use bold text (`**bold**`) for emphasis where appropriate.
6.  **Clarify Ambiguity:** If the user's query is ambiguous, ask a specific clarifying question before providing a detailed answer.
7.  **Tone and History:** Maintain a helpful, expert tone and consider the conversation history for follow-up questions.
"""

# --- RAG System Initialization Function (Called by main.py on startup) ---
async def initialize_rag_system():
    global rag_index, rag_chat_engine, rag_reranker, rag_initialization_status, system_prompt_text # Ensure prompt is global or passed in
    logger.info("Starting RAG system initialization...")
    rag_initialization_status = {"status": "initializing", "message": "Starting..."}
    local_rag_index = None
    local_rag_reranker = None
    local_llm = None
    local_embed_model = None

    # --- Check for required storage ---
    faiss_path = os.path.join(settings.RAG_STORAGE_DIR, "vector_store.faiss")
    if not os.path.exists(settings.RAG_STORAGE_DIR) or not os.path.exists(faiss_path):
        msg = f"RAG storage directory ('{settings.RAG_STORAGE_DIR}') or '{faiss_path}' not found. Please build the index first."
        logger.error(msg)
        rag_initialization_status = {"status": "error", "message": msg}
        return

    # --- Configure LlamaIndex Settings ---
    try:
        logger.info(f"Configuring Embed Model: {settings.EMBEDDING_MODEL_NAME}")
        local_embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)

        logger.info(f"Configuring LLM: {settings.LLM_MODEL_NAME}")
        model_kwargs = {}
        if settings.HF_TOKEN: model_kwargs["token"] = settings.HF_TOKEN

        device_map_setting = "auto"
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Detected GPU VRAM: {vram_gb:.2f} GB")
            # Simple check - adjust models/threshold as needed
            if vram_gb < 6 and ("Llama-3" in settings.LLM_MODEL_NAME):
                 logger.warning(f"Low VRAM detected ({vram_gb:.2f}GB). Forcing LLM to CPU ('device_map=cpu').")
                 # device_map_setting = "cpu" # Uncomment to force if 'auto' fails
        else:
             logger.info("No CUDA GPU detected. LLM running on CPU.")
             device_map_setting = "cpu"

        local_llm = HuggingFaceLLM(
            model_name=settings.LLM_MODEL_NAME,
            tokenizer_name=settings.LLM_MODEL_NAME,
            context_window=8192,
            max_new_tokens=512,
            model_kwargs=model_kwargs,
            query_wrapper_prompt=PromptTemplate("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
            generate_kwargs={"temperature": 0.6, "do_sample": True},
            device_map=device_map_setting,
        )
        logger.info("LLM configured.")

        # Assign to global Settings AFTER successful local creation
        LlamaSettings.llm = local_llm
        LlamaSettings.embed_model = local_embed_model

    except Exception as e:
        logger.error(f"Failed during LlamaIndex Settings configuration: {e}", exc_info=True)
        rag_initialization_status = {"status": "error", "message": f"LLM/Embed model config failed: {e}"}
        return

    # --- Load FAISS Index and Reranker ---
    try:
        logger.info(f"Loading FAISS binary from: {faiss_path}")
        faiss_index = faiss.read_index(faiss_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        logger.info("Loading storage context...")
        storage_context = StorageContext.from_defaults(
            persist_dir=settings.RAG_STORAGE_DIR, vector_store=vector_store
        )

        logger.info("Loading VectorStoreIndex...")
        local_rag_index = load_index_from_storage(storage_context)
        logger.info("Index loaded successfully.")

        logger.info(f"Initializing Reranker: {settings.RERANKER_MODEL_NAME}")
        local_rag_reranker = SentenceTransformerRerank(
            model=settings.RERANKER_MODEL_NAME, top_n=5 # Increased top_n
        )
        logger.info("Reranker initialized.")

        # Assign to global vars AFTER successful local creation
        rag_index = local_rag_index
        rag_reranker = local_rag_reranker

    except Exception as e:
        logger.error(f"Failed to load index or initialize reranker: {e}", exc_info=True)
        rag_initialization_status = {"status": "error", "message": f"Index/Reranker loading failed: {e}"}
        return

    # --- Create Chat Engine ---
    if rag_index and rag_reranker and LlamaSettings.llm:
        try:
            logger.info("Creating Chat Engine...")
            memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

            # Use the improved system prompt
            chat_engine = rag_index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                system_prompt=system_prompt_text, # <-- Use the improved prompt
                node_postprocessors=[rag_reranker],
                similarity_top_k=10,
            )
            logger.info("Chat Engine created successfully.")
            # Assign to global var
            rag_chat_engine = chat_engine # Assign to the global variable
            rag_initialization_status = {"status": "ready", "message": "RAG system initialized successfully."}
        except Exception as e:
            logger.error(f"Failed to create Chat Engine: {e}", exc_info=True)
            rag_initialization_status = {"status": "error", "message": f"Chat engine creation failed: {e}"}
    else:
         msg = "Cannot create chat engine due to previous initialization failures."
         logger.error(msg)
         rag_initialization_status = {"status": "error", "message": msg}

# --- Helper Function (remains the same) ---
def get_rag_status():
    """Returns the current status of the RAG system initialization."""
    return rag_initialization_status

# --- API Endpoint Router ---
router = APIRouter()

# --- Chat Endpoint (Modified) ---
@router.post("/chat", response_model=AdvisorResponse)
async def chat_with_advisor(query: AdvisorQuery):
    """
    Endpoint to interact with the RAG Advisor.
    Uses the pre-initialized RAG system and returns sources.
    """
    start_time = time.time()
    logger.info(f"Received RAG query: '{query.query}'")

    # Use the global chat engine instance
    global rag_chat_engine, rag_initialization_status

    if rag_chat_engine is None or rag_initialization_status["status"] != "ready":
        logger.error(f"RAG chat engine not ready. Status: {rag_initialization_status}")
        raise HTTPException(
            status_code=503,
            detail=f"RAG system is not available: {rag_initialization_status.get('message', 'Initialization error.')}"
        )

    if not query.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        logger.info("Processing query with chat engine...")
        # Use achat for async operation
        response = await rag_chat_engine.achat(query.query)

        response_text = str(response.response or "Sorry, I couldn't generate a response.")
        logger.info("RAG response generated.")

        # --- Extract Source Information ---
        sources = []
        seen_sources = set() # To store unique display names
        if response.source_nodes:
            logger.info(f"Retrieved {len(response.source_nodes)} source nodes (post-reranking).")
            for node in response.source_nodes:
                title = node.metadata.get('title', None)
                file_name = node.metadata.get('file_name', None)
                display_name = title if title and title != "N/A" else file_name
                if display_name and display_name not in seen_sources:
                    sources.append(display_name)
                    seen_sources.add(display_name)
            logger.info(f"Extracted unique sources: {sources}")
        else:
             logger.info("No source nodes returned by chat engine.")

        end_time = time.time()
        logger.info(f"RAG query processed in {end_time - start_time:.2f} seconds.")

        # Return response text and the list of extracted sources
        return AdvisorResponse(response=response_text, sources=sources if sources else None)

    except Exception as e:
        logger.error(f"Error processing RAG query with chat engine: {e}", exc_info=True)
        error_detail = f"Error processing RAG query: {str(e)}"
        if "out of memory" in error_detail.lower():
             error_detail += " (Potential OOM Error - check resources/model size)"
        raise HTTPException(status_code=500, detail=error_detail)

# --- Endpoint to Reset Chat History (Remains the same) ---
@router.post("/reset_chat")
async def reset_chat_history():
    """ Resets the conversation history in the RAG chat engine's memory. """
    global rag_chat_engine # Use global engine
    logger.info("Received request to reset RAG chat history.")
    if rag_chat_engine and hasattr(rag_chat_engine, 'reset'):
        try:
            rag_chat_engine.reset()
            logger.info("RAG chat history reset successfully.")
            return {"message": "Chat history reset successfully."}
        except Exception as e:
             logger.error(f"Failed to reset chat history: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Failed to reset chat history: {e}")
    elif rag_chat_engine:
         logger.warning("Chat engine does not have a 'reset' method. Cannot reset memory.")
         raise HTTPException(status_code=501, detail="Chat engine type does not support history reset.")
    else:
         logger.error("Cannot reset history: Chat engine not initialized.")
         raise HTTPException(status_code=503, detail="RAG system not available.")

