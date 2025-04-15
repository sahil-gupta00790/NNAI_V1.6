# src/qa_system.py

from src.rag_pipeline import load_faiss_index, DEFAULT_EMBED_MODEL

from llama_index.core import Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.postprocessor import SentenceTransformerRerank

import logging
import torch
import os
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

# Configuration
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('llama_index.core.chat_engine').setLevel(logging.INFO)
logging.getLogger('llama_index.core.llms').setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)

# Model Names (Consider making these configurable via environment variables too)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBED_MODEL)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

# --- Global Settings Configuration (Handles LLM setup) ---
logging.info("Configuring global LlamaIndex settings...")
try:
    # Embedding Model
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    logging.info(f"Using embedding model: {Settings.embed_model.model_name}")

    # LLM
    logging.info(f"Setting up LLM: {LLM_MODEL_NAME}")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logging.warning("HF_TOKEN environment variable not set. LLM loading might fail or be restricted.")

    # --- Check for available VRAM and decide device_map ---
    device_map = "auto" # Default to auto (GPU if possible)
    try:
        if torch.cuda.is_available():
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"Detected GPU VRAM: {total_mem_gb:.2f} GB")
            # Set a threshold (e.g., < 6GB VRAM) to force CPU for smaller GPUs with larger models
            if total_mem_gb < 6.0 and "1B" not in LLM_MODEL_NAME: # Adjust threshold as needed
                 logging.warning(f"Low VRAM detected ({total_mem_gb:.2f}GB). Forcing LLM to CPU ('device_map=cpu') to prevent OOM errors. RAG responses might be slow.")
                 device_map = "cpu"
        else:
            logging.info("No CUDA GPU detected. LLM will run on CPU.")
            device_map = "cpu"
    except Exception as gpu_err:
         logging.warning(f"Could not accurately determine GPU VRAM: {gpu_err}. Using device_map='auto'.")
    # --- End device_map logic ---

    Settings.llm = HuggingFaceLLM(
        model_name=LLM_MODEL_NAME,
        tokenizer_name=LLM_MODEL_NAME,
        context_window=131072, # Adjust if needed based on model
        max_new_tokens=512, # Increase if responses are cut short
        model_kwargs={}, # Add quantization config here if needed later
        generate_kwargs={"temperature": 0.6, "do_sample": True}, # Slightly lower temp for more factual
        # token=hf_token, # Pass token if needed for gated models
        device_map=device_map, # Use determined device map
    )
    logging.info(f"LLM '{LLM_MODEL_NAME}' configured successfully on device_map='{device_map}'.")
except Exception as e:
    logging.error(f"Failed during Global Settings configuration: {e}", exc_info=True)
    Settings.llm = None # Ensure LLM is None if setup fails

# --- IMPROVED System Prompt ---
system_prompt = """
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

# --- Reranker Initialization ---
try:
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL_NAME,
        top_n=5 # Increased from 3 to potentially get better context after reranking
    )
    logging.info(f"Reranker '{RERANKER_MODEL_NAME}' initialized successfully (top_n=5).")
except Exception as e:
    logging.error(f"Failed to initialize Reranker: {e}", exc_info=True)
    reranker = None

# --- Function to create chat engine (used by backend) ---
def create_chat_engine(persist_dir="storage"):
    """Loads the index and creates the chat engine."""
    if Settings.llm is None or reranker is None:
        logging.error("Cannot create chat engine: LLM or Reranker failed to initialize.")
        return None

    logging.info("Loading vector index for chat engine...")
    index = load_faiss_index(persist_dir=persist_dir, embed_model_name=EMBEDDING_MODEL_NAME)
    if index is None:
        logging.error("Could not load the index for chat engine.")
        return None
    logging.info("Index loaded successfully for chat engine.")

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900) # Consider adjusting limit

    chat_engine = ContextChatEngine.from_defaults(
        retriever=index.as_retriever(similarity_top_k=10), # Retrieve more initially
        llm=Settings.llm, # Use globally configured LLM
        memory=memory,
        system_prompt=system_prompt,
        node_postprocessors=[reranker] # Apply reranker
    )
    # Alternative using index.as_chat_engine (simpler setup)
    # chat_engine = index.as_chat_engine(
    #     chat_mode="context", # Use "condense_plus_context" for potentially better history handling
    #     memory=memory,
    #     system_prompt=system_prompt,
    #     similarity_top_k=10, # Retrieve more initially
    #     node_postprocessors=[reranker] # Apply reranker
    # )

    logging.info("Chat engine created successfully.")
    return chat_engine

# --- Main Chat Loop (for console testing) ---
if __name__ == "__main__":
    # Check prerequisites
    if Settings.llm is None:
        print("\nERROR: LLM failed to initialize during setup. Cannot start chat system.")
    elif reranker is None:
        print("\nERROR: Reranker failed to initialize. Cannot start chat system.")
    else:
        console = Console()
        console.print("[bold cyan]Starting Console Neural Network RAG Chat System...[/bold cyan]")
        console.print(f"[cyan]LLM: {LLM_MODEL_NAME} (Device Map: {device_map})[/cyan]")
        console.print(f"[cyan]Embedding: {EMBEDDING_MODEL_NAME}[/cyan]")
        console.print(f"[cyan]Reranker: {RERANKER_MODEL_NAME} (top_n=5)[/cyan]")

        chat_engine_instance = create_chat_engine(persist_dir="storage")

        if chat_engine_instance is None:
             console.print("[bold red]ERROR: Failed to create chat engine instance.[/bold red]")
        else:
            console.print("[bold green]Chat engine ready.[/bold green]")
            console.print("[cyan]Type 'exit' or 'quit' to end the chat.[/cyan]\n")

            while True:
                try:
                    user_input = input("You: ")
                    if user_input.lower() in ["exit", "quit"]:
                        console.print("[bold cyan]Exiting chat. Goodbye![/bold cyan]")
                        break
                    if not user_input:
                        continue

                    console.print(f"[grey50]Processing...[/grey50]", end='\r')
                    # --- Use chat, not stream_chat, to get source_nodes easily for console ---
                    # streaming_response = chat_engine_instance.stream_chat(user_input)
                    # Use chat() for non-streaming to easily get source nodes after response
                    response = chat_engine_instance.chat(user_input)
                    console.print("Processing... Done.") # Clear processing message

                    # --- Display Response using Rich Markdown ---
                    console.print(f"[bold green]Assistant:[/bold green]")
                    md = Markdown(response.response)
                    console.print(md)
                    console.print("") # Add a newline after the response

                    # --- Display Source Nodes (Logic remains the same for console) ---
                    source_nodes = response.source_nodes
                    if source_nodes:
                        console.print("[bold yellow]Sources Considered (Post-Reranking):[/bold yellow]")
                        seen_sources = set()
                        # Limit displayed sources to avoid overwhelming console
                        for i, node in enumerate(source_nodes):
                            if i >= 5: # Show top 5 sources used
                                console.print(f"- ... and {len(source_nodes) - 5} more.")
                                break
                            # Prefer title, fallback to filename
                            title = node.metadata.get('title', None)
                            file_name = node.metadata.get('file_name', None)
                            display_name = title if title and title != "N/A" else file_name
                            node_id = node.node_id # For uniqueness if names are identical
                            source_key = f"{display_name}_{node_id}" # Use a more unique key

                            # Ensure we only display unique sources based on display_name and node_id
                            if display_name and source_key not in seen_sources:
                                score_str = f"(Score: {node.score:.3f})" if node.score else ""
                                console.print(f"- {display_name} {score_str}")
                                seen_sources.add(source_key)

                        if not seen_sources:
                            console.print("- (No specific source titles/filenames found in metadata for top nodes)")
                    else:
                         console.print("[yellow]No source nodes were retrieved or passed post-processing.[/yellow]")
                    print("") # Add newline after sources

                except Exception as e:
                    logging.error(f"An error occurred during console chat: {e}", exc_info=True)
                    console.print(f"\n[bold red]Error: {e}[/bold red]")
                    console.print("[yellow]You may need to restart the script.[/yellow]")

