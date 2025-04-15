# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Import the specific lifespan function from advisor, possibly using an alias
from app.api.endpoints.advisor import initialize_rag_system, get_rag_status
# Import other routers
from app.api.endpoints import evolver # Keep evolver router
from app.core.config import settings
import logging
from contextlib import asynccontextmanager # Keep if using lifespan directly here, but we import from advisor

# --- Configure Logging ---
# (Logging setup remains the same)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- FastAPI App Instantiation with Lifespan ---
# Pass the lifespan context manager imported from advisor.py
# This replaces the need for @app.on_event("startup") and @app.on_event("shutdown") here
app = FastAPI(
    title="Neural Nexus AI Platform",
    version="0.1.0" 
)

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup...")
    logger.info("Initializing RAG system via startup event...") # Added log clarity
    try:
        # Call the imported initialization function
        await initialize_rag_system()
        logger.info("RAG system initialization sequence finished.")
    except Exception as e:
        logger.error(f"FATAL: RAG system initialization failed: {e}", exc_info=True)
        # Optionally re-raise to prevent app start on critical failure
        # raise RuntimeError("Failed to initialize RAG system.") from e

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down...")
    # Add any cleanup logic here if needed

# --- CORS Configuration ---
# (CORS config remains the same)
origins = settings.ALLOWED_ORIGINS.split(",") if settings.ALLOWED_ORIGINS else ["*"]
logger.info(f"Configuring CORS for origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routers ---
# Import the router object itself from advisor
from app.api.endpoints import advisor as advisor_router_module
app.include_router(advisor_router_module.router, prefix="/api/v1/advisor", tags=["advisor"])
app.include_router(evolver.router, prefix="/api/v1/evolver", tags=["evolver"])

# --- Root Endpoint ---
# (Root endpoint remains the same, still uses get_rag_status from advisor)
@app.get("/")
def read_root():
    rag_status = get_rag_status()
    return {
        "message": "Welcome to the Neural Nexus AI Platform API",
        "rag_status": rag_status
    }

# --- Health Check Endpoint (Optional) ---
# (Health check remains the same, still uses get_rag_status from advisor)
@app.get("/health")
def health_check():
    rag_status = get_rag_status()
    # Ensure status key exists before accessing
    status_value = rag_status.get("status", "unknown") if isinstance(rag_status, dict) else "unknown"
    return {"status": "ok", "rag_status": status_value}

# --- Optional: Add uvicorn run block for direct execution (if needed) ---
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

