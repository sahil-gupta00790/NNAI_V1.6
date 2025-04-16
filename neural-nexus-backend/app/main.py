# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Import initialization/status functions for RAG
from app.api.endpoints.advisor import initialize_rag_system, get_rag_status
# Import routers
from app.api.endpoints import evolver # Evolver router
from app.api.endpoints import advisor as advisor_router_module # Advisor router
from app.api.endpoints.analysis import router as analysis_router # NEW: Analysis router
from app.api.endpoints import gemini as gemini_router
# Import settings
from app.core.config import settings
import logging
from contextlib import asynccontextmanager # Keep import if needed elsewhere, not used here

# --- Configure Logging ---
# (Logging setup remains the same)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- FastAPI App Instantiation ---
# Using startup/shutdown events, not lifespan here
app = FastAPI(
    title="NeuroForge",
    version="0.1.0"
)

# --- Startup Event (Initializes RAG) ---
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup...")
    logger.info("Initializing RAG system via startup event...")
    try:
        await initialize_rag_system()
        logger.info("RAG system initialization sequence finished.")
    except Exception as e:
        logger.error(f"FATAL: RAG system initialization failed: {e}", exc_info=True)
        # Depending on severity, you might want to stop the app
        # raise RuntimeError("Failed to initialize RAG system.") from e

# --- Shutdown Event ---
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down...")
    # Add any global cleanup logic here if needed

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
logger.info("Including API routers...")
# Advisor Router
app.include_router(advisor_router_module.router, prefix="/api/v1/advisor", tags=["Advisor"])
logger.info("Included Advisor router at /api/v1/advisor")
# Evolver Router
app.include_router(evolver.router, prefix="/api/v1/evolver", tags=["Evolver"])
logger.info("Included Evolver router at /api/v1/evolver")
# --- NEW: Include Analysis Router ---
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])
logger.info("Included Analysis router at /api/v1/analysis") # Log inclusion
# ----------------------------------
app.include_router(gemini_router.router, prefix="/api/v1/gemini", tags=["Gemini Chat"])
logger.info("Included Gemini Chat router at /api/v1/gemini")

# --- Root Endpoint ---
@app.get("/")
def read_root():
    # (Root endpoint remains the same)
    rag_status = get_rag_status()
    return {
        "message": "Welcome to the Neural Nexus AI Platform API",
        "rag_status": rag_status
    }

# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    # (Health check remains the same)
    rag_status = get_rag_status()
    status_value = rag_status.get("status", "unknown") if isinstance(rag_status, dict) else "unknown"
    return {"status": "ok", "rag_status": status_value}

# --- Optional: Add uvicorn run block for direct execution ---
# import uvicorn
# if __name__ == "__main__":
#     logger.info("Starting Uvicorn server directly...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)
