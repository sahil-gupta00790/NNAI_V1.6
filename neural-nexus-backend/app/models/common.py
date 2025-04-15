# app/models/common.py
from pydantic import BaseModel
from typing import Any # Import Any for flexible dict content

class TaskResponse(BaseModel):
    """Response model for endpoints that start a task."""
    task_id: str
    status: str # e.g., "PENDING"

class TaskStatus(BaseModel):
    """Response model for task status endpoints."""
    task_id: str
    status: str # PENDING, STARTED, PROGRESS, SUCCESS, FAILURE, REVOKED, etc.
    progress: float | None = None # Overall progress (0.0 to 1.0)
    message: str | None = None # User-friendly status message

    # --- Add this field ---
    info: dict[str, Any] | None = None # Holds metadata during PROGRESS state
    # Examples of keys within 'info': progress, message, fitness_history, etc.
    # ----------------------

    result: dict[str, Any] | None = None # Holds final result on SUCCESS or error details on FAILURE
    # Examples of keys within 'result': final_model_path, best_fitness, error, message, fitness_history, etc.

    # Ensure backwards compatibility or handle potential None values carefully
    model_config = {
        "extra": "ignore" # Ignore extra fields from backend if any mismatch
    }

