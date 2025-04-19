# app/api/endpoints/evolver.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import FileResponse # For downloads
# NEW: Import BaseModel for response model
from pydantic import BaseModel
from typing import Annotated, Optional
import json
from app.models.common import TaskResponse, TaskStatus
from app.models.evolver import EvolverConfig
from tasks.evolution_tasks import run_evolution_task
from celery.result import AsyncResult
from app.core.celery_app import celery_app
from app.core.config import settings
import os
import uuid
import logging
from werkzeug.utils import secure_filename
import aioredis

logger = logging.getLogger(__name__)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULT_DIR, exist_ok=True)

router = APIRouter()

# --- Helper for Secure Filename (Unchanged) ---
def generate_secure_path(original_filename: str, sub_dir: str = "") -> tuple[str, str]:
    if not original_filename: raise ValueError("Original filename cannot be empty.")
    base_name = secure_filename(original_filename)
    if not base_name: base_name = "unnamed_file"
    unique_prefix = str(uuid.uuid4())[:8]
    secure_name = f"{unique_prefix}_{base_name}"
    target_dir = os.path.join(settings.UPLOAD_DIR, sub_dir)
    os.makedirs(target_dir, exist_ok=True)
    full_path = os.path.join(target_dir, secure_name)
    # Check for collisions - less likely but good practice
    attempts = 0
    while os.path.exists(full_path) and attempts < 5:
        unique_prefix = str(uuid.uuid4())[:8]
        secure_name = f"{unique_prefix}_{base_name}"
        full_path = os.path.join(target_dir, secure_name)
        attempts += 1
    if os.path.exists(full_path): # Extremely unlikely after 5 attempts
        raise IOError(f"Could not generate unique filename for {original_filename} after multiple attempts.")
    logger.debug(f"Generated secure path: {full_path} for original: {original_filename}")
    return full_path, secure_name

# --- NEW: Response Model for Termination ---
class TerminateResponse(BaseModel):
    message: str
    task_id: str
# --- End New Model ---

@router.post("/start", response_model=TaskResponse)
async def start_evolution(
    model_definition: Annotated[UploadFile, File(...)],
    config_json: Annotated[str, Form(...)],
    task_evaluation: Annotated[Optional[UploadFile], File()] = None,
    initial_weights: Annotated[Optional[UploadFile], File()] = None,
    use_standard_eval: Annotated[bool, Form(...)] = False,
):
    logger.info("Received request to start evolution task.")
    logger.debug(f"Backend Received Raw config_json snippet: {config_json[:200]}...") # Avoid logging huge configs

    # --- Config Validation (Unchanged) ---
    try:
        config_data = json.loads(config_json)
        # logger.info(f"Backend Parsed config_data (dict): {config_data}") # Log less verbosely
        config = EvolverConfig.model_validate(config_data)
        # logger.info(f"Backend Validated Config (Pydantic model dump): {config.model_dump()}") # Log less verbosely
        logger.info("Configuration JSON validated successfully.")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON config received.", exc_info=True) # Include traceback on error
        raise HTTPException(status_code=400, detail="Invalid JSON configuration string.")
    except Exception as e:
        logger.error(f"Invalid config data structure: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")

    # --- Evaluation Script Validation (Unchanged) ---
    if not use_standard_eval and task_evaluation is None:
        logger.error("Custom eval requested but no file provided.")
        raise HTTPException(status_code=400, detail="Custom evaluation script required if not using standard.")
    if use_standard_eval and task_evaluation is not None:
        logger.warning("Standard eval requested but custom file also provided. Custom file will be ignored.")
        task_evaluation = None

    # --- Securely save uploaded files (Unchanged logic) ---
    saved_files = {}
    model_def_path: str | None = None
    eval_path: str | None = None
    weights_path: str | None = None
    # Context manager might be cleaner if many files
    try:
        # Save Model Definition
        if model_definition and model_definition.filename:
            model_def_path, _ = generate_secure_path(model_definition.filename, "model_defs")
            logger.info(f"Saving model definition to: {model_def_path}")
            try:
                await model_definition.seek(0)
                contents = await model_definition.read()
                if not contents: raise ValueError("Model definition file is empty.")
                with open(model_def_path, "wb") as f: f.write(contents)
                saved_files['model_def'] = model_def_path
            finally: await model_definition.close()
        else: raise HTTPException(status_code=400, detail="Model definition file or filename missing.")
        # Save Custom Evaluation Script
        if task_evaluation and not use_standard_eval and task_evaluation.filename:
            eval_path, _ = generate_secure_path(task_evaluation.filename, "eval_scripts")
            logger.info(f"Saving custom eval script to: {eval_path}")
            try:
                await task_evaluation.seek(0)
                contents = await task_evaluation.read()
                if not contents: raise ValueError("Custom evaluation file is empty.")
                with open(eval_path, "wb") as f: f.write(contents)
                saved_files['eval'] = eval_path
            finally: await task_evaluation.close()
        elif task_evaluation and not use_standard_eval and not task_evaluation.filename:
            logger.warning("Custom eval file object provided but filename missing.")
            # Decide: Raise error or just skip? Raising is safer.
            raise HTTPException(status_code=400, detail="Custom evaluation filename missing.")
        # Save Initial Weights
        if initial_weights and initial_weights.filename:
            weights_path, _ = generate_secure_path(initial_weights.filename, "weights")
            logger.info(f"Saving initial weights to: {weights_path}")
            try:
                await initial_weights.seek(0)
                contents = await initial_weights.read()
                if not contents: raise ValueError("Initial weights file is empty.")
                with open(weights_path, "wb") as f: f.write(contents)
                saved_files['weights'] = weights_path
            finally: await initial_weights.close()
        elif initial_weights and not initial_weights.filename:
             logger.warning("Initial weights file provided but filename missing. Skipping.")

    except HTTPException as http_exc:
        for file_path in saved_files.values():
            if file_path and os.path.exists(file_path): 
                try: 
                    os.remove(file_path) 
                except OSError: 
                    pass
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error during file processing: {e}", exc_info=True)
        for file_path in saved_files.values():
             if file_path and os.path.exists(file_path): 
                try: 
                    os.remove(file_path) 
                except OSError: 
                    pass
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files.")
    # --- End file saving ---

    # --- Launch Celery task (Unchanged) ---
    logger.info("Dispatching evolution task to Celery...")
    try:
        if not model_def_path: # Final check
            raise RuntimeError("Model definition path missing before dispatch.")

        config_to_send = config.model_dump() # Get dict from Pydantic model
        # logger.debug(f"Backend Sending config to Celery task: {config_to_send}") # Reduce verbosity

        task = run_evolution_task.delay(
            model_definition_path=model_def_path,
            task_evaluation_path=eval_path,
            use_standard_eval=use_standard_eval,
            initial_weights_path=weights_path,
            config=config_to_send # Send the validated dict
        )
        logger.info(f"Task dispatched with ID: {task.id}")
        return TaskResponse(task_id=task.id, status="PENDING")
    except Exception as dispatch_err:
         logger.error(f"Celery task dispatch failed: {dispatch_err}", exc_info=True)
         # Cleanup saved files if dispatch fails
         for file_path in saved_files.values():
             if file_path and os.path.exists(file_path): 
                try: 
                    os.remove(file_path) 
                except OSError: 
                    pass
         raise HTTPException(status_code=500, detail=f"Failed to start evolution task: {dispatch_err}")


# --- get_evolution_status (Unchanged) ---
@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_evolution_status(task_id: str):
    """ Get the status, progress, and metrics of an evolution task. """
    logger.debug(f"Fetching status for task ID: {task_id}")
    task_result = AsyncResult(task_id, app=celery_app)
    status = task_result.status
    info_data, progress, message, error_message = {}, None, None, None
    try:
        if status == 'PENDING': message = "Task is waiting to start."
        elif status == 'STARTED': message = "Task has started."; info_data = task_result.info if isinstance(task_result.info, dict) else {}
        elif status == 'PROGRESS': message = "Task in progress."; info_data = task_result.info if isinstance(task_result.info, dict) else {}
        elif status == 'SUCCESS':
            message = "Task completed successfully."; progress = 1.0
            info_data = task_result.result if isinstance(task_result.result, dict) else {}
            message = info_data.get("message", message)
        elif status == 'FAILURE':
            # Attempt to get details from result first, then info
            if isinstance(task_result.result, dict): info_data = task_result.result
            elif isinstance(task_result.info, dict): info_data = task_result.info
            else: info_data = {}
            error_message = str(info_data.get("error", task_result.info or "Unknown error"))
            message = info_data.get("message", f"Task failed: {error_message}")
            logger.error(f"Task {task_id} failed. Error: {error_message}")
            progress = info_data.get("progress") # Get progress even on failure if available
        elif status == 'REVOKED':
            message = "Task was revoked (terminated)."
            # Try to get last known progress from info if available
            if isinstance(task_result.info, dict): progress = task_result.info.get("progress")
        else: message = f"Task status: {status}"; info_data = task_result.info if isinstance(task_result.info, dict) else {}

        # Update progress/message from info_data if not already set
        if progress is None and isinstance(info_data, dict): progress = info_data.get("progress")
        if message is None and isinstance(info_data, dict): message = info_data.get("message", f"Task status: {status}")

        # Extract histories safely from info_data
        fitness_history = info_data.get("fitness_history") if isinstance(info_data, dict) else None
        avg_fitness_history = info_data.get("avg_fitness_history") if isinstance(info_data, dict) else None
        diversity_history = info_data.get("diversity_history") if isinstance(info_data, dict) else None
        best_hyperparams = info_data.get("best_hyperparameters") if isinstance(info_data, dict) else None

    except Exception as e:
        logger.error(f"Error retrieving status details for task {task_id}: {e}", exc_info=True)
        message = f"Error fetching task details: {e}"; status = "ERROR_FETCHING_STATUS"
        fitness_history, avg_fitness_history, diversity_history, best_hyperparams = None, None, None, None

    response_info, response_result = None, None
    # Combine common info for different states
    common_info = {
        "fitness_history": fitness_history,
        "avg_fitness_history": avg_fitness_history,
        "diversity_history": diversity_history,
        "best_hyperparameters": best_hyperparams # Include potentially partial best hyperparams
    }

    if status in ['STARTED', 'PROGRESS', 'REVOKED']: # Include REVOKED here for last known state
        response_info = {"progress": progress, "message": message, **common_info}
    elif status in ['SUCCESS', 'FAILURE']:
        response_result = {
            "final_model_path": info_data.get("final_model_path"),
            "best_fitness": info_data.get("best_fitness"),
            "message": message,
            "error": error_message,
             **common_info
        }

    return TaskStatus(
        task_id=task_id,
        status=status,
        progress=progress,
        info=response_info, # Contains progress/message/histories for running tasks
        result=response_result, # Contains final results/error and histories for completed tasks
        message=message # Overall message summary
    )


# --- NEW: Redis-based Cooperative Halt Endpoint ---
@router.post("/tasks/{task_id}/terminate", response_model=TerminateResponse)
async def terminate_evolution_task_endpoint(task_id: str):
    """
    Requests graceful termination of a running evolution Celery task by setting a Redis halt flag.
    """
    logger.info(f"Received termination request for task ID: {task_id}")

    # Get the RedisDsn object from settings
    redis_url_setting = settings.REDIS_URL
    # --- FIX: Convert RedisDsn object to string ---
    redis_url_str = str(redis_url_setting)
    # --- End FIX ---

    redis = None # Initialize redis connection variable
    try:
        # Use the string URL to connect
        logger.debug(f"Connecting to Redis at: {redis_url_str}")
        redis = await aioredis.from_url(redis_url_str)
    except Exception as e:
        logger.error(f"Failed to connect to Redis at {redis_url_str}: {e}", exc_info=True)
        # Raise HTTPException if Redis connection fails, as halt cannot be set
        raise HTTPException(status_code=500, detail="Internal server error: Redis connection failed.")

    halt_key = f"task:halt:{task_id}"
    try:
        # Set the halt flag with an expiration (e.g., 1 hour)
        # Use await for aioredis v2+
        await redis.set(halt_key, "1", ex=3600)
        logger.info(f"Set halt flag in Redis for task {task_id} (key: {halt_key})")
        # Close the Redis connection pool when done
        await redis.close()
    except Exception as e:
        logger.error(f"Failed to set halt flag in Redis for task {task_id}: {e}", exc_info=True)
        if redis: await redis.close() # Ensure connection is closed on error too
        raise HTTPException(status_code=500, detail="Internal server error: Failed to set halt flag.")

    return TerminateResponse(
        message=f"Termination requested for task {task_id}. Task will halt shortly if it checks the flag.",
        task_id=task_id
    )



# --- download_evolution_result (Unchanged) ---
@router.get("/results/{task_id}/download")
async def download_evolution_result(task_id: str):
    """ Downloads the final evolved model (.pth) for a completed task. """
    logger.info(f"Received download request for task: {task_id}")
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        if task_result is None: raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found.")
        if task_result.status != 'SUCCESS': raise HTTPException(status_code=400, detail=f"Task {task_id} not completed successfully (Status: {task_result.status}).")
        result_data = task_result.result
        if not isinstance(result_data, dict) or 'final_model_path' not in result_data or not result_data['final_model_path']:
            logger.error(f"Download failed: Task {task_id} succeeded but missing 'final_model_path'. Result: {result_data}")
            raise HTTPException(status_code=500, detail="Result file path missing.")
        relative_file_path = result_data['final_model_path']
        # Use os.path.basename to prevent directory traversal via relative_file_path
        secure_basename = os.path.basename(relative_file_path)
        abs_file_path = os.path.abspath(os.path.join(settings.RESULT_DIR, secure_basename))
        abs_result_dir = os.path.abspath(settings.RESULT_DIR)
        # Ensure the final path is still within the RESULT_DIR
        if not abs_file_path.startswith(abs_result_dir + os.sep):
             logger.error(f"Security Violation: Invalid path construction. Path: {abs_file_path}")
             raise HTTPException(status_code=403, detail="Access denied.")
        if not os.path.exists(abs_file_path) or not os.path.isfile(abs_file_path):
            logger.error(f"Download failed: File not found at: {abs_file_path}")
            raise HTTPException(status_code=404, detail="Result file not found.")
        filename = os.path.basename(abs_file_path) # Use final safe basename
        logger.info(f"Sending file: {abs_file_path} as download: {filename}")
        return FileResponse(path=abs_file_path, filename=filename, media_type='application/octet-stream')
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Error processing download for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not process download: {e}")
