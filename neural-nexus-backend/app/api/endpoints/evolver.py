# app/api/endpoints/evolver.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import FileResponse # For downloads
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

logger = logging.getLogger(__name__)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULT_DIR, exist_ok=True)

router = APIRouter()

# --- Helper for Secure Filename (Unchanged) ---
def generate_secure_path(original_filename: str, sub_dir: str = "") -> tuple[str, str]:
    # ...(Your existing secure path logic remains unchanged)...
    if not original_filename: raise ValueError("Original filename cannot be empty.")
    base_name = secure_filename(original_filename)
    if not base_name: base_name = "unnamed_file"
    unique_prefix = str(uuid.uuid4())[:8]
    secure_name = f"{unique_prefix}_{base_name}"
    target_dir = os.path.join(settings.UPLOAD_DIR, sub_dir)
    os.makedirs(target_dir, exist_ok=True)
    full_path = os.path.join(target_dir, secure_name)
    while os.path.exists(full_path): # Check for collisions
        unique_prefix = str(uuid.uuid4())[:8]
        secure_name = f"{unique_prefix}_{base_name}"
        full_path = os.path.join(target_dir, secure_name)
    logger.debug(f"Generated secure path: {full_path} for original: {original_filename}")
    return full_path, secure_name


@router.post("/start", response_model=TaskResponse)
async def start_evolution(
    model_definition: Annotated[UploadFile, File(...)],
    config_json: Annotated[str, Form(...)],
    task_evaluation: Annotated[Optional[UploadFile], File()] = None,
    initial_weights: Annotated[Optional[UploadFile], File()] = None,
    use_standard_eval: Annotated[bool, Form(...)] = False,
):
    logger.info("Received request to start evolution task.")

    logger.info(f"Backend Received Raw config_json: {config_json}")
    
    # --- Config Validation (Unchanged) ---
    try:
        config_data = json.loads(config_json)
        logger.info(f"Backend Parsed config_data (dict): {config_data}")
        config = EvolverConfig.model_validate(config_data)
        logger.info(f"Backend Validated Config (Pydantic model dump): {config.model_dump()}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON config: {config_json}")
        raise HTTPException(status_code=400, detail="Invalid JSON configuration string.")
    except Exception as e:
        logger.error(f"Invalid config data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")

    # --- Evaluation Script Validation (Unchanged) ---
    if not use_standard_eval and task_evaluation is None:
        logger.error("Custom eval requested but no file provided.")
        raise HTTPException(status_code=400, detail="Custom evaluation script required if not using standard.")
    if use_standard_eval and task_evaluation is not None:
        logger.warning("Standard eval requested but custom file also provided. Custom file will be ignored.")
        task_evaluation = None

    # --- Securely save uploaded files (MODIFIED LOGIC) ---
    saved_files = {}
    model_def_path: str | None = None
    eval_path: str | None = None
    weights_path: str | None = None

    try:
        # Save Model Definition
        if model_definition and model_definition.filename:
            model_def_path, _ = generate_secure_path(model_definition.filename, "model_defs")
            logger.info(f"Saving model definition to: {model_def_path}")
            try:
                await model_definition.seek(0) # Await seek separately
                contents = await model_definition.read() # Await read
                if not contents: raise ValueError("Model definition file is empty.")
                # Use standard synchronous file writing
                with open(model_def_path, "wb") as f:
                    f.write(contents)
                saved_files['model_def'] = model_def_path
            except Exception as file_err:
                logger.error(f"Error writing model definition file: {file_err}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to save model definition file.")
            finally:
                 await model_definition.close() # Close the upload file
        else:
            raise HTTPException(status_code=400, detail="Model definition file or filename missing.")

        # Save Custom Evaluation Script
        if task_evaluation and not use_standard_eval:
            if task_evaluation.filename:
                eval_path, _ = generate_secure_path(task_evaluation.filename, "eval_scripts")
                logger.info(f"Saving custom eval script to: {eval_path}")
                try:
                    await task_evaluation.seek(0) # Await seek
                    contents = await task_evaluation.read() # Await read
                    if not contents: raise ValueError("Custom evaluation file is empty.")
                    with open(eval_path, "wb") as f:
                        f.write(contents)
                    saved_files['eval'] = eval_path
                except Exception as file_err:
                    logger.error(f"Error writing custom eval file: {file_err}", exc_info=True)
                    raise HTTPException(status_code=500, detail="Failed to save custom evaluation file.")
                finally:
                    await task_evaluation.close() # Close the upload file
            else:
                 logger.warning("Custom eval selected but file object or filename missing.")
                 raise HTTPException(status_code=400, detail="Custom evaluation filename missing.")

        # Save Initial Weights
        if initial_weights:
            if initial_weights.filename:
                weights_path, _ = generate_secure_path(initial_weights.filename, "weights")
                logger.info(f"Saving initial weights to: {weights_path}")
                try:
                    await initial_weights.seek(0) # Await seek
                    contents = await initial_weights.read() # Await read
                    if not contents: raise ValueError("Initial weights file is empty.")
                    with open(weights_path, "wb") as f:
                        f.write(contents)
                    saved_files['weights'] = weights_path
                except Exception as file_err:
                    logger.error(f"Error writing initial weights file: {file_err}", exc_info=True)
                    raise HTTPException(status_code=500, detail="Failed to save initial weights file.")
                finally:
                    await initial_weights.close() # Close the upload file
            else:
                 logger.warning("Initial weights file provided but filename missing. Skipping.")

    except HTTPException as http_exc: # Catch validation errors first
        # Attempt cleanup for already saved files if error occurs later
        for file_path in saved_files.values():
            if os.path.exists(file_path): 
                try: 
                    os.remove(file_path); 
                except OSError: 
                    pass
        raise http_exc # Re-raise the specific validation error
    except Exception as e: # Catch generic errors during file handling
        logger.error(f"Unexpected error during file processing: {e}", exc_info=True)
        # Attempt cleanup
        for file_path in saved_files.values():
            if os.path.exists(file_path): 
                try: 
                    os.remove(file_path); 
                except OSError: 
                    pass
        # Use a generic 500 error message, hiding the specific internal error like '__aenter__'
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files.")
    # --- End file saving ---

    # --- Launch Celery task (Unchanged) ---
    logger.info("Dispatching evolution task to Celery...")
    try:
        if not model_def_path: # Check again before dispatch
            raise RuntimeError("Model definition path was not successfully set before dispatch.")

        config_to_send = config.model_dump()
        logger.info(f"Backend Sending config to Celery task: {config_to_send}")

        task = run_evolution_task.delay(
            model_definition_path=model_def_path,
            task_evaluation_path=eval_path,
            use_standard_eval=use_standard_eval,
            initial_weights_path=weights_path,
            config=config.model_dump()
        )
        logger.info(f"Task dispatched with ID: {task.id}")
        return TaskResponse(task_id=task.id, status="PENDING")
    except Exception as dispatch_err:
         logger.error(f"Celery task dispatch failed: {dispatch_err}", exc_info=True)
         # Attempt cleanup if dispatch fails AFTER saving files
         for file_path in saved_files.values():
             if os.path.exists(file_path): 
                try: 
                    os.remove(file_path); 
                except OSError: 
                    pass
         raise HTTPException(status_code=500, detail=f"Failed to start evolution task: {dispatch_err}")


# --- get_evolution_status (Unchanged from previous version) ---
@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_evolution_status(task_id: str):
    """ Get the status, progress, and metrics of an evolution task. """
    # ...(Keep the implementation from the previous correct version)...
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
            info_data = task_result.result if isinstance(task_result.result, dict) else {}
            error_message = str(info_data.get("error", task_result.info or "Unknown error"))
            message = info_data.get("message", f"Task failed: {error_message}")
            logger.error(f"Task {task_id} failed. Error: {error_message}")
            if isinstance(task_result.info, dict): progress = task_result.info.get("progress")
        elif status == 'REVOKED': message = "Task was revoked."
        else: message = f"Task status: {status}"; info_data = task_result.info if isinstance(task_result.info, dict) else {}
        if progress is None: progress = info_data.get("progress")
        if message is None: message = info_data.get("message", f"Task status: {status}")
        fitness_history = info_data.get("fitness_history")
        avg_fitness_history = info_data.get("avg_fitness_history")
        diversity_history = info_data.get("diversity_history")
    except Exception as e:
        logger.error(f"Error retrieving status details for task {task_id}: {e}", exc_info=True)
        message = f"Error fetching task details: {e}"; status = "ERROR_FETCHING_STATUS"
    response_info, response_result = None, None
    if status in ['STARTED', 'PROGRESS']:
        response_info = {"progress": progress, "message": message, "fitness_history": fitness_history, "avg_fitness_history": avg_fitness_history, "diversity_history": diversity_history}
    elif status in ['SUCCESS', 'FAILURE']:
        response_result = {"final_model_path": info_data.get("final_model_path"), "best_fitness": info_data.get("best_fitness"), "message": message, "error": error_message, "fitness_history": fitness_history, "avg_fitness_history": avg_fitness_history, "diversity_history": diversity_history}
    return TaskStatus(task_id=task_id, status=status, progress=progress, info=response_info, result=response_result, message=message)


# --- download_evolution_result (Unchanged from previous version) ---
@router.get("/results/{task_id}/download")
async def download_evolution_result(task_id: str):
    """ Downloads the final evolved model (.pth) for a completed task. """
    # ...(Keep the implementation from the previous correct version)...
    logger.info(f"Received download request for task: {task_id}")
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        if task_result is None: raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found.")
        if task_result.status != 'SUCCESS': raise HTTPException(status_code=400, detail=f"Task {task_id} has not completed successfully (Status: {task_result.status}).")
        result_data = task_result.result
        if not isinstance(result_data, dict) or 'final_model_path' not in result_data or not result_data['final_model_path']:
            logger.error(f"Download failed: Task {task_id} succeeded but result missing 'final_model_path'. Result: {result_data}")
            raise HTTPException(status_code=500, detail="Task completed, but result file path is missing or empty in result data.")
        relative_file_path = result_data['final_model_path']
        abs_file_path = os.path.abspath(os.path.join(settings.RESULT_DIR, os.path.basename(relative_file_path)))
        abs_result_dir = os.path.abspath(settings.RESULT_DIR)
        if not abs_file_path.startswith(abs_result_dir + os.sep):
             logger.error(f"Security Violation: Attempted download access outside RESULT_DIR. Path: {abs_file_path}")
             raise HTTPException(status_code=403, detail="Access denied: Invalid file path.")
        if not os.path.exists(abs_file_path) or not os.path.isfile(abs_file_path):
            logger.error(f"Download failed: File not found at expected path: {abs_file_path}")
            raise HTTPException(status_code=404, detail="Result file not found on server.")
        filename = os.path.basename(abs_file_path)
        logger.info(f"Sending file: {abs_file_path} as download attachment: {filename}")
        return FileResponse(path=abs_file_path, filename=filename, media_type='application/octet-stream')
    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Error processing download for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not process file download: {e}")
