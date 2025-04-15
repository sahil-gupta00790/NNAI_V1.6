# app/api/endpoints/evolver.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import FileResponse # For downloads
from typing import Annotated, Optional # Ensure Optional is imported if using UploadFile | None
import json
from app.models.common import TaskResponse, TaskStatus
from app.models.evolver import EvolverConfig
from tasks.evolution_tasks import run_evolution_task
from celery.result import AsyncResult
from app.core.celery_app import celery_app
from app.core.config import settings # Use config settings
import os
import uuid # For unique filenames
import logging
from werkzeug.utils import secure_filename # For filename sanitization

logger = logging.getLogger(__name__)

# Ensure upload/result dirs exist (handled by Docker volumes ideally, but good practice)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULT_DIR, exist_ok=True)

router = APIRouter()

# --- Helper for Secure Filename ---
def generate_secure_path(original_filename: str, sub_dir: str = "") -> tuple[str, str]:
    """Generates a secure filename and the full path within the UPLOAD_DIR."""
    if not original_filename:
        raise ValueError("Original filename cannot be empty.")
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
    # --- Parameters WITHOUT defaults FIRST ---
    model_definition: Annotated[UploadFile, File(...)],
    config_json: Annotated[str, Form(...)], # <-- Moved UP, no default

    # --- Parameters WITH defaults LAST ---
    task_evaluation: Annotated[Optional[UploadFile], File()] = None,
    initial_weights: Annotated[Optional[UploadFile], File()] = None,
    use_standard_eval: Annotated[bool, Form(...)] = False,
    # --- Dependency Injection (typically comes last or after defaults) ---
    # db: AsyncSession = Depends(get_db) # Add if you have DB dependency
):
    """
    Start the CNN/RNN evolution task. Securely saves files and uses config paths.
    """
    logger.info("Received request to start evolution task.")
    # --- Config Validation ---
    try:
        config_data = json.loads(config_json)
        if 'model_class' not in config_data:
             logger.warning("Request 'config_json' lacks 'model_class'. Using default 'YourModelClass'. Ensure match.")
             config_data['model_class'] = 'YourModelClass'
        config = EvolverConfig.model_validate(config_data)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON config: {config_json}")
        raise HTTPException(status_code=400, detail="Invalid JSON configuration string.")
    except Exception as e:
        logger.error(f"Invalid config data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")

    # --- Evaluation Script Validation ---
    if not use_standard_eval and task_evaluation is None:
        logger.error("Custom eval requested but no file provided.")
        raise HTTPException(status_code=400, detail="Custom evaluation script required if not using standard.")
    if use_standard_eval and task_evaluation is not None:
        logger.error("Standard eval requested but custom file also provided.")
        raise HTTPException(status_code=400, detail="Cannot provide custom script when using standard evaluation.")

    # --- Securely save uploaded files ---
    saved_files = {}
    try:
        # Save Model Definition
        if model_definition.filename:
             model_def_path, secure_name = generate_secure_path(model_definition.filename, "model_defs")
             logger.info(f"Saving model definition to: {model_def_path}")
             with open(model_def_path, "wb") as f:
                contents = await model_definition.read()
                if not contents: raise HTTPException(status_code=400, detail="Model definition file is empty.")
                f.write(contents)
             saved_files['model_def'] = model_def_path
        else:
             raise HTTPException(status_code=400, detail="Model definition filename missing.")

        # Save Custom Evaluation Script
        eval_path: str | None = None
        if task_evaluation and not use_standard_eval:
            if task_evaluation.filename:
                eval_path, secure_name = generate_secure_path(task_evaluation.filename, "eval_scripts")
                logger.info(f"Saving custom eval script to: {eval_path}")
                with open(eval_path, "wb") as f:
                    contents = await task_evaluation.read()
                    if not contents: raise HTTPException(status_code=400, detail="Custom evaluation file is empty.")
                    f.write(contents)
                saved_files['eval'] = eval_path
            else:
                 raise HTTPException(status_code=400, detail="Custom evaluation filename missing.")

        # Save Initial Weights
        weights_path: str | None = None
        if initial_weights:
            if initial_weights.filename:
                weights_path, secure_name = generate_secure_path(initial_weights.filename, "weights")
                logger.info(f"Saving initial weights to: {weights_path}")
                with open(weights_path, "wb") as f:
                    contents = await initial_weights.read()
                    if not contents: raise HTTPException(status_code=400, detail="Initial weights file is empty.")
                    f.write(contents)
                saved_files['weights'] = weights_path
            else:
                 logger.warning("Initial weights file provided but filename missing. Skipping.")

    except HTTPException as http_exc:
        for file_path in saved_files.values():
            if os.path.exists(file_path): 
                try: 
                    os.remove(file_path); 
                except OSError: pass
        raise http_exc
    except Exception as e:
        logger.error(f"Error saving uploaded files: {e}", exc_info=True)
        for file_path in saved_files.values():
            if os.path.exists(file_path): 
                try: 
                    os.remove(file_path); 
                except OSError: 
                    pass
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {e}")
    # --- End file saving ---

    # --- Launch Celery task ---
    logger.info("Dispatching evolution task to Celery...")
    try:
        task = run_evolution_task.delay(
            model_definition_path=saved_files['model_def'],
            task_evaluation_path=eval_path,
            use_standard_eval=use_standard_eval,
            initial_weights_path=weights_path,
            config=config.model_dump()
        )
        logger.info(f"Task dispatched with ID: {task.id}")
        return TaskResponse(task_id=task.id, status="PENDING")
    except Exception as dispatch_err:
         logger.error(f"Celery task dispatch failed: {dispatch_err}", exc_info=True)
         for file_path in saved_files.values():
             if os.path.exists(file_path): 
                try: 
                    os.remove(file_path); 
                except OSError: 
                    pass
         raise HTTPException(status_code=500, detail=f"Failed to start evolution task: {dispatch_err}")

# --- get_evolution_status function (remains the same) ---
@router.get("/status/{task_id}", response_model=TaskStatus)
async def get_evolution_status(task_id: str):
    """ Get the status and progress of an evolution task. """
    logger.debug(f"Fetching status for task ID: {task_id}")
    task_result = AsyncResult(task_id, app=celery_app)
    status = task_result.status
    result_data = None
    progress = None
    message = None
    fitness_history = None

    try:
        if status == 'PENDING': message = "Task is waiting to start."
        elif status == 'STARTED':
            message = "Task has started."
            if isinstance(task_result.info, dict):
                progress = task_result.info.get("progress"); message = task_result.info.get("message", message); fitness_history = task_result.info.get("fitness_history")
        elif status == 'PROGRESS':
            message = "Task in progress."
            if isinstance(task_result.info, dict):
                progress = task_result.info.get("progress"); message = task_result.info.get("message", message); fitness_history = task_result.info.get("fitness_history")
        elif status == 'SUCCESS':
            message = "Task completed successfully."; progress = 1.0; result_data = task_result.get()
            if isinstance(result_data, dict): fitness_history = result_data.get("fitness_history")
        elif status == 'FAILURE':
            error_info = str(task_result.info); message = f"Task failed: {error_info}"; logger.error(f"Task {task_id} failed. Info: {task_result.info}"); result_data = {"error": error_info}
        elif status == 'REVOKED': message = "Task was revoked."
        else:
            message = f"Task status: {status}"
            if isinstance(task_result.info, dict):
                progress = task_result.info.get("progress"); fitness_history = task_result.info.get("fitness_history")
    except Exception as e:
        logger.error(f"Error retrieving status for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching task status: {e}")

    return TaskStatus(
        task_id=task_id, status=status, progress=progress,
        result={"data": result_data, "fitness_history": fitness_history} if result_data or fitness_history else None,
        message=message
    )

# --- download_evolution_result function (remains the same) ---
@router.get("/results/{task_id}/download")
async def download_evolution_result(task_id: str):
    """ Downloads the final evolved model (.pth) for a completed task. """
    logger.info(f"Received download request for task: {task_id}")
    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result:
        logger.warning(f"Download failed: Task ID {task_id} not found."); raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found.")
    if task_result.status != 'SUCCESS':
        logger.warning(f"Download failed: Task {task_id} not successful (Status: {task_result.status})."); raise HTTPException(status_code=400, detail=f"Task {task_id} has not completed successfully (Status: {task_result.status}).")

    try:
        result_data = task_result.get()
        if not isinstance(result_data, dict) or 'final_model_path' not in result_data:
            logger.error(f"Download failed: Task {task_id} succeeded but result missing 'final_model_path'. Result: {result_data}"); raise HTTPException(status_code=500, detail="Task completed, but result file path is missing.")

        file_path = result_data['final_model_path']
        abs_file_path = os.path.abspath(file_path); abs_result_dir = os.path.abspath(settings.RESULT_DIR)

        if not abs_file_path.startswith(abs_result_dir):
             logger.error(f"Security Violation: Attempt access outside RESULT_DIR. Path: {file_path}"); raise HTTPException(status_code=403, detail="Access denied: Invalid file path.")
        if not os.path.exists(abs_file_path) or not os.path.isfile(abs_file_path):
            logger.error(f"Download failed: File not found at path: {abs_file_path}"); raise HTTPException(status_code=404, detail="Result file not found on server.")

        filename = os.path.basename(abs_file_path)
        logger.info(f"Sending file: {abs_file_path} as {filename}")
        return FileResponse(path=abs_file_path, filename=filename, media_type='application/octet-stream')
    except Exception as e:
        logger.error(f"Error processing download for task {task_id}: {e}", exc_info=True); raise HTTPException(status_code=500, detail=f"Could not process file download: {e}")

