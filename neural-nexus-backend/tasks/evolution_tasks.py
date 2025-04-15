# tasks/evolution_tasks.py
from app.core.celery_app import celery_app
from celery import current_task, Task
import time
import random
import os
import numpy as np
import torch
import logging

# Use config settings imported via Celery context or directly
from app.core.config import settings
from app.utils.evolution_helpers import (
    load_pytorch_model, flatten_weights, load_task_eval_function,
    evaluate_population_step, select_parents, crossover, mutate, load_weights_from_flat
)

logger = logging.getLogger(__name__)

# Standard evaluation script path from config
STANDARD_EVAL_SCRIPT_PATH = settings.STANDARD_EVAL_SCRIPT_PATH

# Result directory from config
RESULT_DIR = settings.RESULT_DIR
os.makedirs(RESULT_DIR, exist_ok=True) # Ensure it exists

# --- Task Base Class with Retry ---
class EvolutionTaskWithRetry(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 2} # Fewer retries for long tasks
    retry_backoff = True
    retry_backoff_max = 7000
    retry_jitter = False
    acks_late = True # Acknowledge after task completes/fails
    reject_on_worker_lost = True

@celery_app.task(bind=True, base=EvolutionTaskWithRetry)
def run_evolution_task(self: Task, model_definition_path: str, task_evaluation_path: str | None, use_standard_eval: bool, initial_weights_path: str | None, config: dict):
    """ Celery task for evolution. Uses config paths and detailed logging. """
    task_id = self.request.id
    logger.info(f"[Task {task_id}] Starting evolution...")

    # --- Determine Evaluation Script ---
    if use_standard_eval:
        actual_eval_script_path = STANDARD_EVAL_SCRIPT_PATH
        if not os.path.exists(actual_eval_script_path):
             error_msg = f"Standard evaluation script not found at configured path: {actual_eval_script_path}"
             logger.error(f"[Task {task_id}] {error_msg}")
             raise FileNotFoundError(error_msg) # Task will fail
        logger.info(f"[Task {task_id}] Using standard evaluation script: {actual_eval_script_path}")
    elif task_evaluation_path and os.path.exists(task_evaluation_path):
        actual_eval_script_path = task_evaluation_path
        logger.info(f"[Task {task_id}] Using custom evaluation script: {actual_eval_script_path}")
    else:
        error_msg = f"Custom evaluation script path invalid or not provided: {task_evaluation_path}"
        logger.error(f"[Task {task_id}] {error_msg}")
        raise FileNotFoundError(error_msg) # Task will fail


    # --- Setup ---
    try:
        generations = int(config.get('generations', 10))
        population_size = int(config.get('population_size', 20))
        mutation_rate = float(config.get('mutation_rate', 0.1))
        mutation_strength = float(config.get('mutation_strength', 0.05))
        init_mutation_rate = float(config.get('init_mutation_rate', 0.2))
        init_mutation_strength = float(config.get('init_mutation_strength', 0.1))
        model_class = config.get('model_class', 'MyCNN') # Ensure this is passed correctly
        model_args = config.get('model_args', [])
        model_kwargs = config.get('model_kwargs', {})

        if not model_class:
            raise ValueError("Configuration must include 'model_class' matching the class name in the model definition file.")

        final_model_filename = f"evolved_{task_id}.pth" # Use task ID in filename
        final_model_path = os.path.join(RESULT_DIR, final_model_filename)

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Task {task_id}] Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"[Task {task_id}] CUDA Device: {torch.cuda.get_device_name(0)}")
            free_mem, total_mem = torch.cuda.mem_get_info()
            logger.info(f"[Task {task_id}] CUDA Memory: {free_mem/1e9:.2f} GB free / {total_mem/1e9:.2f} GB total")

        logger.info(f"[Task {task_id}] Config: Gens={generations}, Pop={population_size}, MutRate={mutation_rate}, MutStr={mutation_strength}")
        self.update_state(state='STARTED', meta={'progress': 0.0, 'message': 'Initialization starting...'})

        # --- Load Task Evaluation Function ---
        task_eval_func = load_task_eval_function(actual_eval_script_path)

        # --- Load Initial Model & Weights ---
        logger.info(f"[Task {task_id}] Loading initial model '{model_class}' from {model_definition_path}")
        initial_model = load_pytorch_model(
            model_definition_path, model_class, initial_weights_path, device, *model_args, **model_kwargs
        )
        initial_weights = flatten_weights(initial_model)
        logger.info(f"[Task {task_id}] Initial model loaded. Weight vector size: {initial_weights.shape[0]}")
        del initial_model
        if device.type == 'cuda': torch.cuda.empty_cache()

        # --- Initialize Population ---
        population = [initial_weights.copy()]
        logger.info(f"[Task {task_id}] Initializing population ({population_size})...")
        for i in range(population_size - 1):
             # Add some logging for mutation params during init
             logger.debug(f"[Task {task_id}] Mutating initial weights for individual {i+2} (Rate={init_mutation_rate}, Strength={init_mutation_strength})")
             mutated_initial = mutate(initial_weights, init_mutation_rate, init_mutation_strength)
             population.append(mutated_initial)

        self.update_state(state='PROGRESS', meta={'progress': 0.01, 'message': 'Initialization complete. Starting generations...'})

    except Exception as init_err:
         error_msg = f"Initialization failed: {init_err}"
         logger.error(f"[Task {task_id}] {error_msg}", exc_info=True)
         self.update_state(state='FAILURE', meta={'message': error_msg})
         raise # Re-raise to mark task as failed

    # --- Evolution Loop ---
    best_fitness_overall = -float('inf')
    best_weights_overall = initial_weights.copy()
    fitness_history_overall = []

    try:
        for gen in range(generations):
            gen_num = gen + 1
            logger.info(f"[Task {task_id}] --- Generation {gen_num}/{generations} ---")
            gen_start_time = time.time()

            # 1. Evaluate Population
            try:
                fitness_scores = evaluate_population_step(
                    population, model_definition_path, model_class, task_eval_func, device, model_args, model_kwargs
                )
            except Exception as eval_err:
                 # Log error but try to continue if possible by assigning low fitness
                 logger.error(f"[Task {task_id}] Error during evaluation in Gen {gen_num}: {eval_err}", exc_info=False)
                 fitness_scores = [-float('inf')] * len(population)

            # 2. Process Results
            valid_scores = [f for f in fitness_scores if f > -float('inf')]
            if not valid_scores:
                error_msg = "All individuals failed evaluation in this generation. Stopping evolution."
                logger.error(f"[Task {task_id}] {error_msg}")
                raise RuntimeError(error_msg) # Stop the task

            max_fitness = np.max(valid_scores)
            avg_fitness = np.mean(valid_scores)
            best_idx_current = np.argmax(fitness_scores) # Index in current population

            fitness_history_overall.append(float(max_fitness)) # Store as float

            logger.info(f"[Task {task_id}] Gen {gen_num} Stats: MaxFit={max_fitness:.4f}, AvgFit={avg_fitness:.4f}")

            # Update overall best if current generation's best is better
            if fitness_scores[best_idx_current] > best_fitness_overall:
                best_fitness_overall = fitness_scores[best_idx_current]
                best_weights_overall = population[best_idx_current].copy()
                logger.info(f"[Task {task_id}] *** New best overall fitness: {best_fitness_overall:.4f} ***")

            # --- Update Task State ---
            current_progress = gen_num / generations
            progress_message = f'Gen {gen_num}/{generations} | Max Fitness: {max_fitness:.4f}'
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': current_progress,
                    'message': progress_message,
                    'current_max_fitness': float(max_fitness),
                    'current_avg_fitness': float(avg_fitness),
                    'fitness_history': fitness_history_overall # Cumulative history
                }
            )

            # 3. Selection
            num_parents = max(2, population_size // 2)
            parents = select_parents(population, fitness_scores, num_parents)
            if not parents:
                # Fallback if selection fails: Re-initialize from best overall
                logger.warning(f"[Task {task_id}] Parent selection yielded no parents in Gen {gen_num}. Re-populating from best overall.")
                population = [best_weights_overall.copy()] + [mutate(best_weights_overall, mutation_rate, mutation_strength) for _ in range(population_size - 1)]
                continue # Skip crossover/mutation for this gen

            # 4. Reproduction
            next_population = [population[best_idx_current].copy()] # Elitism: Keep best from current gen
            while len(next_population) < population_size:
                try:
                    p1, p2 = random.sample(parents, 2)
                    child = crossover(p1, p2)
                    mutated_child = mutate(child, mutation_rate, mutation_strength)
                    next_population.append(mutated_child)
                except ValueError as repr_err: # Handle errors from crossover/mutation if they occur
                    logger.warning(f"[Task {task_id}] Reproduction error in Gen {gen_num}: {repr_err}. Substituting with mutated parent.")
                    if parents and len(next_population) < population_size:
                        next_population.append(mutate(random.choice(parents), mutation_rate, mutation_strength))
                except IndexError: # Handle sampling error if parents list is too small unexpectedly
                     logger.warning(f"[Task {task_id}] Could not sample parents in Gen {gen_num}. Substituting.")
                     if parents and len(next_population) < population_size:
                        next_population.append(mutate(random.choice(parents), mutation_rate, mutation_strength))

            population = next_population
            gen_time = time.time() - gen_start_time
            logger.info(f"[Task {task_id}] Gen {gen_num} finished in {gen_time:.2f}s")

        # --- Evolution Finished ---
        logger.info(f"[Task {task_id}] Evolution finished after {generations} generations.")

        # --- Save Final Best Model ---
        logger.info(f"[Task {task_id}] Saving best model (Fitness: {best_fitness_overall:.4f}) to {final_model_path}...")
        final_best_model = load_pytorch_model(
             model_definition_path, model_class, None, device, *model_args, **model_kwargs
        )
        load_weights_from_flat(final_best_model, best_weights_overall)
        torch.save(final_best_model.state_dict(), final_model_path)
        logger.info(f"[Task {task_id}] Best model saved successfully.")

        # Clean up uploaded files (optional - configure retention)
        logger.info(f"[Task {task_id}] Cleaning up temporary uploaded files...")
        files_to_clean = [model_definition_path, task_evaluation_path, initial_weights_path]
        for f_path in files_to_clean:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    logger.debug(f"[Task {task_id}] Removed temp file: {f_path}")
                except OSError as rm_err:
                    logger.warning(f"[Task {task_id}] Could not remove temp file {f_path}: {rm_err}")


        # --- Return final result data ---
        success_message = f'Evolution completed successfully. Best fitness: {best_fitness_overall:.4f}.'
        final_result = {
            'message': success_message,
            'final_model_path': final_model_path, # Path inside container
            'best_fitness': float(best_fitness_overall),
            'fitness_history': fitness_history_overall
        }
        logger.info(f"[Task {task_id}] Task successful. Result: {final_result}")
        return final_result

    except Exception as e:
        error_message = f'Evolution task failed during run: {str(e)}'
        logger.error(f"[Task {task_id}] {error_message}", exc_info=True)
        # Update state with failure message
        self.update_state(state='FAILURE', meta={'message': error_message})
        raise # Re-raise for Celery status and retries
