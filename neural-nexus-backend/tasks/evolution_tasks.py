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
# --- Import ALL helper functions ---
from app.utils.evolution_helpers import (
    load_pytorch_model, flatten_weights, load_task_eval_function,
    evaluate_population_step, load_weights_from_flat,
    # Selection
    select_parents_tournament,
    select_parents_roulette,
    # Crossover
    crossover_one_point,
    crossover_uniform,
    crossover_average, # If you want to support this too
    # Mutation
    mutate_gaussian,
    mutate_uniform_random
)

logger = logging.getLogger(__name__)
# Configure logging if not done globally
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')

# Standard evaluation script path from config
STANDARD_EVAL_SCRIPT_PATH = settings.STANDARD_EVAL_SCRIPT_PATH
# Result directory from config
RESULT_DIR = settings.RESULT_DIR
os.makedirs(RESULT_DIR, exist_ok=True) # Ensure it exists

# --- Helper Function for Diversity (Example: Avg Pairwise Distance) ---
# (Keep the calculate_population_diversity function as provided before)
def calculate_population_diversity(population_weights_list: list[np.ndarray]) -> float:
    """Calculates average pairwise Euclidean distance between weight vectors."""
    if not population_weights_list or len(population_weights_list) < 2: return 0.0
    num_individuals = len(population_weights_list)
    flat_weights = [w.flatten() for w in population_weights_list if isinstance(w, np.ndarray)]
    if len(flat_weights) < 2: return 0.0
    first_len = len(flat_weights[0])
    if not all(len(w) == first_len for w in flat_weights):
        logger.warning("Inconsistent weight vector lengths in population for diversity calc.")
        consistent_weights = [w for w in flat_weights if len(w) == first_len]
        if len(consistent_weights) < 2: return 0.0
        flat_weights = consistent_weights
        num_individuals = len(flat_weights)
    total_distance = 0.0
    num_pairs = 0
    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            distance = np.linalg.norm(flat_weights[i] - flat_weights[j])
            total_distance += distance
            num_pairs += 1
    return (total_distance / num_pairs) if num_pairs > 0 else 0.0

# --- Task Base Class with Retry ---
# (Keep the EvolutionTaskWithRetry class as provided before)
class EvolutionTaskWithRetry(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 2}
    retry_backoff = True
    retry_backoff_max = 7000
    retry_jitter = False
    acks_late = True
    reject_on_worker_lost = True

@celery_app.task(bind=True, base=EvolutionTaskWithRetry)
def run_evolution_task(self: Task, model_definition_path: str, task_evaluation_path: str | None, use_standard_eval: bool, initial_weights_path: str | None, config: dict):
    """ Celery task for evolution. Includes configurable operators and metric tracking. """
    try:
         # Try logging immediately, even before accessing self.request.id
         logger.info(f"TASK ENTRY POINT REACHED. Raw config type: {type(config)}")
         # If the above works, then try accessing request id
         task_id = self.request.id
         logger.info(f"[Task {task_id}] Starting evolution...")
         logger.info(f"[Task {task_id}] Received config dictionary: {config}")
    except Exception as entry_err:
         # Log any error happening right at the entry point
         logger.critical(f"CRITICAL ERROR AT TASK ENTRY: {entry_err}", exc_info=True)
         # Re-raise to ensure Celery knows it failed fundamentally
         raise


    # --- Determine Evaluation Script ---
    # (Keep this section as provided before)
    if use_standard_eval:
        actual_eval_script_path = STANDARD_EVAL_SCRIPT_PATH
        # ... (rest of eval script path logic) ...
    elif task_evaluation_path and os.path.exists(task_evaluation_path):
        actual_eval_script_path = task_evaluation_path
        logger.info(f"[Task {task_id}] Using custom evaluation script: {actual_eval_script_path}")
    else:
        error_msg = f"Custom evaluation script path invalid or not provided: {task_evaluation_path}"
        logger.error(f"[Task {task_id}] {error_msg}")
        raise FileNotFoundError(error_msg)

    # --- Setup & Config Extraction ---
    try:
        # Helper function for safe conversion
        def safe_convert(key, target_type, default_value, is_tuple=False, expected_len=None):
            value = config.get(key, default_value)
            if value is None:
                # If None is the default, return it directly if target_type allows None
                # For required types using default, log warning and use default
                if default_value is None and target_type is not tuple: # Allow None for non-tuple optionals
                    return None
                logger.warning(f"Config key '{key}' is None, using default: {default_value}")
                value = default_value # Use the provided default

            try:
                if is_tuple:
                    if not isinstance(value, (list, tuple)):
                         raise TypeError(f"Value for '{key}' must be a list or tuple")
                    if expected_len is not None and len(value) != expected_len:
                        raise ValueError(f"Value for '{key}' must have length {expected_len}")
                    # Convert elements inside tuple if needed (e.g., all floats)
                    # Assuming tuple of floats for range:
                    converted_value = tuple(float(v) for v in value)
                elif target_type is float:
                    converted_value = float(value)
                elif target_type is int:
                    converted_value = int(value)
                elif target_type is str:
                    converted_value = str(value).lower() # Keep lowercasing for strategies
                elif target_type is bool:
                     # Handle potential string 'true'/'false' from Form data if not bool
                     if isinstance(value, str):
                          converted_value = value.lower() in ['true', '1', 'yes']
                     else:
                          converted_value = bool(value)
                else: # Assume list, dict are okay as is if not None
                     converted_value = value

                return converted_value
            except (TypeError, ValueError) as conv_err:
                logger.error(f"Invalid type for config key '{key}'. Expected {target_type.__name__} convertible, got {type(value)} ({value}). Error: {conv_err}. Using default: {default_value}", exc_info=True)
                # If default itself fails conversion, this indicates a code bug
                try:
                    if is_tuple: return tuple(float(v) for v in default_value)
                    else: return target_type(default_value)
                except:
                    logger.critical(f"Default value '{default_value}' for key '{key}' is invalid for type {target_type.__name__}!", exc_info=True)
                    raise ValueError(f"Invalid configuration for '{key}' and invalid default.")


        # --- General Config ---
        generations = safe_convert('generations', int, 10)
        population_size = safe_convert('population_size', int, 20)
        model_class = config.get('model_class') # Check directly below
        if not model_class or not isinstance(model_class, str):
             raise ValueError("'model_class' (string) is required in the configuration JSON.")
        model_args = safe_convert('model_args', list, [])
        model_kwargs = safe_convert('model_kwargs', dict, {})

        # --- GA Operator Config ---
        selection_strategy = safe_convert("selection_strategy", str, "tournament")
        crossover_operator = safe_convert("crossover_operator", str, "one_point")
        mutation_operator = safe_convert("mutation_operator", str, "gaussian")
        elitism_count = safe_convert("elitism_count", int, 1)

        # --- Mutation Params ---
        mutation_rate = safe_convert('mutation_rate', float, 0.1)
        mutation_strength = safe_convert('mutation_strength', float, 0.05)
        tournament_size = safe_convert('tournament_size', int, 3)
        uniform_crossover_prob = safe_convert('uniform_crossover_prob', float, 0.5)
        # Explicitly check tuple conversion for range
        uniform_mutation_range = safe_convert('uniform_mutation_range', tuple, (-1.0, 1.0), is_tuple=True, expected_len=2)

        # Init Mutation Params (use the base rates/strengths as defaults passed to safe_convert)
        init_mutation_rate = safe_convert('init_mutation_rate', float, mutation_rate)
        init_mutation_strength = safe_convert('init_mutation_strength', float, mutation_strength)

        # Other Config
        final_model_filename = f"evolved_{task_id}.pth"
        final_model_path = os.path.join(RESULT_DIR, final_model_filename)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Update logging with converted values
        logger.info(f"[Task {task_id}] Using device: {device}")
        # ... (CUDA info logging) ...
        logger.info(f"[Task {task_id}] Config Parsed: Gens={generations}, Pop={population_size}, Select='{selection_strategy}', Cross='{crossover_operator}', Mut='{mutation_operator}', Elitism={elitism_count}, MutRate={mutation_rate:.2f}, MutStr={mutation_strength:.3f}")

        # Initial state update
        self.update_state(state='STARTED', meta={
            'progress': 0.0, 'message': 'Initialization starting...',
            'fitness_history': [], 'avg_fitness_history': [], 'diversity_history': []
        })

        # --- Load Task Evaluation Function ---
        task_eval_func = load_task_eval_function(actual_eval_script_path)

        # --- Load Initial Model & Weights ---
        logger.info(f"[Task {task_id}] Loading initial model '{model_class}'...")
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
            # --- Use configured mutation operator for initialization ---
            if mutation_operator == "gaussian":
                mutated_initial = mutate_gaussian(initial_weights, init_mutation_rate, init_mutation_strength)
            elif mutation_operator == "uniform_random":
                mutated_initial = mutate_uniform_random(initial_weights, init_mutation_rate, value_range=uniform_mutation_range)
            else:
                logger.warning(f"Unsupported mutation operator '{mutation_operator}' for init, defaulting to gaussian.")
                mutated_initial = mutate_gaussian(initial_weights, init_mutation_rate, init_mutation_strength)
            population.append(mutated_initial)

        self.update_state(state='PROGRESS', meta={
            'progress': 0.01, 'message': 'Initialization complete. Starting generations...',
             'fitness_history': [], 'avg_fitness_history': [], 'diversity_history': []
        })

    except Exception as init_err:
        # ...(Keep existing error handling for initialization)...
        error_msg = f"Initialization failed: {init_err}"
        logger.error(f"[Task {task_id}] {error_msg}", exc_info=True)
        self.update_state(state='FAILURE', meta={'message': error_msg, 'error': str(init_err)})
        raise # Re-raise to let Celery handle retry/failure


    # --- Evolution Loop ---
    best_fitness_overall = -float('inf')
    best_weights_overall = initial_weights.copy()
    fitness_history_overall = []
    avg_fitness_history_overall = []
    diversity_history_overall = []

    try:
        for gen in range(generations):
            gen_num = gen + 1
            logger.info(f"[Task {task_id}] --- Generation {gen_num}/{generations} ---")
            gen_start_time = time.time()

            # 1. Evaluate Population
            # ...(Keep evaluation logic as before)...
            try:
                fitness_scores = evaluate_population_step(
                    population, model_definition_path, model_class, task_eval_func, device, model_args, model_kwargs
                )
                fitness_scores = [float(f) if f is not None else -float('inf') for f in fitness_scores]
            except Exception as eval_err:
                logger.error(f"[Task {task_id}] Error during evaluation in Gen {gen_num}: {eval_err}", exc_info=False)
                fitness_scores = [-float('inf')] * len(population)


            # 2. Process Results & Calculate Metrics
            # ...(Keep metric calculation and best update logic as before)...
            valid_scores = [f for f in fitness_scores if f > -float('inf')]
            if not valid_scores:
                 error_msg = f"All individuals failed evaluation in Gen {gen_num}."
                 logger.error(f"[Task {task_id}] {error_msg}")
                 raise RuntimeError(error_msg)
            max_fitness = np.max(valid_scores)
            avg_fitness = np.mean(valid_scores)
            diversity = calculate_population_diversity(population)
            best_idx_current = np.argmax(fitness_scores)
            fitness_history_overall.append(float(max_fitness))
            avg_fitness_history_overall.append(float(avg_fitness))
            diversity_history_overall.append(float(diversity))
            logger.info(f"[Task {task_id}] Gen {gen_num} Stats: MaxFit={max_fitness:.4f}, AvgFit={avg_fitness:.4f}, Diversity={diversity:.4f}")
            current_best_fitness = fitness_scores[best_idx_current]
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                best_weights_overall = population[best_idx_current].copy()
                logger.info(f"[Task {task_id}] *** New best overall fitness: {best_fitness_overall:.4f} ***")

            # --- Update Task State (remains the same) ---
            current_progress = gen_num / generations
            progress_message = f'Gen {gen_num}/{generations} | Best Fit: {best_fitness_overall:.4f} | Avg Fit: {avg_fitness:.4f}'
            self.update_state(
                state='PROGRESS', meta={ 'progress': current_progress, 'message': progress_message, 'fitness_history': fitness_history_overall, 'avg_fitness_history': avg_fitness_history_overall, 'diversity_history': diversity_history_overall }
            )

            # --- 3. SELECTION ---
            num_parents_to_select = population_size # Select enough to potentially fill next gen via crossover
            parents = []
            logger.debug(f"[Task {task_id}] Performing Selection using '{selection_strategy}'...")
            if selection_strategy == "tournament":
                # Pass tournament_size from config
                parents = select_parents_tournament(population, fitness_scores, num_parents_to_select, tournament_size=tournament_size)
            elif selection_strategy == "roulette":
                parents = select_parents_roulette(population, fitness_scores, num_parents_to_select)
            else: # Default/fallback
                 logger.warning(f"Unsupported selection strategy '{selection_strategy}', defaulting to tournament.")
                 parents = select_parents_tournament(population, fitness_scores, num_parents_to_select, tournament_size=tournament_size)

            # Handle selection failure
            if not parents or len(parents) < 2:
                logger.warning(f"[Task {task_id}] Parent selection yielded < 2 parents ({len(parents)}) in Gen {gen_num}. Re-populating from best.")
                # Fallback: Repopulate with mutated copies of the overall best
                population = [best_weights_overall.copy()] + [mutate_gaussian(best_weights_overall, mutation_rate, mutation_strength) for _ in range(population_size - 1)]
                continue # Skip to next generation

            # --- 4. REPRODUCTION (Elitism + Crossover + Mutation) ---
            next_population = []
            logger.debug(f"[Task {task_id}] Performing Reproduction (Elitism={elitism_count}, Cross='{crossover_operator}', Mut='{mutation_operator}')...")

            # Elitism: Add best individuals from current generation directly
            if elitism_count > 0:
                 # Sort by fitness descending, take top N indices
                 elite_indices = np.argsort(fitness_scores)[::-1][:elitism_count]
                 for elite_idx in elite_indices:
                      # Ensure not adding duplicates if best was selected multiple ways
                      if len(next_population) < population_size and elite_idx < len(population):
                           next_population.append(population[elite_idx].copy())

            # Fill remaining spots with offspring
            while len(next_population) < population_size:
                try:
                    # --- Choose Parents (randomly from selected parent pool) ---
                    p1, p2 = random.sample(parents, 2)

                    # --- CROSSOVER ---
                    child1_weights, child2_weights = None, None
                    if crossover_operator == "one_point":
                         child1_weights, child2_weights = crossover_one_point(p1, p2)
                    elif crossover_operator == "uniform":
                         # Pass crossover probability from config
                         child1_weights, child2_weights = crossover_uniform(p1, p2, crossover_prob=uniform_crossover_prob)
                    elif crossover_operator == "average":
                         child1_weights, child2_weights = crossover_average(p1, p2)
                    else: # Default/fallback
                         logger.warning(f"Unsupported crossover operator '{crossover_operator}', defaulting to one_point.")
                         child1_weights, child2_weights = crossover_one_point(p1, p2)

                    # --- MUTATION ---
                    mutated_child1, mutated_child2 = None, None
                    if mutation_operator == "gaussian":
                         mutated_child1 = mutate_gaussian(child1_weights, mutation_rate, mutation_strength)
                         mutated_child2 = mutate_gaussian(child2_weights, mutation_rate, mutation_strength)
                    elif mutation_operator == "uniform_random":
                         # Pass value range from config
                         mutated_child1 = mutate_uniform_random(child1_weights, mutation_rate, value_range=uniform_mutation_range)
                         mutated_child2 = mutate_uniform_random(child2_weights, mutation_rate, value_range=uniform_mutation_range)
                    else: # Default/fallback
                         logger.warning(f"Unsupported mutation operator '{mutation_operator}', defaulting to gaussian.")
                         mutated_child1 = mutate_gaussian(child1_weights, mutation_rate, mutation_strength)
                         mutated_child2 = mutate_gaussian(child2_weights, mutation_rate, mutation_strength)

                    # Add offspring to next population if space available
                    if len(next_population) < population_size:
                        next_population.append(mutated_child1)
                    if len(next_population) < population_size:
                        next_population.append(mutated_child2)

                except Exception as repr_err:
                     logger.warning(f"[Task {task_id}] Reproduction step error in Gen {gen_num}: {repr_err}. Substituting.")
                     # Fallback: Add a mutated version of a random parent
                     if parents and len(next_population) < population_size:
                          substitute = mutate_gaussian(random.choice(parents), mutation_rate, mutation_strength) # Default mutation for fallback
                          next_population.append(substitute)

            # Update population for the next generation
            population = next_population[:population_size] # Ensure exact size
            gen_time = time.time() - gen_start_time
            logger.info(f"[Task {task_id}] Gen {gen_num} finished in {gen_time:.2f}s")

        # --- Evolution Finished ---
        logger.info(f"[Task {task_id}] Evolution finished after {generations} generations.")

        # --- Save Final Best Model ---
        # ...(Keep saving logic as before)...
        logger.info(f"[Task {task_id}] Saving best model (Fitness: {best_fitness_overall:.4f}) to {final_model_path}...")
        final_best_model = load_pytorch_model(model_definition_path, model_class, None, device, *model_args, **model_kwargs)
        load_weights_from_flat(final_best_model, best_weights_overall)
        torch.save(final_best_model.state_dict(), final_model_path)
        logger.info(f"[Task {task_id}] Best model saved successfully.")


        # --- Clean up uploaded files ---
        # ...(Keep cleanup logic as before)...
        logger.info(f"[Task {task_id}] Cleaning up temporary uploaded files...")
        # ... (os.remove loops) ...

        # --- Return final result data including ALL histories ---
        success_message = f'Evolution completed successfully. Best fitness: {best_fitness_overall:.4f}.'
        final_result = {
            'message': success_message,
            'final_model_path': final_model_path,
            'best_fitness': float(best_fitness_overall) if best_fitness_overall > -np.inf else None,
            # Final cumulative histories
            'fitness_history': fitness_history_overall,
            'avg_fitness_history': avg_fitness_history_overall,
            'diversity_history': diversity_history_overall
        }
        logger.info(f"[Task {task_id}] Task successful.")
        return final_result # Return dict for Celery result backend

    except Exception as e:
        # ...(Keep main error handling block as before)...
        error_message = f'Evolution task failed during run: {str(e)}'
        logger.error(f"[Task {task_id}] {error_message}", exc_info=True)
        meta_fail = {
            'message': error_message, 'error': str(e),
            'fitness_history': fitness_history_overall,
            'avg_fitness_history': avg_fitness_history_overall,
            'diversity_history': diversity_history_overall
        }
        self.update_state(state='FAILURE', meta=meta_fail)
        raise
