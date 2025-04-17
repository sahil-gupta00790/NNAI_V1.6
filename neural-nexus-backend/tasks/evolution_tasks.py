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
    crossover_average,
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
def calculate_population_diversity(population_weights_list: list[np.ndarray]) -> float:
    """Calculates average pairwise Euclidean distance between weight vectors."""
    if not population_weights_list or len(population_weights_list) < 2: return 0.0
    num_individuals = len(population_weights_list)
    # Ensure weights are numpy arrays before flattening
    flat_weights = [w.flatten() for w in population_weights_list if isinstance(w, np.ndarray)]
    if len(flat_weights) < 2: return 0.0

    # Check for consistent lengths, excluding inconsistent ones
    first_len = len(flat_weights[0])
    consistent_weights = [w for w in flat_weights if len(w) == first_len]
    if len(consistent_weights) < 2:
        logger.warning("Less than 2 weight vectors with consistent lengths found for diversity calc.")
        return 0.0
    if len(consistent_weights) < num_individuals:
         logger.warning(f"Inconsistent weight vector lengths in population for diversity calc. Using {len(consistent_weights)} individuals.")
         flat_weights = consistent_weights
         num_individuals = len(flat_weights)

    total_distance = 0.0
    num_pairs = 0
    for i in range(num_individuals):
        for j in range(i + 1, num_individuals):
            # Ensure subtraction is possible (should be due to consistency check)
            try:
                distance = np.linalg.norm(flat_weights[i] - flat_weights[j])
                total_distance += distance
                num_pairs += 1
            except ValueError as e:
                 logger.warning(f"Could not calculate distance between individuals {i} and {j}: {e}")

    return (total_distance / num_pairs) if num_pairs > 0 else 0.0

# --- Task Base Class with Retry ---
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
    task_id = None # Initialize task_id
    try:
         task_id = self.request.id
         logger.info(f"[Task {task_id}] Starting evolution...")
         # Use a preview to avoid overly long log messages
         config_preview = {k: v for i, (k, v) in enumerate(config.items()) if i < 10} # Log more items
         logger.info(f"[Task {task_id}] Received config dictionary preview: {config_preview}")

    except Exception as entry_err:
         logger.critical(f"CRITICAL ERROR AT TASK ENTRY (task_id may be None): {entry_err}", exc_info=True)
         raise # Re-raise to ensure Celery knows it failed fundamentally

    # --- Determine Evaluation Script ---
    if use_standard_eval:
        actual_eval_script_path = STANDARD_EVAL_SCRIPT_PATH
        logger.info(f"[Task {task_id}] Using standard evaluation script: {actual_eval_script_path}")
        if not os.path.exists(actual_eval_script_path):
            error_msg = f"Standard evaluation script not found at configured path: {actual_eval_script_path}"
            logger.error(f"[Task {task_id}] {error_msg}")
            raise FileNotFoundError(error_msg)
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
                if default_value is None and not is_tuple: # Allow None for non-tuple optionals
                    return None
                logger.warning(f"Config key '{key}' is None, using default: {default_value}")
                value = default_value

            try:
                if is_tuple:
                    if not isinstance(value, (list, tuple)):
                         raise TypeError(f"Value for '{key}' must be a list or tuple")
                    if expected_len is not None and len(value) != expected_len:
                        raise ValueError(f"Value for '{key}' must have length {expected_len}")
                    converted_value = tuple(float(v) for v in value)
                elif target_type is float:
                    converted_value = float(value)
                elif target_type is int:
                    converted_value = int(value)
                elif target_type is str:
                    if key in ["selection_strategy", "crossover_operator", "mutation_operator"]:
                         converted_value = str(value).lower()
                    else:
                         converted_value = str(value)
                elif target_type is bool:
                     if isinstance(value, str):
                          converted_value = value.lower() in ['true', '1', 'yes']
                     else:
                          converted_value = bool(value)
                else: # list, dict assumed okay if not None
                     converted_value = value
                return converted_value
            except (TypeError, ValueError) as conv_err:
                logger.error(f"Invalid type for config key '{key}'. Expected {target_type.__name__} convertible, got {type(value)} ({value}). Error: {conv_err}. Using default: {default_value}", exc_info=True)
                try: # Try converting the default itself
                    if is_tuple: return tuple(float(v) for v in default_value)
                    else: return target_type(default_value)
                except Exception as default_err:
                    logger.critical(f"Default value '{default_value}' for key '{key}' is invalid for type {target_type.__name__}! Error: {default_err}", exc_info=True)
                    raise ValueError(f"Invalid configuration for '{key}' and invalid default value provided.")

        # --- General Config ---
        generations = safe_convert('generations', int, 10)
        population_size = safe_convert('population_size', int, 20)
        model_class = config.get('model_class') # Get raw value first
        if not model_class or not isinstance(model_class, str) or not model_class.strip():
             raise ValueError("'model_class' (string) is required in the configuration JSON.")
        model_class = model_class.strip()

        model_args = safe_convert('model_args', list, [])
        model_kwargs = safe_convert('model_kwargs', dict, {})
        # Extract eval_config separately - THIS IS IMPORTANT
        eval_config = safe_convert('eval_config', dict, {})

        # --- GA Operator Config ---
        selection_strategy = safe_convert("selection_strategy", str, "tournament")
        crossover_operator = safe_convert("crossover_operator", str, "one_point")
        mutation_operator = safe_convert("mutation_operator", str, "gaussian")
        elitism_count = safe_convert("elitism_count", int, 1)
        if elitism_count < 0: elitism_count = 0
        if elitism_count >= population_size: elitism_count = max(0, population_size - 1)

        # --- Operator-Specific Params ---
        mutation_rate = safe_convert('mutation_rate', float, 0.1) # Fixed rate if dynamic is off
        mutation_strength = safe_convert('mutation_strength', float, 0.05)
        tournament_size = safe_convert('tournament_size', int, 3)
        uniform_crossover_prob = safe_convert('uniform_crossover_prob', float, 0.5)
        uniform_mutation_range = safe_convert('uniform_mutation_range', tuple, (-1.0, 1.0), is_tuple=True, expected_len=2)

        # --- Dynamic Mutation Params (Extract if present) ---
        use_dynamic_mutation_rate = safe_convert('use_dynamic_mutation_rate', bool, False)
        dynamic_mutation_heuristic = safe_convert('dynamic_mutation_heuristic', str, 'time_decay')
        initial_mutation_rate = safe_convert('initial_mutation_rate', float, mutation_rate)
        final_mutation_rate = safe_convert('final_mutation_rate', float, 0.01)
        high_fitness_mutation_rate = safe_convert('high_fitness_mutation_rate', float, 0.05)
        low_fitness_mutation_rate = safe_convert('low_fitness_mutation_rate', float, 0.25)
        base_mutation_rate = safe_convert('base_mutation_rate', float, mutation_rate)
        diversity_threshold_low = safe_convert('diversity_threshold_low', float, 0.1)
        mutation_rate_increase_factor = safe_convert('mutation_rate_increase_factor', float, 1.5)

        # Other Config
        final_model_filename = f"evolved_{task_id}.pth"
        final_model_path = os.path.join(RESULT_DIR, final_model_filename)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[Task {task_id}] Using device: {device}")
        if device.type == 'cuda': logger.info(f"[Task {task_id}] CUDA Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"[Task {task_id}] Config Parsed: Gens={generations}, Pop={population_size}, Select='{selection_strategy}', Cross='{crossover_operator}', Mut='{mutation_operator}', Elitism={elitism_count}")
        if use_dynamic_mutation_rate:
             logger.info(f"[Task {task_id}] Dynamic Mutation Enabled: Heuristic='{dynamic_mutation_heuristic}'")
        else:
             logger.info(f"[Task {task_id}] Using Fixed Mutation Rate: {mutation_rate:.3f}")
        logger.info(f"[Task {task_id}] Using Mutation Strength: {mutation_strength:.4f}")
        logger.info(f"[Task {task_id}] Eval Config to be used: {eval_config}") # Log extracted eval_config

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
        init_rate_for_spread = initial_mutation_rate if use_dynamic_mutation_rate else mutation_rate
        init_strength_for_spread = mutation_strength # Using base strength for init

        for i in range(population_size - 1):
            if mutation_operator == "gaussian":
                mutated_initial = mutate_gaussian(initial_weights, init_rate_for_spread, init_strength_for_spread)
            elif mutation_operator == "uniform_random":
                mutated_initial = mutate_uniform_random(initial_weights, init_rate_for_spread, value_range=uniform_mutation_range)
            else:
                logger.warning(f"Unsupported mutation operator '{mutation_operator}' for init, defaulting to gaussian.")
                mutated_initial = mutate_gaussian(initial_weights, init_rate_for_spread, init_strength_for_spread)
            population.append(mutated_initial)

        self.update_state(state='PROGRESS', meta={
            'progress': 0.01, 'message': 'Initialization complete. Starting generations...',
             'fitness_history': [], 'avg_fitness_history': [], 'diversity_history': []
        })

    except Exception as init_err:
        error_msg = f"Initialization failed: {init_err}"
        logger.error(f"[Task {task_id}] {error_msg}", exc_info=True)
        self.update_state(state='FAILURE', meta={'message': error_msg, 'error': str(init_err)})
        raise

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

            # --- Determine Current Mutation Rate for this Generation ---
            current_mutation_rate = mutation_rate # Default to fixed rate
            if use_dynamic_mutation_rate:
                # Implement dynamic rate calculation based on 'dynamic_mutation_heuristic'
                if dynamic_mutation_heuristic == 'time_decay':
                    progress = gen / max(1, generations - 1)
                    current_mutation_rate = initial_mutation_rate - (initial_mutation_rate - final_mutation_rate) * progress
                    current_mutation_rate = max(final_mutation_rate, current_mutation_rate)
                # Add elif blocks for 'fitness_based' and 'diversity_based' here
                # Need fitness_scores (from prev gen) or diversity (calculated)
                # elif dynamic_mutation_heuristic == 'diversity_based':
                #     diversity = calculate_population_diversity(population) # Calculate earlier if needed
                #     if diversity < diversity_threshold_low:
                #         current_mutation_rate = base_mutation_rate * mutation_rate_increase_factor
                #     else:
                #         current_mutation_rate = base_mutation_rate
                else:
                     # Placeholder/default if heuristic logic not fully implemented
                     current_mutation_rate = base_mutation_rate # Default for others for now

            # 1. Evaluate Population
            try:
                # *** FIX: Pass the extracted eval_config dictionary ***
                fitness_scores = evaluate_population_step(
                    population, model_definition_path, model_class, task_eval_func, device, model_args, model_kwargs, eval_config=eval_config # Added eval_config
                )
                # Convert potential None scores (errors) to -inf
                fitness_scores = [float(f) if f is not None and np.isfinite(f) else -float('inf') for f in fitness_scores]

            except Exception as eval_err:
                logger.error(f"[Task {task_id}] Error during population evaluation step in Gen {gen_num}: {eval_err}", exc_info=True)
                fitness_scores = [-float('inf')] * len(population)

            # 2. Process Results & Calculate Metrics
            valid_scores = [f for f in fitness_scores if f > -float('inf')]
            if not valid_scores:
                 error_msg = f"All individuals failed evaluation in Gen {gen_num}."
                 logger.error(f"[Task {task_id}] {error_msg}")
                 # Raise error to stop execution if no individuals are valid
                 raise RuntimeError(error_msg)

            max_fitness = np.max(valid_scores)
            avg_fitness = np.mean(valid_scores)
            diversity = calculate_population_diversity(population)
            fitness_history_overall.append(float(max_fitness))
            avg_fitness_history_overall.append(float(avg_fitness))
            diversity_history_overall.append(float(diversity))

            logger.info(f"[Task {task_id}] Gen {gen_num} Stats: MaxFit={max_fitness:.4f}, AvgFit={avg_fitness:.4f}, Diversity={diversity:.4f}, RateUsed={current_mutation_rate:.3f}")

            # Update overall best
            best_idx_current = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[best_idx_current]
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                if best_idx_current < len(population):
                     best_weights_overall = population[best_idx_current].copy()
                     logger.info(f"[Task {task_id}] *** New best overall fitness: {best_fitness_overall:.4f} ***")
                else:
                     logger.warning(f"[Task {task_id}] Best index {best_idx_current} out of bounds for population size {len(population)}.")

            # --- Update Task State ---
            current_progress = gen_num / generations
            progress_message = f'Gen {gen_num}/{generations} | Best Fit: {best_fitness_overall:.4f} | Avg Fit: {avg_fitness:.4f}'
            self.update_state(
                state='PROGRESS', meta={ 'progress': current_progress, 'message': progress_message, 'fitness_history': fitness_history_overall, 'avg_fitness_history': avg_fitness_history_overall, 'diversity_history': diversity_history_overall }
            )

            # --- 3. SELECTION ---
            num_parents_to_select = population_size
            parents = []
            logger.debug(f"[Task {task_id}] Performing Selection using '{selection_strategy}'...")
            selectable_indices = [i for i, f in enumerate(fitness_scores) if f > -float('inf')]
            selectable_population = [population[i] for i in selectable_indices]
            selectable_fitness = [fitness_scores[i] for i in selectable_indices]

            if not selectable_indices: # Should not happen due to RuntimeError above, but keep as safeguard
                 logger.warning(f"[Task {task_id}] No individuals selectable in Gen {gen_num}. Re-populating.")
                 population = [best_weights_overall.copy()] + [mutate_gaussian(best_weights_overall, current_mutation_rate, mutation_strength) for _ in range(population_size - 1)]
                 continue

            if selection_strategy == "tournament":
                parents = select_parents_tournament(selectable_population, selectable_fitness, num_parents_to_select, tournament_size=tournament_size)
            elif selection_strategy == "roulette":
                parents = select_parents_roulette(selectable_population, selectable_fitness, num_parents_to_select)
            else:
                 logger.warning(f"Unsupported selection strategy '{selection_strategy}', defaulting to tournament.")
                 parents = select_parents_tournament(selectable_population, selectable_fitness, num_parents_to_select, tournament_size=tournament_size)

            if not parents or len(parents) < 2:
                logger.warning(f"[Task {task_id}] Parent selection yielded < 2 parents ({len(parents)}) in Gen {gen_num}. Re-populating.")
                population = [best_weights_overall.copy()] + [mutate_gaussian(best_weights_overall, current_mutation_rate, mutation_strength) for _ in range(population_size - 1)]
                continue

            # --- 4. REPRODUCTION (Elitism + Crossover + Mutation) ---
            next_population = []
            logger.debug(f"[Task {task_id}] Performing Reproduction (Elitism={elitism_count}, Cross='{crossover_operator}', Mut='{mutation_operator}')...")

            # Elitism
            if elitism_count > 0:
                 elite_indices = np.argsort(fitness_scores)[::-1][:elitism_count]
                 for elite_idx in elite_indices:
                      if len(next_population) < population_size and elite_idx < len(population):
                           next_population.append(population[elite_idx].copy())

            # Offspring
            while len(next_population) < population_size:
                try:
                    if len(parents) < 2: p1, p2 = parents[0], parents[0]
                    else: p1, p2 = random.sample(parents, 2)

                    # Crossover
                    child1_weights, child2_weights = None, None
                    if crossover_operator == "one_point": child1_weights, child2_weights = crossover_one_point(p1, p2)
                    elif crossover_operator == "uniform": child1_weights, child2_weights = crossover_uniform(p1, p2, crossover_prob=uniform_crossover_prob)
                    elif crossover_operator == "average": child1_weights, child2_weights = crossover_average(p1, p2)
                    else: child1_weights, child2_weights = crossover_one_point(p1, p2) # Default

                    # Mutation (using current_mutation_rate)
                    mutated_child1, mutated_child2 = None, None
                    if mutation_operator == "gaussian":
                         mutated_child1 = mutate_gaussian(child1_weights, current_mutation_rate, mutation_strength)
                         mutated_child2 = mutate_gaussian(child2_weights, current_mutation_rate, mutation_strength)
                    elif mutation_operator == "uniform_random":
                         mutated_child1 = mutate_uniform_random(child1_weights, current_mutation_rate, value_range=uniform_mutation_range)
                         mutated_child2 = mutate_uniform_random(child2_weights, current_mutation_rate, value_range=uniform_mutation_range)
                    else:
                         mutated_child1 = mutate_gaussian(child1_weights, current_mutation_rate, mutation_strength) # Default
                         mutated_child2 = mutate_gaussian(child2_weights, current_mutation_rate, mutation_strength) # Default

                    if len(next_population) < population_size: next_population.append(mutated_child1)
                    if len(next_population) < population_size: next_population.append(mutated_child2)

                except Exception as repr_err:
                     logger.warning(f"[Task {task_id}] Reproduction step error in Gen {gen_num}: {repr_err}. Substituting.", exc_info=True)
                     if len(next_population) < population_size:
                          substitute_parent = random.choice(parents) if parents else best_weights_overall
                          substitute = mutate_gaussian(substitute_parent, current_mutation_rate, mutation_strength)
                          next_population.append(substitute)

            population = next_population[:population_size]
            gen_time = time.time() - gen_start_time
            logger.info(f"[Task {task_id}] Gen {gen_num} finished in {gen_time:.2f}s")

        # --- Evolution Finished ---
        if best_fitness_overall == -float('inf'):
             raise RuntimeError("Evolution completed, but no individuals ever evaluated successfully.")
        logger.info(f"[Task {task_id}] Evolution finished after {generations} generations.")

        # --- Save Final Best Model ---
        logger.info(f"[Task {task_id}] Saving best model (Fitness: {best_fitness_overall:.4f}) to {final_model_path}...")
        final_best_model = load_pytorch_model(model_definition_path, model_class, None, device, *model_args, **model_kwargs)
        load_weights_from_flat(final_best_model, best_weights_overall)
        torch.save(final_best_model.state_dict(), final_model_path)
        logger.info(f"[Task {task_id}] Best model saved successfully.")

        # --- Clean up uploaded files ---
        logger.info(f"[Task {task_id}] Cleaning up temporary uploaded files...")
        files_to_remove = [model_definition_path, task_evaluation_path, initial_weights_path]
        for f_path in files_to_remove:
             if f_path and os.path.exists(f_path):
                  try:
                       os.remove(f_path)
                       logger.debug(f"[Task {task_id}] Removed file: {f_path}")
                  except OSError as rm_err:
                       logger.warning(f"[Task {task_id}] Could not remove temporary file {f_path}: {rm_err}")

        # --- Return final result data ---
        success_message = f'Evolution completed successfully. Best fitness: {best_fitness_overall:.4f}.'
        final_result = {
            'message': success_message,
            'final_model_path': final_model_path,
            'best_fitness': float(best_fitness_overall),
            'fitness_history': fitness_history_overall,
            'avg_fitness_history': avg_fitness_history_overall,
            'diversity_history': diversity_history_overall
        }
        logger.info(f"[Task {task_id}] Task successful.")
        self.update_state(state='SUCCESS', meta=final_result)
        return final_result

    except Exception as e:
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
