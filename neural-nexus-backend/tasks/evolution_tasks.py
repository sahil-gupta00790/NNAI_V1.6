# tasks/evolution_tasks.py

from app.core.celery_app import celery_app
from celery import current_task, Task
import time
import random
import os
import numpy as np
import torch
import logging
from typing import List, Dict, Any, Tuple
import redis
from celery.exceptions import SoftTimeLimitExceeded

# Use config settings imported via Celery context or directly
from app.core.config import settings
# --- Import ALL helper functions ---
from app.utils.evolution_helpers import (
    load_pytorch_model, flatten_weights, load_task_eval_function,
    evaluate_population_step, load_weights_from_flat,
    decode_hyperparameters,
    # Selection
    select_parents_tournament,
    select_parents_roulette,
    # Crossover
    crossover_one_point,
    crossover_uniform,
    crossover_average,
    # Mutation
    mutate_hyperparams_gaussian,
    mutate_weights_gaussian,
    mutate_weights_uniform_random
)

logger = logging.getLogger(__name__)

STANDARD_EVAL_SCRIPT_PATH = settings.STANDARD_EVAL_SCRIPT_PATH
RESULT_DIR = settings.RESULT_DIR
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Helper Function for Diversity (Modified) ---
def calculate_population_diversity(population: list[np.ndarray], num_hyperparams: int) -> float:
    """Calculates average pairwise Euclidean distance between the weight vectors."""
    if not population or len(population) < 2 or num_hyperparams < 0: return 0.0
    num_individuals = len(population)
    weight_vectors = [ind[num_hyperparams:] for ind in population if isinstance(ind, np.ndarray) and len(ind) > num_hyperparams]
    if len(weight_vectors) < 2: return 0.0
    flat_weights = [w.flatten() for w in weight_vectors]
    if len(flat_weights) < 2: return 0.0
    first_len = len(flat_weights[0])
    consistent_weights = [w for w in flat_weights if len(w) == first_len]
    num_consistent = len(consistent_weights)
    if num_consistent < 2:
        logger.debug("Less than 2 consistent weight vectors for diversity calc.") # Less alarming
        return 0.0
    if num_consistent < num_individuals:
         logger.debug(f"Using {num_consistent}/{num_individuals} individuals for diversity.") # Less alarming
    total_distance = 0.0
    num_pairs = 0
    for i in range(num_consistent):
        for j in range(i + 1, num_consistent):
            try:
                # Using float32 for potentially better performance on norm
                distance = np.linalg.norm(consistent_weights[i].astype(np.float32) - consistent_weights[j].astype(np.float32))
                total_distance += distance
                num_pairs += 1
            except (ValueError, FloatingPointError) as e: # Catch potential FP errors
                 logger.warning(f"Error calculating distance between individuals {i} and {j}: {e}")
    return float(total_distance / num_pairs) if num_pairs > 0 else 0.0

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
def run_evolution_task(self: Task, model_definition_path: str, task_evaluation_path: str | None, use_standard_eval: bool, initial_weights_path: str | None, config: Dict[str, Any]):
    """ Celery task for evolution with dynamic rates, hyperparameter optimization, and cooperative halt check. """
    task_id = None
    redis_client = None # Initialize redis client variable
    halt_key = None

    try:
         task_id = self.request.id
         if not task_id: logger.error("CRITICAL: Failed to get task_id"); raise ValueError("Task ID missing")
         # Construct key early, now that halt_key exists in this scope
         halt_key = f"task:halt:{task_id}"
         logger.info(f"[Task {task_id}] Starting evolution...")
         config_preview = {k: v for i, (k, v) in enumerate(config.items()) if i < 15}
         logger.info(f"[Task {task_id}] Received config preview: {config_preview}")

         # --- NEW: Initialize Redis Client ---
         try:
             redis_url = str(settings.REDIS_URL)
             # Use decode_responses=True for easier handling of keys/values as strings
             redis_client = redis.from_url(redis_url, decode_responses=True)
             # Test connection (optional but good practice)
             redis_client.ping()
             logger.info(f"[Task {task_id}] Successfully connected to Redis at {redis_url}")
         except Exception as redis_err:
             # Log warning but allow task to continue if Redis isn't essential for core logic (only for halting)
             logger.warning(f"[Task {task_id}] Failed to connect to Redis for halt check: {redis_err}. Halt feature disabled for this run.", exc_info=False)
             redis_client = None # Ensure client is None if connection failed
         # --- End Redis Init ---

    except Exception as entry_err:
         logger.critical(f"CRITICAL ERROR AT TASK ENTRY: {entry_err}", exc_info=True)
         raise

    # --- Determine Evaluation Script ---
    if use_standard_eval:
        actual_eval_script_path = STANDARD_EVAL_SCRIPT_PATH
        logger.info(f"[Task {task_id}] Using standard evaluation script: {actual_eval_script_path}")
        if not os.path.exists(actual_eval_script_path):
            raise FileNotFoundError(f"Standard eval script not found: {actual_eval_script_path}")
    elif task_evaluation_path and os.path.exists(task_evaluation_path):
        actual_eval_script_path = task_evaluation_path
        logger.info(f"[Task {task_id}] Using custom evaluation script: {actual_eval_script_path}")
    else:
        raise FileNotFoundError(f"Custom eval script path invalid or not found: {task_evaluation_path}")

    # --- Setup & Config Extraction ---
    try:
        # Helper function (remains mostly the same)
        def safe_convert(key, target_type, default_value, is_tuple=False, expected_len=None):
            value = config.get(key, default_value)
            if value is None:
                if default_value is None and not is_tuple: return None
                logger.debug(f"Config '{key}' is None, using default: {default_value}") # Less alarming
                value = default_value
            try:
                if is_tuple:
                    if not isinstance(value, (list, tuple)): raise TypeError(f"'{key}' must be list/tuple")
                    if expected_len is not None and len(value) != expected_len: raise ValueError(f"'{key}' needs length {expected_len}")
                    return tuple(float(v) for v in value)
                if target_type is float: return float(value)
                if target_type is int: return int(value)
                if target_type is str: return str(value).lower() if key in ["selection_strategy", "crossover_operator", "mutation_operator", "dynamic_mutation_heuristic"] else str(value)
                if target_type is bool: return str(value).lower() in ['true', '1', 'yes'] if isinstance(value, str) else bool(value)
                return value # list, dict assumed okay
            except (TypeError, ValueError) as conv_err:
                logger.error(f"Invalid type for config '{key}'. Expected {target_type.__name__}, got {type(value)} ({value}). Error: {conv_err}. Using default: {default_value}", exc_info=False)
                try:
                    if is_tuple: return tuple(float(v) for v in default_value)
                    else: return target_type(default_value)
                except Exception as default_err:
                    logger.critical(f"Default value '{default_value}' for '{key}' is invalid! Error: {default_err}", exc_info=True)
                    raise ValueError(f"Invalid config for '{key}' and invalid default.")

        # --- General Config ---
        generations = safe_convert('generations', int, 10)
        population_size = safe_convert('population_size', int, 20)
        model_class = safe_convert('model_class', str, None)
        if not model_class: raise ValueError("'model_class' is required.")
        model_args = safe_convert('model_args', list, [])
        model_kwargs_static = safe_convert('model_kwargs', dict, {})
        eval_config = safe_convert('eval_config', dict, {})

        # --- Hyperparameter Evolution Config ---
        evolvable_hyperparams_config: Dict[str, Dict[str, Any]] = config.get('evolvable_hyperparams', {})
        hyperparam_keys: List[str] = list(evolvable_hyperparams_config.keys())
        num_hyperparams: int = len(hyperparam_keys)
        logger.info(f"[Task {task_id}] Evolving {num_hyperparams} hyperparameters: {hyperparam_keys}")
        hyperparam_mutation_strength: float = safe_convert('hyperparam_mutation_strength', float, 0.02)

        # --- GA Operator Config ---
        selection_strategy = safe_convert("selection_strategy", str, "tournament")
        crossover_operator = safe_convert("crossover_operator", str, "one_point")
        mutation_operator = safe_convert("mutation_operator", str, "gaussian")
        elitism_count = safe_convert("elitism_count", int, 1)
        elitism_count = max(0, min(elitism_count, population_size - 1))

        # --- Operator-Specific Params (Weights) ---
        mutation_rate = safe_convert('mutation_rate', float, 0.1) # Base fixed rate
        mutation_strength = safe_convert('mutation_strength', float, 0.05)
        tournament_size = safe_convert('tournament_size', int, 3)
        uniform_crossover_prob = safe_convert('uniform_crossover_prob', float, 0.5)
        uniform_mutation_range = safe_convert('uniform_mutation_range', tuple, (-1.0, 1.0), is_tuple=True, expected_len=2)

        # --- Dynamic Weight Mutation Rate Params ---
        use_dynamic_mutation_rate = safe_convert('use_dynamic_mutation_rate', bool, False)
        dynamic_mutation_heuristic = safe_convert('dynamic_mutation_heuristic', str, 'time_decay')
        initial_mutation_rate = safe_convert('initial_mutation_rate', float, mutation_rate) # Default based on fixed rate
        final_mutation_rate = safe_convert('final_mutation_rate', float, 0.01)
        # Renamed for clarity based on implementation: high_fitness_rate is the 'normal' rate
        normal_fitness_mutation_rate = safe_convert('high_fitness_mutation_rate', float, 0.05)
        stagnation_mutation_rate = safe_convert('low_fitness_mutation_rate', float, 0.25)
        stagnation_threshold: float = safe_convert('stagnation_threshold', float, 0.001) # Threshold for fitness-based heuristic

        base_mutation_rate = safe_convert('base_mutation_rate', float, mutation_rate) # Default based on fixed rate
        diversity_threshold_low = safe_convert('diversity_threshold_low', float, 0.1)
        mutation_rate_increase_factor = safe_convert('mutation_rate_increase_factor', float, 1.5)
        if mutation_rate_increase_factor < 1.0:
             logger.warning(f"mutation_rate_increase_factor ({mutation_rate_increase_factor}) < 1.0. Setting to 1.0")
             mutation_rate_increase_factor = 1.0

        # --- Other Config ---
        final_model_filename = f"evolved_{task_id}.pth"
        final_model_path = os.path.join(RESULT_DIR, final_model_filename)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[Task {task_id}] Using device: {device}")
        if device.type == 'cuda': logger.info(f"[Task {task_id}] CUDA Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"[Task {task_id}] Config Summary: Gens={generations}, Pop={population_size}, Elitism={elitism_count}")
        logger.info(f"[Task {task_id}] Operators: Select='{selection_strategy}', Cross='{crossover_operator}', Mut(W)='{mutation_operator}'")
        if use_dynamic_mutation_rate: logger.info(f"[Task {task_id}] Dynamic Rate(W) Enabled: H='{dynamic_mutation_heuristic}'")
        else: logger.info(f"[Task {task_id}] Fixed Rate(W): {mutation_rate:.3f}")
        logger.info(f"[Task {task_id}] Strengths: Weights={mutation_strength:.4f}, Hyperparams={hyperparam_mutation_strength:.4f}")
        logger.info(f"[Task {task_id}] Eval Config: {eval_config}")

        # --- Load Task Evaluation Function ---
        task_eval_func = load_task_eval_function(actual_eval_script_path)

        # --- Initialize Population (Hyperparams + Weights) ---
        self.update_state(state='PROGRESS', meta={'progress': 0.0, 'message': 'Initializing population...', 'fitness_history': [], 'avg_fitness_history': [], 'diversity_history': []})
        logger.info(f"[Task {task_id}] Initializing population ({population_size})...")

        # 1. Get initial/reference weights
        initial_weights = None
        try:
            # Instantiate model once to get weight structure/size
            # Use static kwargs + mid-range placeholder dynamic kwargs for structure definition
            placeholder_hparams = {k: (evolvable_hyperparams_config[k]['range'][0] + evolvable_hyperparams_config[k]['range'][1]) / 2 for k in hyperparam_keys if 'range' in evolvable_hyperparams_config[k]}
            ref_model = load_pytorch_model(model_definition_path, model_class, initial_weights_path, device, *model_args, **model_kwargs_static, **placeholder_hparams)
            initial_weights = flatten_weights(ref_model)
            if initial_weights_path and os.path.exists(initial_weights_path):
                logger.info(f"[Task {task_id}] Loaded initial weights. Weight vector size: {initial_weights.shape[0]}")
            else:
                logger.info(f"[Task {task_id}] Using default model weights. Weight vector size: {initial_weights.shape[0]}")
            del ref_model
        except Exception as e:
             error_msg = f"Failed to instantiate model '{model_class}' to get weight structure: {e}"
             logger.error(f"[Task {task_id}] {error_msg}", exc_info=True)
             raise ValueError(error_msg) from e

        if initial_weights.size == 0:
            raise ValueError("Initial weight vector size is zero. Cannot proceed.")

        # 3. Generate initial population chromosomes
        population: List[np.ndarray] = []
        init_rate_for_spread = initial_mutation_rate if use_dynamic_mutation_rate else mutation_rate # Use initial rate if dynamic for spread
        for i in range(population_size):
            # Generate random initial hyperparams
            hyperparam_values = np.zeros(num_hyperparams, dtype=np.float64)
            for idx, key in enumerate(hyperparam_keys):
                 h_config = evolvable_hyperparams_config[key]
                 h_min, h_max = h_config.get('range', [0.0, 1.0])
                 hyperparam_values[idx] = random.uniform(h_min, h_max)

            # Generate initial weights (copy or mutated copy)
            if i == 0: # First individual keeps initial weights
                 individual_weights = initial_weights.copy()
            else: # Mutate initial weights for diversity
                 if mutation_operator == "gaussian":
                     individual_weights = mutate_weights_gaussian(initial_weights, init_rate_for_spread, mutation_strength, num_hyperparams=0)
                 elif mutation_operator == "uniform_random":
                     individual_weights = mutate_weights_uniform_random(initial_weights, init_rate_for_spread, uniform_mutation_range, num_hyperparams=0)
                 else:
                     individual_weights = mutate_weights_gaussian(initial_weights, init_rate_for_spread, mutation_strength, num_hyperparams=0)

            chromosome = np.concatenate((hyperparam_values, individual_weights))
            population.append(chromosome)

        if device.type == 'cuda': torch.cuda.empty_cache()
        self.update_state(state='PROGRESS', meta={'progress': 0.01, 'message': 'Initialization complete. Starting generations...', 'fitness_history': [], 'avg_fitness_history': [], 'diversity_history': []})

    except Exception as init_err:
        error_msg = f"Initialization failed: {init_err}"
        logger.error(f"[Task {task_id}] {error_msg}", exc_info=True)
        self.update_state(state='FAILURE', meta={'message': error_msg, 'error': str(init_err)})
        raise

    # --- Evolution Loop ---
    best_fitness_overall = -float('inf')
    best_chromosome_overall = population[0].copy()
    fitness_history_overall = []
    avg_fitness_history_overall = []
    diversity_history_overall = []
    # No need for last_fitness_scores, calculate current avg directly

    try:
        for gen in range(generations):
            gen_num = gen + 1
            logger.info(f"[Task {task_id}] --- Generation {gen_num}/{generations} ---")
            gen_start_time = time.time()

            # --- Check for Halt Request ---
            if redis_client and halt_key: # Check if client is valid and key was constructed
                try:
                    # --- ADDED Debug Logs ---
                    logger.debug(f"[Task {task_id}] Checking Redis for halt key: {halt_key}")
                    exists = redis_client.exists(halt_key)
                    logger.debug(f"[Task {task_id}] Redis exists result for {halt_key}: {exists}")
                    # --- End Debug Logs ---
                    if exists: # exists returns number of keys found (1 if found)
                        logger.warning(f"[Task {task_id}] Halt flag detected. Stopping evolution.")
                        try:
                            deleted_count = redis_client.delete(halt_key)
                            logger.info(f"[Task {task_id}] Deleted halt key {halt_key} (Count: {deleted_count})")
                        except Exception as del_err:
                            logger.error(f"[Task {task_id}] Failed to delete halt flag {halt_key}: {del_err}")

                        halt_message = "Task halted by user request."
                        current_progress = (gen / generations) if generations > 0 else 0
                        best_hparams_halt = {}
                        if best_chromosome_overall is not None and num_hyperparams > 0:
                             try: best_hparams_halt = decode_hyperparameters(best_chromosome_overall[:num_hyperparams], hyperparam_keys, evolvable_hyperparams_config)
                             except Exception as decode_err: logger.error(f"Error decoding best hparams during halt: {decode_err}")
                        halt_meta = {
                            'message': halt_message, 'progress': current_progress,
                            'fitness_history': fitness_history_overall,
                            'avg_fitness_history': avg_fitness_history_overall,
                            'diversity_history': diversity_history_overall,
                            'best_hyperparameters': best_hparams_halt
                        }
                        self.update_state(state='HALTED', meta=halt_meta)
                        logger.info(f"[Task {task_id}] Task state set to HALTED.")
                        return {'message': halt_message, 'status': 'HALTED_BY_USER'}
                except redis.exceptions.ConnectionError as redis_conn_err:
                     logger.error(f"[Task {task_id}] Redis connection error during halt check: {redis_conn_err}. Halt check disabled.", exc_info=False)
                     redis_client = None # Disable further checks
                except Exception as check_err:
                     logger.error(f"[Task {task_id}] Unexpected error checking halt flag: {check_err}. Continuing run.", exc_info=True)
            # --- End Halt Check ---

            # 1. Evaluate Population
            try:
                fitness_scores = evaluate_population_step(
                    population, model_definition_path, model_class, task_eval_func, device,
                    model_args, model_kwargs_static, eval_config,
                    num_hyperparams=num_hyperparams,
                    evolvable_hyperparams_config=evolvable_hyperparams_config
                )
                fitness_scores = [float(f) if f is not None and np.isfinite(f) else -float('inf') for f in fitness_scores]

            except Exception as eval_err:
                logger.error(f"[Task {task_id}] Critical evaluation error in Gen {gen_num}: {eval_err}", exc_info=True)
                raise RuntimeError(f"Evaluation failed critically in Gen {gen_num}: {eval_err}") from eval_err

            # 2. Process Results & Calculate Metrics
            valid_scores = [f for f in fitness_scores if f > -float('inf')]
            if not valid_scores:
                 error_msg = f"All individuals failed evaluation in Gen {gen_num}."
                 logger.error(f"[Task {task_id}] {error_msg}")
                 raise RuntimeError(error_msg) # Stop if no valid scores

            max_fitness = np.max(valid_scores)
            avg_fitness = np.mean(valid_scores) # Current average fitness
            diversity = calculate_population_diversity(population, num_hyperparams)
            fitness_history_overall.append(float(max_fitness))
            avg_fitness_history_overall.append(float(avg_fitness))
            diversity_history_overall.append(float(diversity))

            # --- Determine Current Mutation Rate for NEXT Generation's Reproduction ---
            current_mutation_rate = mutation_rate # Default to fixed rate
            if use_dynamic_mutation_rate:
                try:
                    if dynamic_mutation_heuristic == 'time_decay':
                        # Calculate based on NEXT generation index (gen+1) or current (gen)?
                        # Let's use current `gen` relative to `generations` for rate applied in THIS reproduction phase
                        progress = gen / max(1, generations - 1)
                        rate = initial_mutation_rate - (initial_mutation_rate - final_mutation_rate) * progress
                        current_mutation_rate = max(final_mutation_rate, rate)

                    elif dynamic_mutation_heuristic == 'fitness_based':
                        if gen > 0 and avg_fitness_history_overall: # Need at least one previous average
                            prev_avg_fit = avg_fitness_history_overall[-2] if len(avg_fitness_history_overall) > 1 else avg_fitness_history_overall[-1] # Compare to previous or same if only 1 entry
                            improvement = avg_fitness - prev_avg_fit if prev_avg_fit > -np.inf else 0 # Use avg_fitness calculated above

                            if improvement < stagnation_threshold:
                                 current_mutation_rate = stagnation_mutation_rate # Use the increased rate
                                 logger.debug(f"Fitness stagnant (imp: {improvement:.4f}), using rate {current_mutation_rate:.3f}")
                            else:
                                 current_mutation_rate = normal_fitness_mutation_rate # Use the normal rate
                                 logger.debug(f"Fitness improving (imp: {improvement:.4f}), using rate {current_mutation_rate:.3f}")
                        else:
                             current_mutation_rate = initial_mutation_rate # Start with initial rate

                    elif dynamic_mutation_heuristic == 'diversity_based':
                        # Use diversity calculated above for the current population
                        if diversity < diversity_threshold_low:
                            current_mutation_rate = base_mutation_rate * mutation_rate_increase_factor
                            logger.debug(f"Low diversity ({diversity:.3f}), using rate {current_mutation_rate:.3f}")
                        else:
                            current_mutation_rate = base_mutation_rate
                            logger.debug(f"Sufficient diversity ({diversity:.3f}), using rate {current_mutation_rate:.3f}")
                    else:
                         logger.warning(f"Unknown dynamic heuristic '{dynamic_mutation_heuristic}', using fixed rate {mutation_rate:.3f}.")
                         current_mutation_rate = mutation_rate
                except Exception as dyn_err:
                    logger.error(f"Error calculating dynamic mutation rate: {dyn_err}. Using fallback fixed rate.", exc_info=True)
                    current_mutation_rate = mutation_rate

            # Log the rate determined for the REPRODUCTION step that follows
            logger.info(f"[Task {task_id}] Gen {gen_num} Stats: MaxFit={max_fitness:.4f}, AvgFit={avg_fitness:.4f}, Div(W)={diversity:.4f}, RateForNextRepro={current_mutation_rate:.3f}")


            # Update overall best chromosome
            best_idx_current = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[best_idx_current]
            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                if best_idx_current < len(population):
                     best_chromosome_overall = population[best_idx_current].copy()
                     logger.info(f"[Task {task_id}] *** New best overall fitness: {best_fitness_overall:.4f} ***")
                     try: # Safely log best hyperparams
                         best_hparams_decoded = decode_hyperparameters(best_chromosome_overall[:num_hyperparams], hyperparam_keys, evolvable_hyperparams_config)
                         logger.info(f"[Task {task_id}] Best Hyperparameters so far: {best_hparams_decoded}")
                     except Exception as decode_err: logger.error(f"Err logging best hparams: {decode_err}")
                else:
                     logger.warning(f"[Task {task_id}] Best index {best_idx_current} out of bounds {len(population)}.")

            # --- Update Task State ---
            current_progress = gen_num / generations
            progress_message = f'Gen {gen_num}/{generations} | Best Fit: {best_fitness_overall:.4f} | Avg Fit: {avg_fitness:.4f}'
            self.update_state(
                state='PROGRESS', meta={ 'progress': current_progress, 'message': progress_message, 'fitness_history': fitness_history_overall, 'avg_fitness_history': avg_fitness_history_overall, 'diversity_history': diversity_history_overall }
            )

            # --- 3. SELECTION ---
            num_parents_to_select = population_size
            parents = []
            logger.debug(f"[Task {task_id}] Selecting parents using '{selection_strategy}'...")
            selectable_indices = [i for i, f in enumerate(fitness_scores) if f > -float('inf')]
            selectable_population_chromosomes = [population[i] for i in selectable_indices]
            selectable_fitness = [fitness_scores[i] for i in selectable_indices]
            # No need to check selectable_indices again, handled by RuntimeError above

            if selection_strategy == "tournament":
                parents = select_parents_tournament(selectable_population_chromosomes, selectable_fitness, num_parents_to_select, tournament_size=tournament_size)
            elif selection_strategy == "roulette":
                parents = select_parents_roulette(selectable_population_chromosomes, selectable_fitness, num_parents_to_select)
            else:
                 logger.warning(f"Unsupported selection strategy '{selection_strategy}', defaulting to tournament.")
                 parents = select_parents_tournament(selectable_population_chromosomes, selectable_fitness, num_parents_to_select, tournament_size=tournament_size)

            if not parents or len(parents) < 2:
                logger.warning(f"[Task {task_id}] Parent selection yielded < 2 parents ({len(parents)}). Re-populating from best.")
                next_population = [best_chromosome_overall.copy()]
                for _ in range(population_size - 1):
                    # Mutate based on rate calculated for *this* reproduction step
                    mutated_best_h = mutate_hyperparams_gaussian(best_chromosome_overall, hyperparam_mutation_strength, num_hyperparams)
                    mutated_best_w = mutate_weights_gaussian(mutated_best_h, current_mutation_rate, mutation_strength, num_hyperparams)
                    next_population.append(mutated_best_w)
                population = next_population[:population_size]
                continue

            # --- 4. REPRODUCTION (Elitism + Crossover + Mutation) ---
            next_population = []
            logger.debug(f"[Task {task_id}] Reproducing (Elitism={elitism_count}, Cross='{crossover_operator}', Mut(W)='{mutation_operator}')...")

            # Elitism
            if elitism_count > 0:
                 elite_indices = np.argsort(fitness_scores)[::-1][:elitism_count]
                 for elite_idx in elite_indices:
                      if len(next_population) < population_size and elite_idx < len(population):
                           next_population.append(population[elite_idx].copy())

            # Offspring Generation
            while len(next_population) < population_size:
                try:
                    if len(parents) < 2: p1_chrom, p2_chrom = parents[0], parents[0]
                    else: p1_chrom, p2_chrom = random.sample(parents, 2)

                    # Crossover
                    child1_chrom, child2_chrom = None, None
                    if crossover_operator == "one_point": child1_chrom, child2_chrom = crossover_one_point(p1_chrom, p2_chrom, num_hyperparams)
                    elif crossover_operator == "uniform": child1_chrom, child2_chrom = crossover_uniform(p1_chrom, p2_chrom, num_hyperparams, crossover_prob=uniform_crossover_prob)
                    elif crossover_operator == "average": child1_chrom, child2_chrom = crossover_average(p1_chrom, p2_chrom, num_hyperparams)
                    else: child1_chrom, child2_chrom = crossover_one_point(p1_chrom, p2_chrom, num_hyperparams) # Default

                    # Mutation (using rate determined before reproduction)
                    # 1. Mutate Hyperparams
                    mutated_child1_h = mutate_hyperparams_gaussian(child1_chrom, hyperparam_mutation_strength, num_hyperparams)
                    mutated_child2_h = mutate_hyperparams_gaussian(child2_chrom, hyperparam_mutation_strength, num_hyperparams)
                    # 2. Mutate Weights
                    mutated_child1_w, mutated_child2_w = None, None
                    if mutation_operator == "gaussian":
                        mutated_child1_w = mutate_weights_gaussian(mutated_child1_h, current_mutation_rate, mutation_strength, num_hyperparams)
                        mutated_child2_w = mutate_weights_gaussian(mutated_child2_h, current_mutation_rate, mutation_strength, num_hyperparams)
                    elif mutation_operator == "uniform_random":
                        mutated_child1_w = mutate_weights_uniform_random(mutated_child1_h, current_mutation_rate, uniform_mutation_range, num_hyperparams)
                        mutated_child2_w = mutate_weights_uniform_random(mutated_child2_h, current_mutation_rate, uniform_mutation_range, num_hyperparams)
                    else: # Default
                        mutated_child1_w = mutate_weights_gaussian(mutated_child1_h, current_mutation_rate, mutation_strength, num_hyperparams)
                        mutated_child2_w = mutate_weights_gaussian(mutated_child2_h, current_mutation_rate, mutation_strength, num_hyperparams)

                    if len(next_population) < population_size: next_population.append(mutated_child1_w)
                    if len(next_population) < population_size: next_population.append(mutated_child2_w)

                except Exception as repr_err:
                     logger.warning(f"[Task {task_id}] Reproduction error: {repr_err}. Substituting.", exc_info=True)
                     if len(next_population) < population_size:
                          substitute_parent_chrom = random.choice(parents) if parents else best_chromosome_overall
                          mut_sub_h = mutate_hyperparams_gaussian(substitute_parent_chrom, hyperparam_mutation_strength, num_hyperparams)
                          mut_sub_w = mutate_weights_gaussian(mut_sub_h, current_mutation_rate, mutation_strength, num_hyperparams) # Use current rate
                          next_population.append(mut_sub_w)

            population = next_population[:population_size]
            gen_time = time.time() - gen_start_time
            logger.info(f"[Task {task_id}] Gen {gen_num} finished in {gen_time:.2f}s")

        # --- Evolution Finished ---
        if best_fitness_overall == -float('inf'):
             raise RuntimeError("Evolution completed, but no valid individuals found.")
        logger.info(f"[Task {task_id}] Evolution finished. Best Fitness: {best_fitness_overall:.4f}")

        # --- Save Final Best Model ---
        logger.info(f"[Task {task_id}] Saving best model to {final_model_path}...")
        best_hyperparams = decode_hyperparameters(best_chromosome_overall[:num_hyperparams], hyperparam_keys, evolvable_hyperparams_config)
        logger.info(f"[Task {task_id}] Decoded Best Hyperparameters: {best_hyperparams}")
        final_best_model = load_pytorch_model(
            model_definition_path, model_class, None, device,
            *model_args, **model_kwargs_static, **best_hyperparams
        )
        best_weights = best_chromosome_overall[num_hyperparams:]
        load_weights_from_flat(final_best_model, best_weights)
        torch.save(final_best_model.state_dict(), final_model_path)
        logger.info(f"[Task {task_id}] Best model saved successfully.")

        # --- Clean up uploaded files ---
        logger.info(f"[Task {task_id}] Cleaning up temporary uploaded files...")
        files_to_remove = [model_definition_path, task_evaluation_path, initial_weights_path]
        for f_path in files_to_remove:
             if f_path and os.path.exists(f_path):
                  try: os.remove(f_path); logger.debug(f"Removed: {f_path}")
                  except OSError as rm_err: logger.warning(f"Could not remove {f_path}: {rm_err}")

        # --- Return final result data ---
        success_message = f'Evolution completed. Best fitness: {best_fitness_overall:.4f}.'
        final_result = {
            'message': success_message,
            'final_model_path': final_model_path,
            'best_fitness': float(best_fitness_overall),
            'best_hyperparameters': best_hyperparams,
            'fitness_history': fitness_history_overall,
            'avg_fitness_history': avg_fitness_history_overall,
            'diversity_history': diversity_history_overall
        }
        logger.info(f"[Task {task_id}] Task successful.")
        self.update_state(state='SUCCESS', meta=final_result)
        return final_result

    # --- Exception Handling (Added SoftTimeLimitExceeded) ---
    except SoftTimeLimitExceeded: # Catch Celery's exception if SIGUSR1 was used (alternative)
        logger.warning(f"[Task {task_id}] Soft time limit exceeded. Task halting.")
        # Update state to HALTED or a custom state
        halt_message = "Task halted due to time limit or signal."
        halt_meta = { 'message': halt_message, 'fitness_history': fitness_history_overall, 'avg_fitness_history': avg_fitness_history_overall, 'diversity_history': diversity_history_overall, 'best_hyperparameters': decode_hyperparameters(best_chromosome_overall[:num_hyperparams], hyperparam_keys, evolvable_hyperparams_config) if num_hyperparams > 0 else {} }
        self.update_state(state='HALTED', meta=halt_meta)
        # Return a result
        return {'message': halt_message, 'status': 'HALTED_BY_SIGNAL'}
    except Exception as e:
        error_message = f'Evolution task failed: {str(e)}'
        logger.error(f"[Task {task_id}] {error_message}", exc_info=True)
        # ... (existing failure handling remains the same) ...
        best_hyperparams_fail = {};
        if num_hyperparams > 0 and len(best_chromosome_overall) >= num_hyperparams:
            try: best_hyperparams_fail = decode_hyperparameters(best_chromosome_overall[:num_hyperparams], hyperparam_keys, evolvable_hyperparams_config)
            except Exception as decode_err: logger.error(f"Error decoding best hparams on failure: {decode_err}")
        meta_fail = { 'message': error_message, 'error': str(e), 'best_hyperparameters': best_hyperparams_fail, 'fitness_history': fitness_history_overall, 'avg_fitness_history': avg_fitness_history_overall, 'diversity_history': diversity_history_overall }
        self.update_state(state='FAILURE', meta=meta_fail)
        raise

    finally:
        # --- NEW: Close Redis connection ---
        if redis_client:
            try:
                redis_client.close()
                logger.info(f"[Task {task_id}] Closed Redis connection.")
            except Exception as close_err:
                logger.error(f"[Task {task_id}] Error closing Redis connection: {close_err}")
        # --- End Redis Close ---