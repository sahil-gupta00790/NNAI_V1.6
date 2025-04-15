# app/utils/evolution_helpers.py
# Contains functions adapted from final_evolve_script.py for use in the Celery task.

import torch
import torch.nn as nn
import numpy as np
import os
import importlib.util
import time
import random

# --- Utility Functions (Copied/Adapted from final_evolve_script.py [1]) ---

def flatten_weights(model):
    """ Flattens all model parameters into a single numpy vector. """
    try:
        weights = []
        for param in model.parameters():
            if param.requires_grad:
                weights.append(param.data.cpu().numpy().flatten())
        if not weights:
            raise ValueError("No trainable parameters found in the model to flatten.")
        return np.concatenate(weights)
    except Exception as e:
        print(f"Error during weight flattening: {e}")
        raise

def load_weights_from_flat(model, flat_weights):
    """ Loads flattened weights back into a model instance. """
    try:
        offset = 0
        if not isinstance(flat_weights, np.ndarray):
            flat_weights = np.array(flat_weights)
        flat_weights_tensor = torch.from_numpy(flat_weights).float()
        model_device = next(model.parameters()).device
        for param in model.parameters():
            if param.requires_grad:
                numel = param.numel()
                param_shape = param.size()
                if offset + numel > len(flat_weights_tensor):
                    raise ValueError(f"Shape mismatch: Not enough data in flat_weights to fill parameter {param.shape} (offset {offset}, numel {numel}, flat_weights len {len(flat_weights_tensor)})")
                param_slice = flat_weights_tensor[offset:offset + numel].view(param_shape).to(model_device)
                param.data.copy_(param_slice)
                offset += numel
        if offset != len(flat_weights_tensor):
            print(f"Warning: Size mismatch after loading weights. Offset {offset} != flat_weights length {len(flat_weights_tensor)}. Check model definition correspondence.")
    except Exception as e:
        print(f"Error loading weights from flat vector: {e}")
        raise

def load_pytorch_model(model_definition_path, class_name, state_dict_path, device, *model_args, **model_kwargs):
    """ Loads the model class, instantiates it, and loads the state_dict. """
    try:
        spec = importlib.util.spec_from_file_location("model_module", model_definition_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module at {model_definition_path}")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        if not hasattr(model_module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in {model_definition_path}")
        ModelClass = getattr(model_module, class_name)
        print(f"Instantiating model '{class_name}' with args: {model_args}, kwargs: {model_kwargs}")
        model = ModelClass(*model_args, **model_kwargs)
        model.to(device)
        if state_dict_path and os.path.exists(state_dict_path):
            print(f"Loading state_dict from: {state_dict_path}")
            try:
                state_dict = torch.load(state_dict_path, map_location=device)
                model.load_state_dict(state_dict)
                print("State_dict loaded successfully.")
            except Exception as load_err:
                print(f"Error loading state_dict: {load_err}. Check architecture.")
                raise
        elif state_dict_path:
            print(f"Warning: state_dict path '{state_dict_path}' provided but not found. Using initial model weights.")
        else:
            print("No state_dict path provided. Using initial model weights.")
        model.eval()
        return model
    except Exception as e:
        print(f"Error in load_pytorch_model: {e}")
        raise

def load_task_eval_function(task_module_path):
    """ Loads the fitness evaluation function from a specified file. """
    try:
        spec = importlib.util.spec_from_file_location("task_module", task_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module at {task_module_path}")
        task_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_module)
        if not hasattr(task_module, 'evaluate_network_on_task'):
            raise AttributeError(f"Function 'evaluate_network_on_task(model_instance, device)' not found in {task_module_path}")
        return getattr(task_module, 'evaluate_network_on_task')
    except Exception as e:
        print(f"Error loading task evaluation function: {e}")
        raise

def evaluate_population_step(population_weights, model_definition_path, class_name, task_eval_func, device, model_args, model_kwargs):
    """ Evaluates the fitness of each individual in the population for one generation. """
    fitness_scores = []
    num_individuals = len(population_weights)
    evaluation_times = []

    print(f"Evaluating {num_individuals} individuals...")
    for i, flat_weights in enumerate(population_weights):
        individual_start_time = time.time()
        current_model = None
        # torch.compile is generally disabled here based on script source [1]
        # compiled_model_obj = None
        # compile_attempted = False
        # compile_successful = False

        # Minimal logging per individual to reduce noise
        # print(f"\rEvaluating individual {i+1}/{num_individuals}...", end="")

        try:
            # 1. Create & Load Model
            current_model = load_pytorch_model(model_definition_path, class_name, None, device, *model_args, **model_kwargs)
            load_weights_from_flat(current_model, flat_weights)
            current_model.to(device)
            current_model.eval()

            model_to_evaluate = current_model # Assuming no compile

            # 2. Call Evaluation Function
            fitness = task_eval_func(model_to_evaluate, device)

            # 3. Process Fitness
            if not isinstance(fitness, (float, int)):
                print(f"\nWarning: Individual {i+1} fitness func returned non-numeric ({type(fitness)}). Setting -inf.")
                fitness = -float('inf')
            fitness_scores.append(float(fitness))

        except Exception as e:
            print(f"\nError evaluating individual {i+1}: {e}")
            fitness_scores.append(-float('inf')) # Assign low fitness on error
        finally:
            # 4. Cleanup
            del model_to_evaluate # Might be original or compiled
            # Careful cleanup if compile was used
            # if current_model is not None and compiled_model_obj is not None and current_model is not compiled_model_obj: del current_model
            # elif current_model is not None and compiled_model_obj is None: del current_model
            # compiled_model_obj = None
            if current_model is not None:
                del current_model # Delete the uncompiled model instance

            if device.type == 'cuda':
                torch.cuda.empty_cache()
            eval_time = time.time() - individual_start_time
            evaluation_times.append(eval_time)

    # print() # Newline after progress
    avg_eval_time = np.mean(evaluation_times) if evaluation_times else 0
    print(f"Finished population evaluation. Avg time/individual: {avg_eval_time:.3f}s")
    return fitness_scores


def select_parents(population_weights, fitness_scores, num_parents):
    """ Selects parents using tournament selection. """
    parents = []
    population_size = len(population_weights)
    if population_size == 0: return []
    tournament_size = max(2, min(population_size, 5))
    valid_indices = [i for i, f in enumerate(fitness_scores) if f > -float('inf')]

    if not valid_indices:
        print("Warning: All individuals failed evaluation. Cannot select parents.")
        return []

    for _ in range(num_parents):
        # Handle case where valid_indices < tournament_size
        current_tournament_size = min(len(valid_indices), tournament_size)
        if current_tournament_size == 0: # Should not happen if valid_indices is checked, but safety first
             print("Error: No valid individuals to select from in tournament.")
             continue
        tournament_candidate_indices = random.sample(valid_indices, current_tournament_size)

        best_fitness_in_tournament = -float('inf')
        winner_index_in_population = -1

        for idx in tournament_candidate_indices:
            if fitness_scores[idx] > best_fitness_in_tournament:
                best_fitness_in_tournament = fitness_scores[idx]
                winner_index_in_population = idx

        if winner_index_in_population != -1:
            parents.append(population_weights[winner_index_in_population])
        elif valid_indices: # Fallback if tournament somehow failed but valid individuals exist
            print("Warning: Could not select a winner in tournament. Picking random valid individual.")
            parents.append(population_weights[random.choice(valid_indices)])
        # else: # No valid individuals left at all

    return parents


def crossover(parent1, parent2):
    """ Simple average crossover for weight vectors. """
    p1 = np.array(parent1)
    p2 = np.array(parent2)
    if p1.shape != p2.shape:
        raise ValueError(f"Parent shapes do not match for crossover: {p1.shape} vs {p2.shape}")
    child = (p1 + p2) / 2.0
    return child

def mutate(weights, mutation_rate, mutation_strength):
    """ Adds Gaussian noise to a fraction of weights based on mutation rate. """
    if mutation_rate <= 0 or mutation_strength <= 0:
        return weights
    mutated_weights = weights.copy()
    num_weights_to_mutate = int(len(weights) * mutation_rate)
    if num_weights_to_mutate == 0 and mutation_rate > 0 and len(weights) > 0:
        num_weights_to_mutate = 1 # Ensure at least one mutation if rate > 0

    if num_weights_to_mutate > 0:
        indices_to_mutate = np.random.choice(len(weights), num_weights_to_mutate, replace=False)
        noise = np.random.normal(0, mutation_strength, size=num_weights_to_mutate)
        mutated_weights[indices_to_mutate] += noise.astype(mutated_weights.dtype)
    return mutated_weights

# --- End of Helper Functions ---
