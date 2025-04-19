# app/models/analysis.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any # Import necessary types

class GaAnalysisRequest(BaseModel):
    # --- Core Data ---
    fitness_history: List[float] = Field(..., description="List of max fitness values per generation.")
    avg_fitness_history: Optional[List[float]] = Field(None, description="Optional list of average fitness values per generation.")
    diversity_history: Optional[List[float]] = Field(None, description="Optional list of population diversity values per generation.")

    # --- GA Configuration ---
    generations: int = Field(..., description="Total number of generations run.")
    population_size: int = Field(..., description="Number of individuals per generation.")

    # --- Weight Mutation Config ---
    # Mutation Strength (applies always)
    mutation_strength: Optional[float] = Field(None, description="Magnitude of weight mutation (e.g., std dev).")

    # Fixed Rate (Only if dynamic is False)
    mutation_rate: Optional[float] = Field(None, description="Fixed mutation rate (used if dynamic rate is False).")

    # --- Dynamic Weight Mutation Rate Config ---
    use_dynamic_mutation_rate: bool = Field(False, description="Whether dynamic mutation rate was used for weights.")
    dynamic_mutation_heuristic: Optional[str] = Field(None, description="Heuristic used if dynamic rate is True (e.g., 'time_decay', 'fitness_based', 'diversity_based').")

    # Dynamic Rate - Time Decay Params
    initial_mutation_rate: Optional[float] = Field(None, description="Initial weight mutation rate for time decay.")
    final_mutation_rate: Optional[float] = Field(None, description="Final weight mutation rate for time decay.")

    # Dynamic Rate - Fitness Based Params
    normal_fitness_mutation_rate: Optional[float] = Field(None, description="Weight mutation rate used when fitness is improving.")
    stagnation_mutation_rate: Optional[float] = Field(None, description="Weight mutation rate used when fitness stagnates.")
    stagnation_threshold: Optional[float] = Field(None, description="Threshold for fitness stagnation.")

    # Dynamic Rate - Diversity Based Params
    base_mutation_rate: Optional[float] = Field(None, description="Base weight mutation rate for diversity-based heuristic.")
    diversity_threshold_low: Optional[float] = Field(None, description="Diversity threshold to trigger rate increase.")
    mutation_rate_increase_factor: Optional[float] = Field(None, description="Factor to increase rate by when diversity is low.")

    # --- Hyperparameter Optimization Config & Results ---
    # Describes which hyperparameters were evolved (keys) and their config (values)
    evolvable_hyperparams: Optional[Dict[str, Any]] = Field(None, description="Configuration of hyperparameters that were evolved (e.g., {'param_name': {'type':'int', 'range':[min, max]}}).")
    # The best set found by the GA
    best_hyperparameters: Optional[Dict[str, Any]] = Field(None, description="The best set of hyperparameters found by the evolution.")

    class Config:
        # Pydantic v2 config class
        # If using Pydantic v1, use 'Config' without inheriting 'BaseModel'
        # Example for Pydantic v2:
        # model_config = {
        #     "extra": "ignore" # Ignore extra fields from request if needed
        # }

        # Example for Pydantic v1:
        # extra = 'ignore' # Ignore extra fields from request if needed
        pass


class GaAnalysisResponse(BaseModel):
    analysis_text: str

