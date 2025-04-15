# app/models/evolver.py
from pydantic import BaseModel, Field
from typing import Any # Import Any for nested dicts

class EvolverConfig(BaseModel):
    # --- REQUIRED ---
    model_class: str = Field(..., description="Name of the model class in the definition file.") # Make mandatory
    generations: int = Field(..., gt=0, description="Number of generations to run.")
    population_size: int = Field(..., gt=1, description="Number of individuals in the population.")

    # --- ADD ALL OTHER EXPECTED FIELDS ---
    selection_strategy: str | None = Field(default="tournament", description="Parent selection strategy (e.g., tournament, roulette)")
    crossover_operator: str | None = Field(default="one_point", description="Crossover operator (e.g., one_point, uniform)")
    mutation_operator: str | None = Field(default="gaussian", description="Mutation operator (e.g., gaussian, uniform_random)")
    elitism_count: int | None = Field(default=1, ge=0, description="Number of best individuals to carry over.")
    mutation_rate: float | None = Field(default=0.1, ge=0.0, le=1.0, description="Probability of mutation.")
    mutation_strength: float | None = Field(default=0.05, description="Strength/std deviation for mutation (esp. Gaussian).")
    tournament_size: int | None = Field(default=3, gt=1, description="Size of tournament for selection.")
    uniform_crossover_prob: float | None = Field(default=0.5, ge=0.0, le=1.0, description="Probability for uniform crossover swap.")
    uniform_mutation_range: tuple[float, float] | None = Field(default=(-1.0, 1.0), description="Range for uniform random mutation.")
    init_mutation_rate: float | None = Field(default=None, ge=0.0, le=1.0, description="Initial mutation rate (defaults to mutation_rate if None).")
    init_mutation_strength: float | None = Field(default=None, description="Initial mutation strength (defaults to mutation_strength if None).")
    model_args: list | None = Field(default=None, description="Positional arguments for model constructor.")
    model_kwargs: dict | None = Field(default=None, description="Keyword arguments for model constructor.")
    eval_config: dict[str, Any] | None = Field(default=None, description="Configuration dictionary for the evaluation script.")

    # Allow extra fields if needed, but defining them is better
    model_config = {
        "extra": "ignore" # Or "allow" if you truly want unexpected fields
    }
