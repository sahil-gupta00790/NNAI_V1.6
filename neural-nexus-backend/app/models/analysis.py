from pydantic import BaseModel

class GaAnalysisRequest(BaseModel):
    fitness_history: list[float]
    avg_fitness_history: list[float] | None = None
    diversity_history: list[float] | None = None
    generations: int
    population_size: int
    mutation_rate: float | None = None
    mutation_strength: float | None = None

class GaAnalysisResponse(BaseModel):
    analysis_text: str
