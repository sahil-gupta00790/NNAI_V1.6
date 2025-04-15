# app/models/evolver.py
from pydantic import BaseModel, Field

class EvolverConfig(BaseModel):
    generations: int = Field(..., gt=0)
    population_size: int = Field(..., gt=0)
    # Add other GA parameters (mutation rate, crossover type, etc.)

# Note: File uploads are handled directly in the endpoint using FastAPI's UploadFile
# The config might be sent as JSON alongside the files.
