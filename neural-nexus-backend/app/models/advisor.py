# app/models/advisor.py
from pydantic import BaseModel

class AdvisorQuery(BaseModel):
    query: str
    history: list[dict] | None = None # Optional conversation history

class AdvisorResponse(BaseModel):
    response: str
    sources: list[str] | None = None # Optional sources used by RAG

# NO OTHER "from app.models.advisor import ..." lines should be here
