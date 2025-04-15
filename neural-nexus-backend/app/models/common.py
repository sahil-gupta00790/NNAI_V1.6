# app/models/common.py
from pydantic import BaseModel

class TaskResponse(BaseModel):
    task_id: str
    status: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float | None = None
    result: dict | None = None # Or specific result model
    message: str | None = None
