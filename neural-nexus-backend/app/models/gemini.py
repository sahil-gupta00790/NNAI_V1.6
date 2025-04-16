# app/models/gemini.py
from pydantic import BaseModel
from typing import List, Dict, Any

# Structure matching Gemini API's history format
class GeminiHistoryItem(BaseModel):
    role: str # 'user' or 'model'
    parts: List[Dict[str, Any]] # Usually [{'text': 'message content'}]

class GeminiChatRequest(BaseModel):
    query: str
    history: List[GeminiHistoryItem] | None = None # Optional history

class GeminiChatResponse(BaseModel):
    reply: str
