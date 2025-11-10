from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    source_document: Optional[str] = None
    confidence: Optional[float] = None
    chat_id: Optional[str] = None

class ChatMessage(BaseModel):
    query: str
    response: str
    timestamp: datetime

class SearchResult(BaseModel):
    content: str
    confidence: float
    document_name: str
    modified_date: str
    is_most_recent: bool