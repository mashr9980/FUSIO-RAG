# models.py
from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel, Field
from typing import Optional

class GenerationRequest(BaseModel):
    prompt: str
    agent_id: Optional[str] = None
    model_name: str = "gpt-4o"
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    knowledge_base_flag: bool = False

class GenerationResponse(BaseModel):
    status: str
    generated_text: str
    time: float
    error: Optional[str] = None

class DocumentUpload(BaseModel):
    url: str

class DocumentResponse(BaseModel):
    document_id: str
    message: str

class ChatRequest(BaseModel):
    document_id: str
    question: str

class DocumentStatus:
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"