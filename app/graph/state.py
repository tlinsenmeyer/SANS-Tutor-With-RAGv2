from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class TutorState(BaseModel):
    # User input
    question: Optional[str] = None
    user_answer: Optional[str] = None

    # Retrieval agent output
    retrieved_docs: Optional[List[str]] = None

    # Tutor agent output
    tutor_response: Optional[str] = None

    # Grading agent output
    grade: Optional[str] = None
    score: Optional[float] = None

    # Metadata for orchestration, logging, debugging
    metadata: Dict[str, Any] = {}
