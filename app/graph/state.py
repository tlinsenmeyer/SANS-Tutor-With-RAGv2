from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class TutorState(BaseModel):
    # User input
    question: Optional[str] = None
    user_answer: Optional[str] = None

    # Retrieval agent output
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None   # Qdrant
    graph_nodes: Optional[List[Dict[str, Any]]] = None         # Neo4j
    merged_context: Optional[str] = None                       # Combined context

    # Tutor agent output
    tutor_response: Optional[str] = None
    tutor_reasoning: Optional[str] = None

    # Verification agent output
    verified_answer: Optional[str] = None
    corrections: Optional[List[str]] = None
    additions: Optional[List[str]] = None
    modern_updates: Optional[List[str]] = None
    final_answer: Optional[str] = None
    confidence: Optional[float] = None

    # Metadata for orchestration, logging, debugging
    metadata: Dict[str, Any] = {}
