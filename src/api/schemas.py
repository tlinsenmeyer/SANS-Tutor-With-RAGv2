from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="The cybersecurity question for the SANS tutor.")
    top_k: int = Field(5, description="Number of vector chunks to retrieve from Qdrant.")
    include_graph: bool = Field(True, description="Whether to include Neo4j graph relationships in context.")

class SourceReference(BaseModel):
    chunk_id: str
    page_number: int
    source_file: str
    score: float

class QueryResponse(BaseModel):
    trace_id: str
    answer: str
    sources: List[SourceReference]
    graph_context: List[Dict[str, Any]] = []
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    qdrant: str
    neo4j: str
    vllm: str