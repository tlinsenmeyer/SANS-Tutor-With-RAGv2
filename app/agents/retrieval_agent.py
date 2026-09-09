# app/agents/retrieval_agent.py

from typing import Dict, Any, List
from app.graph.state import TutorState
from app.llm.vllm_client import VLLMClient
from app.data.qdrant import search_qdrant
from app.data.neo4j import search_neo4j

llm = VLLMClient()

async def retrieval_agent(state: TutorState) -> TutorState:
    """
    Retrieval agent:
    - embeds the user's question
    - queries Qdrant (semantic chunks)
    - queries Neo4j (graph nodes)
    - merges context into a single string
    """

    if not state.question:
        raise ValueError("State.question is empty — cannot perform retrieval.")

    # 1. Embed the question via vLLM
    query_vector = await llm.embed(state.question)

    # 2. Query Qdrant (semantic retrieval)
    qdrant_results = search_qdrant(
        collection="sans_docs",
        query_vector=query_vector,
        limit=5,
    )
    state.retrieved_chunks = qdrant_results

    # 3. Query Neo4j (graph retrieval)
    graph_results = search_neo4j(
        query=state.question,
        limit=10,
    )
    state.graph_nodes = graph_results

    # 4. Build merged context
    chunks_text = "\n\n".join(
        f"[Chunk {c.get('chunk_index')}] {c.get('text')}"
        for c in qdrant_results
    )

    graph_text = "\n\n".join(
        f"[{g.get('labels')}] {g.get('properties').get('summary', g.get('properties').get('name', ''))}"
        for g in graph_results
    )

    merged = []
    if chunks_text:
        merged.append("=== Semantic Context (Qdrant) ===\n" + chunks_text)
    if graph_text:
        merged.append("=== Graph Context (Neo4j) ===\n" + graph_text)

    state.merged_context = "\n\n".join(merged)

    return state
