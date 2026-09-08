from app.graph.state import TutorState
from app.data.qdrant import search_qdrant
from typing import List

# Placeholder embedding function — you will replace this with your real embedder
def embed_text(text: str) -> List[float]:
    # TODO: Replace with your embedding model (e.g., sentence-transformers, instructor-xl, etc.)
    return [0.1] * 768  # placeholder vector

def retrieval_agent(state: TutorState) -> TutorState:
    """
    Retrieval agent:
    - embeds the user's question
    - queries Qdrant
    - stores retrieved docs in state
    """
    if not state.question:
        raise ValueError("State.question is empty — cannot perform retrieval.")

    # Embed the question
    query_vector = embed_text(state.question)

    # Query Qdrant
    docs = search_qdrant(
        collection="sans_docs",  # your collection name
        query_vector=query_vector,
        limit=5
    )

    # Update state
    state.retrieved_docs = docs
    return state
