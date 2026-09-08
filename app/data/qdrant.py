from qdrant_client import QdrantClient
from typing import List

# Qdrant connection
qdrant = QdrantClient(
    host="192.168.1.87",
    port=6333,  # REST API port
)

def search_qdrant(collection: str, query_vector: List[float], limit: int = 5):
    """
    Perform a vector search against Qdrant and return the top-k payloads.
    """
    results = qdrant.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit
    )

    # Extract text payloads
    docs = [hit.payload.get("text", "") for hit in results]
    return docs
