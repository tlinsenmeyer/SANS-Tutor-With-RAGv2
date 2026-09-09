# app/data/qdrant.py

from qdrant_client import QdrantClient
from typing import List, Dict, Any

qdrant = QdrantClient(
    host="192.168.1.87",
    port=6333,
)

def search_qdrant(collection: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Perform a vector search against Qdrant and return structured chunk objects.
    """

    results = qdrant.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit
    )

    structured = []
    for hit in results:
        structured.append({
            "id": hit.id,
            "score": hit.score,
            "text": hit.payload.get("text", ""),
            "source": hit.payload.get("source", None),
            "chunk_index": hit.payload.get("chunk_index", None),
            "metadata": hit.payload,
        })

    return structured

