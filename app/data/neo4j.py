# app/data/neo4j.py

from neo4j import GraphDatabase
from typing import List, Dict, Any

# Neo4j connection
driver = GraphDatabase.driver(
    "bolt://192.168.1.87:7687",
    auth=("neo4j", "your_password_here")
)

def search_neo4j(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search Neo4j for nodes relevant to the user's question.
    Uses a simple full-text index (you will create this during ingestion).
    """

    cypher = """
    CALL db.index.fulltext.queryNodes('manualIndex', $query) YIELD node, score
    RETURN node, score
    LIMIT $limit
    """

    with driver.session() as session:
        results = session.run(cypher, query=query, limit=limit)

        structured = []
        for record in results:
            node = record["node"]
            score = record["score"]

            structured.append({
                "id": node.id,
                "labels": list(node.labels),
                "properties": dict(node),
                "score": score,
                "summary": node.get("summary", None),
            })

        return structured
