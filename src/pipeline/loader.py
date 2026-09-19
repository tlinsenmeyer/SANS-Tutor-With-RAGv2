import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridLoader:
    """Synchronizes embedded chunks into Qdrant (vectors) and Neo4j (graph nodes)."""
    
    def __init__(
        self, 
        qdrant_url: str = "http://192.168.100.87:6333",
        qdrant_collection: str = "sans_vectors",
        neo4j_uri: str = "bolt://192.168.100.72:7687",
        neo4j_auth: tuple = ("neo4j", "password")
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = qdrant_collection
        
        logger.info(f"Connecting to Qdrant at {self.qdrant_url}...")
        self.qdrant_client = QdrantClient(url=self.qdrant_url)
        
        logger.info(f"Connecting to Neo4j at {neo4j_uri}...")
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        
        self._init_qdrant_collection()

    def _init_qdrant_collection(self, vector_size: int = 384):
        """Ensures the Qdrant collection exists with proper cosine distance indexing."""
        collections = self.qdrant_client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating Qdrant collection '{self.collection_name}'...")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,  # Default dimension for all-MiniLM-L6-v2
                    distance=models.Distance.COSINE
                )
            )
        else:
            logger.info(f"Qdrant collection '{self.collection_name}' already exists.")

    def load_to_qdrant(self, embedded_chunks: List[Dict[str, Any]]):
        """Batch upserts vector embeddings and payloads into Qdrant."""
        logger.info(f"Upserting {len(embedded_chunks)} points to Qdrant collection '{self.collection_name}'...")
        
        points = []
        for idx, chunk in enumerate(embedded_chunks):
            points.append(
                models.PointStruct(
                    id=idx,  # In production, use UUIDs mapped from chunk_id hashes
                    vector=chunk["vector"],
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "page_number": chunk["metadata"]["page_number"],
                        "source_file": chunk["metadata"]["source_file"]
                    }
                )
            )
            
        # Upsert in batches of 100 to optimize network throughput
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        logger.info("Successfully completed Qdrant batch upsert.")

    def load_to_neo4j(self, embedded_chunks: List[Dict[str, Any]]):
        """Creates structural document and chunk nodes in Neo4j for GraphRAG traversal."""
        logger.info(f"Writing {len(embedded_chunks)} structural nodes to Neo4j graph...")
        
        query = """
        UNWIND $chunks AS row
        MERGE (d:Document {filename: row.metadata.source_file})
        CREATE (c:Chunk {
            chunk_id: row.chunk_id,
            page_number: row.metadata.page_number,
            text_snippet: substring(row.text, 0, 150)
        })
        MERGE (d)-[:CONTAINS]->(c)
        """
        
        # Format payload dictionary for Neo4j unroll transaction
        row_data = [
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            }
            for chunk in embedded_chunks
        ]
        
        with self.neo4j_driver.session() as session:
            session.run(query, chunks=row_data)
            
        logger.info("Successfully synchronized structural graph nodes to Neo4j.")

    def close(self):
        self.neo4j_driver.close()