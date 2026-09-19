import os
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridLoader:
    """Synchronizes embedded chunks into Qdrant (vectors) and Neo4j (graph nodes and extracted knowledge)."""
    
    def __init__(
        self, 
        qdrant_url: str = None,
        qdrant_collection: str = "sans_vectors",
        neo4j_uri: str = None,
        neo4j_auth: tuple = None
    ):
        # Fall back to environment variables or safe defaults
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = qdrant_collection
        
        neo4j_host = neo4j_uri or os.getenv("NEO4J_URI", "bolt://192.168.100.72:7687")
        neo4j_pwd = os.getenv("NEO4J_PASSWORD", "password")
        auth_tuple = neo4j_auth or ("neo4j", neo4j_pwd)
        
        logger.info(f"Connecting to Qdrant at {self.qdrant_url}...")
        self.qdrant_client = QdrantClient(url=self.qdrant_url)
        
        logger.info(f"Connecting to Neo4j at {neo4j_host}...")
        self.neo4j_driver = GraphDatabase.driver(neo4j_host, auth=auth_tuple)
        
        if not self.check_health():
            raise ConnectionError("Failed to connect to Qdrant or Neo4j databases during initialization.")
            
        self._init_qdrant_collection()

    def check_health(self) -> bool:
        """Verifies that both Qdrant and Neo4j are reachable and responsive."""
        try:
            # Test Qdrant connectivity
            self.qdrant_client.get_collections()
            
            # Test Neo4j connectivity
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
                
            logger.info("✅ Qdrant and Neo4j health check passed.")
            return True
        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
            return False

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
                    id=idx,  
                    vector=chunk["vector"],
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "page_number": chunk["metadata"]["page_number"],
                        "source_file": chunk["metadata"]["source_file"]
                    }
                )
            )
            
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

    def load_extracted_knowledge_to_neo4j(self, extraction_results: List[Dict[str, Any]]):
        """Pushes LLM-extracted technical entities and relationships into Neo4j."""
        logger.info(f"Syncing extracted knowledge entities and relations into Neo4j...")
        
        entity_query = """
        UNWIND $entities AS ent
        MERGE (e:Entity {name: ent.name})
        ON CREATE SET e.type = ent.type
        ON MATCH SET e.type = ent.type
        """
        
        safe_rel_query = """
        UNWIND $relations AS rel
        MERGE (s:Entity {name: rel.source})
        MERGE (t:Entity {name: rel.target})
        MERGE (s)-[r:RELATED_TO {type: rel.relation}]->(t)
        """

        entities_list = []
        relations_list = []

        for res in extraction_results:
            data = res.get("extracted_data", {})
            for ent in data.get("entities", []):
                if isinstance(ent, dict):
                    entities_list.append({"name": ent.get("name"), "type": ent.get("type", "UNKNOWN")})
                else:
                    entities_list.append({"name": ent, "type": "UNKNOWN"})
                    
            for rel in data.get("relationships", []):
                if isinstance(rel, dict):
                    relations_list.append({
                        "source": rel.get("source"),
                        "target": rel.get("target"),
                        "relation": rel.get("relation", "RELATED")
                    })

        with self.neo4j_driver.session() as session:
            if entities_list:
                session.run(entity_query, entities=entities_list)
            if relations_list:
                session.run(safe_rel_query, relations=relations_list)
                
        logger.info(f"Successfully synchronized {len(entities_list)} entities and {len(relations_list)} relationships into Neo4j.")

    def close(self):
        self.neo4j_driver.close()