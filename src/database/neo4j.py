import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class Neo4jClient:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://192.168.100.72:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j") # Fixed trailing space
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def __del__(self):
        try:
            self.driver.close()
        except:
            pass
    
    def write_graph_from_extraction(self, file_stem: str, point_id: int, chunk_text: str, entities: list):
        with self.driver.session() as session:
            session.execute_write(self._create_extracted_nodes, file_stem, point_id, chunk_text, entities)

    @staticmethod
    def _create_extracted_nodes(tx, file_stem: str, point_id: int, chunk_text: str, entities: list):
        query = """
        MERGE (m:Manual {name: $file_stem})
        MERGE (c:Chunk {id: $point_id})
        ON CREATE SET c.text = $chunk_text
        MERGE (m)-[:HAS_CHUNK]->(c)
        """
        tx.run(query, file_stem=file_stem, point_id=point_id, chunk_text=chunk_text)

        for item in entities:
            label = item.get("label", "Concept")
            name = item.get("name")
            relation = item.get("relation", "MENTIONS")
            
            if not name:
                continue

            dynamic_query = f"""
            MATCH (c:Chunk {{id: $point_id}})
            MERGE (e:`{label}` {{name: $name}})
            MERGE (c)-[:{relation}]->(e)
            """
            tx.run(dynamic_query, point_id=point_id, name=name)