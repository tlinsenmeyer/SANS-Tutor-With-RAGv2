import uuid
import logging
from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

from src.api.schemas import QueryRequest, QueryResponse, SourceReference, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Tutor GraphRAG"])

# Initialize clients and models using your standard configuration
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant_client = AsyncQdrantClient(url="http://localhost:6333")
neo4j_driver = AsyncGraphDatabase.driver("bolt://192.168.100.72:7687", auth=("neo4j", "your_password"))
vllm_client = AsyncOpenAI(api_key="not-needed", base_url="http://localhost:8000/v1")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    qdrant_status = "healthy"
    neo4j_status = "healthy"
    vllm_status = "healthy"

    try:
        await qdrant_client.get_collections()
    except Exception:
        qdrant_status = "unhealthy"

    try:
        async with neo4j_driver.session() as session:
            await session.run("RETURN 1")
    except Exception:
        neo4j_status = "unhealthy"

    overall = "healthy" if all(s == "healthy" for s in [qdrant_status, neo4j_status, vllm_status]) else "degraded"

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        neo4j=neo4j_status,
        vllm=vllm_status
    )

@router.post("/query", response_model=QueryResponse)
async def query_sans_tutor(payload: QueryRequest):
    trace_id = f"tr_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{trace_id}] Received query: '{payload.question}'")

    try:
        # Step 1: Vectorize incoming query
        query_vector = embedding_model.encode(payload.question).tolist()

        # Step 2: Query Qdrant first (Semantic Search)
        search_result = await qdrant_client.query_points(
            collection_name="sans_vectors",
            query=query_vector,
            limit=payload.top_k,
            with_payload=True
        )

        sources = []
        retrieved_texts = []
        extracted_entities = set()

        for point in search_result.points:
            p_data = point.payload or {}
            chunk_id = p_data.get("chunk_id", str(point.id))
            page_num = p_data.get("page_number", 0)
            source_file = p_data.get("source_file", "SEC-504")
            text_content = p_data.get("text", "")

            sources.append(SourceReference(
                chunk_id=chunk_id,
                page_number=page_num,
                source_file=source_file,
                score=float(point.score)
            ))
            retrieved_texts.append(text_content)

            for ent in p_data.get("entities", []):
                if isinstance(ent, dict) and "name" in ent:
                    extracted_entities.add(ent["name"])
                elif isinstance(ent, str):
                    extracted_entities.add(ent)

        # Step 3: Query Neo4j second (Structural Graph Traversal)
        graph_context = []
        if payload.include_graph and extracted_entities:
            async with neo4j_driver.session() as session:
                cypher_query = """
                MATCH (e:Entity)-[r]->(target)
                WHERE e.name IN $entities
                RETURN e.name AS source, type(r) AS relation, target.name AS target
                LIMIT 15
                """
                result = await session.run(cypher_query, entities=list(extracted_entities))
                async for record in result:
                    graph_context.append({
                        "source": record["source"],
                        "relation": record["relation"],
                        "target": record["target"]
                    })

        # Step 4: Build prompt & query vLLM
        combined_context = "\n\n".join(retrieved_texts)
        graph_text = "\n".join([f"- {g['source']} --[{g['relation']}]--> {g['target']}" for g in graph_context])

        system_prompt = (
            "You are an expert SANS cybersecurity tutor. Answer the user's question accurately "
            "using only the provided technical context and graph relationships from the SANS manuals."
        )
        
        user_prompt = (
            f"Question: {payload.question}\n\n"
            f"Retrieved Document Context:\n{combined_context}\n\n"
            f"Knowledge Graph Relationships:\n{graph_text}\n\n"
            "Provide a detailed, instructional answer formatted cleanly."
        )

        response = await vllm_client.chat.completions.create(
            model="your-deployed-vllm-model",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )

        answer = response.choices[0].message.content or "No response generated."

        return QueryResponse(
            trace_id=trace_id,
            answer=answer,
            sources=sources,
            graph_context=graph_context,
            status="success"
        )

    except Exception as e:
        logger.error(f"[{trace_id}] Error handling query pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))