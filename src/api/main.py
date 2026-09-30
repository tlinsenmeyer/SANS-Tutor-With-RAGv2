import os
from contextlib import asynccontextmanager
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))
    neo4j_driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", ""),
        ),
    )
    app.state.qdrant = qdrant
    app.state.neo4j = neo4j_driver

    try:
        yield
    finally:
        qdrant.close()
        neo4j_driver.close()


app = FastAPI(title="SANS Tutor API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def readiness(request: Request) -> dict[str, Any]:
    services = {"qdrant": "ok", "neo4j": "ok"}

    try:
        request.app.state.qdrant.get_collections()
    except Exception:
        services["qdrant"] = "unavailable"

    try:
        request.app.state.neo4j.verify_connectivity()
    except Exception:
        services["neo4j"] = "unavailable"

    if "unavailable" in services.values():
        raise HTTPException(
            status_code=503,
            detail={"status": "unavailable", "services": services},
        )

    return {"status": "ok", "services": services}