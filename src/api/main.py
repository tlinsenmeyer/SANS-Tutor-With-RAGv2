import logging
import uvicorn
from fastapi import FastAPI
from src.api.endpoints import router as api_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SANS GraphRAG Tutor API",
    description="API for querying vectorized SANS manuals and Neo4j structural knowledge graphs.",
    version="1.0.0"
)

# Register API routes
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "SANS GraphRAG Tutor API is running. Head to /docs for interactive Swagger UI."}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8080,
        reload=True
    )