from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="SANS Tutor With RAG v2",
    version="1.0.0"
)

app.include_router(router)

