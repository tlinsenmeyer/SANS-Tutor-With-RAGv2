# src/ingestion/tasks_prefect.py

from pathlib import Path
import json
import re
import pymupdf4llm
from prefect import task
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from openai import OpenAI

from src.ingestion.chunker import chunk_document
from src.database.neo4j import Neo4jClient

# ---------------------------------------------------------------------------
# LLM CONFIG
# ---------------------------------------------------------------------------
llm_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"

# ---------------------------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------------------------
embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# ---------------------------------------------------------------------------
# QDRANT (server)
# ---------------------------------------------------------------------------
qdrant = QdrantClient(
    host="192.168.100.72",
    port=6333
)
COLLECTION_NAME = "sans_manuals"

# ---------------------------------------------------------------------------
# Ensure Qdrant collection exists
# ---------------------------------------------------------------------------
@task
def ensure_collection():
    collections = qdrant.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in collections):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        print(f"[QDRANT] Created collection: {COLLECTION_NAME}")

# ---------------------------------------------------------------------------
# PDF → Markdown
# ---------------------------------------------------------------------------
@task
def convert_pdf_to_md(pdf_path: str, md_dir: str) -> str:
    pdf = Path(pdf_path)
    md_dir = Path(md_dir)
    md_dir.mkdir(parents=True, exist_ok=True)

    output_file = md_dir / f"{pdf.stem}.md"
    markdown = pymupdf4llm.to_markdown(str(pdf))
    output_file.write_text(markdown, encoding="utf-8")

    return str(output_file)

# ---------------------------------------------------------------------------
# Markdown Cleaner
# ---------------------------------------------------------------------------
@task
def clean_markdown(md_path: str) -> str:
    file_path = Path(md_path)
    content = file_path.read_text(encoding="utf-8")

    content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
    content = re.sub(r'<!-- Start of picture text -->.*?<!-- End of picture text -->', '', content, flags=re.DOTALL)
    content = re.sub(r'©\s*\d{4}\s+.*?All rights reserved.*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'©\s*\d{4}\s+Ed Skoudis and John Strand', '', content)
    content = re.sub(r'^\s*\d{1,3}\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'.*Techniques,\s*Exploits,\s*&\s*Incident Handling.*', '', content, flags=re.IGNORECASE)

    lines = content.splitlines()
    cleaned_lines = []
    buffer = []
    seen_headers = set()

    def flush():
        if buffer:
            cleaned_lines.append(" ".join(buffer).strip())
            buffer.clear()

    in_table = False
    in_code = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush()
            in_code = not in_code
            cleaned_lines.append(line)
            continue

        if in_code:
            cleaned_lines.append(line)
            continue

        if stripped.startswith("#"):
            flush()
            if stripped.lower() not in seen_headers:
                seen_headers.add(stripped.lower())
                cleaned_lines.append(line)
            continue

        is_table = stripped.startswith("|") and stripped.endswith("|")
        if is_table:
            flush()
            in_table = True
            cleaned_lines.append(line)
            continue
        elif in_table and not is_table:
            in_table = False

        is_list = re.match(r"^(\*|-|\d+\.)\s+", stripped)
        if is_list:
            flush()
            cleaned_lines.append(line)
            continue

        if not stripped:
            flush()
            cleaned_lines.append("")
            continue

        buffer.append(stripped)

    flush()
    final = "\n".join(cleaned_lines)
    final = re.sub(r'\n{3,}', '\n\n', final)

    file_path.write_text(final.strip(), encoding="utf-8")
    return md_path

# ---------------------------------------------------------------------------
# Chunk Markdown
# ---------------------------------------------------------------------------
@task
def chunk_markdown(md_path: str):
    text = Path(md_path).read_text(encoding="utf-8")
    return chunk_document(text)

# ---------------------------------------------------------------------------
# Embed Chunks
# ---------------------------------------------------------------------------
@task
def embed_chunks(chunks):
    texts = [c.text for c in chunks]
    vectors = embed_model.encode(texts, batch_size=64, show_progress_bar=False).tolist()
    return vectors

# ---------------------------------------------------------------------------
# Entity Extraction
# ---------------------------------------------------------------------------
@task
def extract_entities(file_stem: str, point_id: int, chunk_text: str):
    prompt = f"""
    Extract cybersecurity entities from the following text.

    Return ONLY valid JSON with:
    - label: one of ["Tool", "Vulnerability", "Protocol", "System", "Concept"]
    - name: the entity name
    - relation: one of ["MENTIONS", "USES_TOOL", "DISCUSSES", "COVERS_CONCEPT"]

    Text:
    {chunk_text}
    """

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You output strictly valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return data.get("entities", [])

# ---------------------------------------------------------------------------
# Write to Neo4j
# ---------------------------------------------------------------------------
@task
def write_entities_to_neo4j(file_stem: str, point_id: int, chunk_text: str, entities: list):
    neo = Neo4jClient()
    neo.write_graph_from_extraction(file_stem, point_id, chunk_text, entities)

# ---------------------------------------------------------------------------
# Write to Qdrant
# ---------------------------------------------------------------------------
@task
def upsert_to_qdrant(file_stem: str, chunks, vectors):
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        pid = hash(f"{file_stem}_{idx}") & 0x7FFFFFFF
        points.append({
            "id": pid,
            "vector": vector,
            "payload": {
                "text": chunk.text,
                "source_file": file_stem,
                "type": chunk.metadata.get("type", "block") if chunk.metadata else "block"
            }
        })

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return points
