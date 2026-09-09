# src/ingestion/flow_prefect.py

from prefect import flow
from src.ingestion.tasks_prefect import (
    ensure_collection,
    convert_pdf_to_md,
    clean_markdown,
    chunk_markdown,
    embed_chunks,
    extract_entities,
    write_entities_to_neo4j,
    upsert_to_qdrant
)

@flow(name="sans_ingestion_flow")
def ingestion_flow(pdf_path: str, md_dir: str = "./processed_md"):
    ensure_collection()

    md_path = convert_pdf_to_md(pdf_path, md_dir)
    md_path = clean_markdown(md_path)

    chunks = chunk_markdown(md_path)
    vectors = embed_chunks(chunks)

    file_stem = pdf_path.split("/")[-1].split(".")[0]

    points = upsert_to_qdrant(file_stem, chunks, vectors)

    for p, c in zip(points, chunks):
        entities = extract_entities(file_stem, p["id"], c.text)
        if entities:
            write_entities_to_neo4j(file_stem, p["id"], c.text, entities)
