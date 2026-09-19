import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class SANSEntityExtractor:
    """Asynchronously extracts cybersecurity entities and relationships from chunks using containerized vLLM with noise filtering."""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1", max_concurrency: int = 4):
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key="not-needed", base_url=base_url)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def _clean_chunk_text(self, text: str) -> str:
        """Removes common slide deck boilerplate, repeating footers, and copyright noise."""
        # Strip common SANS footer patterns, copyright notices, and page numbers
        cleaned = re.sub(r"©\s*\d{4}.*?(?:All Rights Reserved|Ed Skoudis|John Strand)", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"Tools,\s*Techniques,\s*Exploits,.*?\d+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"Page\s*\d+", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    async def _extract_single_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Sends a cleaned chunk to vLLM with strict filtering instructions."""
        async with self.semaphore:
            raw_text = chunk["text"]
            chunk_text = self._clean_chunk_text(raw_text)
            chunk_id = chunk["chunk_id"]
            page_num = chunk["metadata"].get("page_number", "Unknown")

            system_prompt = (
                "You are a precise cybersecurity data extraction assistant specializing in SANS training material. "
                "Always respond in valid JSON format with keys: 'entities' (list of objects with 'name' and 'type') "
                "and 'relationships' (list of objects with 'source', 'target', and 'relation')."
            )

            prompt = (
                "Analyze the technical text segment below extracted from a SANS manual.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Ignore and do NOT extract repeating slide headers, footers, page numbers, course titles, or copyright text.\n"
                "2. Only extract concrete, high-value cybersecurity entities (e.g., tools, protocols, vulnerabilities, concepts, commands, commands flags).\n"
                "3. Avoid fragmented words, broken OCR artifacts, or generic UI words.\n\n"
                f"Text Segment (Page {page_num}):\n{chunk_text}\n\n"
                "Provide your output in valid JSON."
            )

            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                extracted_data = json.loads(content)
                
                return {
                    "chunk_id": chunk_id,
                    "page_number": page_num,
                    "extracted_data": extracted_data,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"❌ Error extracting entities for chunk {chunk_id}: {e}")
                return {
                    "chunk_id": chunk_id,
                    "page_number": page_num,
                    "extracted_data": {"entities": [], "relationships": []},
                    "status": "error",
                    "error": str(e)
                }

    async def _process_chunks_async(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tasks = [self._extract_single_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        return results

    def process_and_save(self, embedded_chunks: List[Dict[str, Any]], output_md_path: str):
        logger.info(f"🚀 Starting asynchronous LLM entity extraction for {len(embedded_chunks)} chunks with noise filtering...")
        
        start_time = time.time()
        results = asyncio.run(self._process_chunks_async(embedded_chunks))
        elapsed = time.time() - start_time
        
        logger.info(f"⚡ Extraction batch complete in {elapsed:.2f}s ({elapsed/len(embedded_chunks):.2f}s avg per chunk).")

        # Compile results into audit markdown
        output_path = Path(output_md_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# SANS Knowledge Extraction Audit Report (Filtered)\n\n")
            f.write(f"* **Total Chunks Processed:** {len(embedded_chunks)}\n")
            f.write(f"* **Extraction Time:** {elapsed:.2f} seconds\n\n---\n\n")

            for res in results:
                f.write(f"### Chunk ID: {res['chunk_id']} (Page {res['page_number']})\n")
                data = res["extracted_data"]
                
                f.write("**Entities:**\n")
                for entity in data.get("entities", []):
                    name = entity.get("name", entity) if isinstance(entity, dict) else entity
                    etype = entity.get("type", "UNKNOWN") if isinstance(entity, dict) else "UNKNOWN"
                    f.write(f"- {name} *({etype})*\n")
                
                f.write("\n**Relationships:**\n")
                for rel in data.get("relationships", []):
                    if isinstance(rel, dict):
                        f.write(f"- {rel.get('source')} --[{rel.get('relation')}]--> {rel.get('target')}\n")
                    else:
                        f.write(f"- {rel}\n")
                
                f.write("\n---\n\n")

        logger.info(f"📁 Filtered audit markdown successfully compiled and saved to {output_md_path}")