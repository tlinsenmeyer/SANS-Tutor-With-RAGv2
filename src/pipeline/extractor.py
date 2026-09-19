import os
import json
import logging
from typing import List, Dict, Any
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SANSEntityExtractor:
    """Uses the local vLLM inference server to extract cybersecurity entities and relationships for GraphRAG."""
    
    def __init__(
        self, 
        vllm_base_url: str = "http://localhost:8000/v1", 
        model_name: str = "mistralai/Mistral-Nemo-Instruct-2407"
    ):
        # Point the OpenAI client to your local air-gapped vLLM control plane
        self.client = OpenAI(base_url=vllm_base_url, api_key="not-needed-for-local")
        self.model_name = model_name

    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Sends a text chunk to the local LLM to extract entities and relations in JSON format."""
        prompt = f"""
        Analyze the following SANS cybersecurity technical text and extract key entities and relationships.
        Entities should include types: TOOL, PROTOCOL, VULNERABILITY (CVE), CONCEPT, DEFENSE.
        Relationships should map how these entities interact with each other.
        
        Return ONLY a valid JSON object with two keys:
        1. "entities": a list of objects containing "name" and "type".
        2. "relations": a list of objects containing "source", "target", and "relation".
        
        Text:
        {text[:3000]}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic output for structured extraction
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Entity extraction inference failed: {e}")
            return {"entities": [], "relations": []}

    def process_and_save(
        self, 
        chunks: List[Dict[str, Any]], 
        output_md_path: str = "output/extracted_knowledge.md"
    ) -> List[Dict[str, Any]]:
        """
        Iterates through chunks, extracts graph nodes and edges via LLM,
        saves an audit markdown file, and returns enriched chunk payloads.
        """
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
        enriched_chunks = []
        
        markdown_report = "# SANS Extracted Knowledge Graph Entities & Relations\n\n"

        logger.info(f"Starting LLM entity extraction pipeline for {len(chunks)} chunks...")

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            page_num = chunk["metadata"]["page_number"]
            
            extraction = self.extract_from_text(chunk["text"])
            
            # Attach extracted knowledge directly to the chunk payload
            chunk_copy = chunk.copy()
            chunk_copy["extracted_data"] = extraction
            enriched_chunks.append(chunk_copy)

            # Append structured logs to our local audit markdown file
            markdown_report += f"## Chunk: {chunk_id} (Page {page_num})\n"
            markdown_report += f"**Entities Found:**\n```json\n{json.dumps(extraction.get('entities', []), indent=2)}\n```\n"
            markdown_report += f"**Relations Found:**\n```json\n{json.dumps(extraction.get('relations', []), indent=2)}\n```\n\n---\n"

        # Write out the human-auditable markdown report file
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)
            
        logger.info(f"Extraction complete. Markdown audit report saved to {output_md_path}")
        return enriched_chunks