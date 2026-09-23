import logging
from typing import List, Dict, Any
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SANSChunker:
    """Splits parsed SANS markdown documents into overlapping semantic chunks with robust whitespace handling."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, parsed_pages: List[Any]) -> List[Dict[str, Any]]:
        logger.info(f"Starting robust chunking process for {len(parsed_pages)} pages...")
        
        chunked_payloads = []
        global_chunk_idx = 0

        for idx, page in enumerate(parsed_pages):
            page_text = ""
            page_num = idx + 1
            source_file = "unknown_sans_manual.pdf"

            if isinstance(page, str):
                page_text = page
            elif isinstance(page, (dict, defaultdict)):
                # Try getting text, and if it's whitespace/empty, try fallback keys
                raw_text = page.get("text", "")
                if raw_text and isinstance(raw_text, str) and raw_text.strip():
                    page_text = raw_text
                else:
                    for k in ["content", "markdown", "body", "page_text"]:
                        alt = page.get(k, "")
                        if alt and isinstance(alt, str) and alt.strip():
                            page_text = alt
                            break

                # If still empty but dict has other strings, gather them
                if not page_text.strip():
                    fallback_strs = [str(v) for v in page.values() if isinstance(v, str) and len(v.strip()) > 5]
                    if fallback_strs:
                        page_text = "\n".join(fallback_strs)

                metadata = page.get("metadata", {})
                if isinstance(metadata, dict):
                    page_num = metadata.get("page", metadata.get("page_number", idx + 1))
                    source_file = metadata.get("file_path", source_file)
            else:
                page_text = str(page)

            # If a page is truly devoid of text, use a placeholder instead of dropping it entirely
            if not page_text or not page_text.strip():
                page_text = f"[Blank or Image Page {page_num}]"

            splits = self.splitter.split_text(page_text)

            for local_idx, chunk_text in enumerate(splits):
                chunk_id = f"page_{page_num}_chunk_{local_idx}"
                
                chunk_payload = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "page_number": page_num,
                        "global_index": global_chunk_idx,
                        "source_file": source_file
                    }
                }
                chunked_payloads.append(chunk_payload)
                global_chunk_idx += 1

        logger.info(f"Generated {len(chunked_payloads)} total semantic chunks from document stream.")
        return chunked_payloads