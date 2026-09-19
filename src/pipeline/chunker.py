import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SANSChunker:
    """Splits parsed SANS markdown documents into overlapping semantic chunks for hybrid RAG."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Recursive character text splitter prioritizes natural language boundaries:
        # paragraphs (\n\n), lines (\n), spaces, and characters.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, parsed_pages: List[Dict[Any, Any]]) -> List[Dict[str, Any]]:
        """
        Takes page-level markdown dictionaries and transforms them into 
        granular, indexed text chunks with preserved metadata.
        """
        logger.info(f"Starting chunking process for {len(parsed_pages)} pages (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        
        chunked_payloads = []
        global_chunk_idx = 0

        for page in parsed_pages:
            page_text = page.get("text", "")
            page_metadata = page.get("metadata", {})
            page_num = page_metadata.get("page", 0)

            if not page_text.strip():
                continue

            # Split the page markdown into chunks
            splits = self.splitter.split_text(page_text)

            for local_idx, chunk_text in enumerate(splits):
                chunk_id = f"page_{page_num}_chunk_{local_idx}"
                
                chunk_payload = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "page_number": page_num,
                        "global_index": global_chunk_idx,
                        "source_file": page_metadata.get("file_path", "unknown_sans_manual.pdf")
                    }
                }
                chunked_payloads.append(chunk_payload)
                global_chunk_idx += 1

        logger.info(f"Generated {len(chunked_payloads)} total semantic chunks from document stream.")
        return chunked_payloads