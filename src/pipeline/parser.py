import os
import logging
import pymupdf4llm
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SANSParser:
    """Parses SANS technical training PDFs into structured page-by-page markdown chunks."""
    
    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        self.pdf_path = pdf_path

    def extract_markdown(self) -> List[Dict[Any, Any]]:
        """
        Extracts the PDF content into markdown chunks per page, preserving 
        multi-column formatting, code blocks, and tables.
        """
        logger.info(f"Extracting markdown from {self.pdf_path} using pymupdf4llm...")
        try:
            # pymupdf4llm extracts pages as a list of dictionaries with 'text' and metadata
            md_pages = pymupdf4llm.to_markdown(self.pdf_path, page_chunks=True)
            logger.info(f"Successfully extracted {len(md_pages)} pages/chunks from PDF.")
            return md_pages
        except Exception as e:
            logger.error(f"Failed to parse PDF {self.pdf_path}: {e}")
            raise e