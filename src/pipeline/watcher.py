import os
import time
import logging
from typing import NamedTuple
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import your pipeline components
from pipeline.parser import SANSParser
from pipeline.chunker import SANSChunker
from pipeline.embedder import SANSEmbedder
from pipeline.extractor import SANSEntityExtractor
from pipeline.loader import HybridLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PipelineConfig(NamedTuple):
    qdrant_url: str = "http://192.168.100.87:6333"
    neo4j_uri: str = "bolt://192.168.100.72:7687"
    embedding_device: str = "cuda:1"

class SANSFileHandler(FileSystemEventHandler):
    """Event handler that triggers the hybrid RAG ingestion pipeline upon PDF drop."""
    
    def __init__(self, watch_directory: str):
        self.watch_directory = watch_directory
        self.config = PipelineConfig()

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        if not file_path.lower().endswith(".pdf"):
            return

        logger.info(f"New PDF detected: {file_path}. Waiting for write completion...")
        self._wait_for_file_stability(file_path)
        
        try:
            self.run_pipeline(file_path)
        except Exception as e:
            logger.error(f"Pipeline execution failed for {file_path}: {e}")

    def _wait_for_file_stability(self, file_path: str, timeout: int = 60, check_interval: float = 1.0):
        """Ensures large files finish fully copying/dropping before parsing begins."""
        start_time = time.time()
        initial_size = -1
        
        while time.time() - start_time < timeout:
            if not os.path.exists(file_path):
                time.sleep(check_interval)
                continue
                
            current_size = os.path.getsize(file_path)
            if current_size == initial_size and current_size > 0:
                logger.info(f"File stability verified for: {file_path}")
                return
                
            initial_size = current_size
            time.sleep(check_interval)
            
        logger.warning(f"File stability timeout reached for {file_path}, proceeding anyway.")

    def run_pipeline(self, pdf_path: str):
        logger.info(f"=== Starting Ingestion Pipeline for: {pdf_path} ===")
        
        # Step 1: Parse PDF into markdown pages via pymupdf4llm
        parser = SANSParser(pdf_path=pdf_path)
        pages = parser.extract_markdown()

        # Step 2: Semantic sliding-window chunking
        chunker = SANSChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk_documents(pages)

        # Step 3: Generate dense vector embeddings on GPU 1
        embedder = SANSEmbedder(model_name="all-MiniLM-L6-v2", device=self.config.embedding_device)
        embedded_chunks = embedder.generate_embeddings(chunks)

        # Step 4: LLM-assisted entity & relation extraction via local vLLM on GPU 0
        extractor = SANSEntityExtractor()
        filename = os.path.basename(pdf_path)
        audit_path = f"output/audit_{filename}.md"
        enriched_chunks = extractor.process_and_save(embedded_chunks, output_md_path=audit_path)

        # Step 5: Dual-write synchronization to Qdrant & Neo4j
        loader = HybridLoader(
            qdrant_url=self.config.qdrant_url,
            neo4j_uri=self.config.neo4j_uri
        )
        loader.load_to_qdrant(enriched_chunks)
        loader.load_to_neo4j(enriched_chunks)
        loader.close()

        logger.info(f"=== Successfully Completed Pipeline for: {pdf_path} ===")


if __name__ == "__main__":
    WATCH_DIR = "files"
    os.makedirs(WATCH_DIR, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    event_handler = SANSFileHandler(watch_directory=WATCH_DIR)
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=False)

    logger.info(f"Starting file watcher service on directory: ./{WATCH_DIR}/")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("File watcher service stopped by user.")
    observer.join()