import subprocess
import time
import requests
import logging
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 1. Persistent File and Console Logging Setup
os.makedirs("logs", exist_ok=True)
log_file_path = "logs/pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.INFO)

from src.pipeline.parser import SANSParser
from src.pipeline.chunker import SANSChunker
from src.pipeline.embedder import SANSEmbedder
from src.pipeline.extractor import SANSEntityExtractor
from src.pipeline.loader import HybridLoader

class SANSFileHandler(FileSystemEventHandler):
    """Event handler that orchestrates parsing, chunking, embedding, extraction, and database loading."""
    def __init__(self, extractor: SANSEntityExtractor, processed_dir: Path):
        self.extractor = extractor
        self.processed_dir = processed_dir
        self.chunker = SANSChunker(chunk_size=1000, chunk_overlap=200)
        self.embedder = SANSEmbedder(model_name="all-MiniLM-L6-v2", device="cuda:0")

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        
        if file_path.suffix == ".filepart" or file_path.name.startswith("."):
            return

        logger.info(f"📄 New document detected via Watchdog: {file_path.name}. Beginning full pipeline ingestion...")

        try:
            pipeline_start_time = time.time()
            time.sleep(1.0)
            
            # Step 1: Parse PDF into page-level markdown
            parser = SANSParser(str(file_path))
            raw_pages = parser.extract_markdown()
            
            # Step 2: Chunk pages into semantic segments
            chunks = self.chunker.chunk_documents(raw_pages)
            logger.info(f"🧩 Chunked {file_path.name} into {len(chunks)} segments.")

            # Step 3: Generate dense vector embeddings on the GPU
            embedded_chunks = self.embedder.generate_embeddings(chunks)
            logger.info(f"⚡ Generated and attached embeddings for {len(embedded_chunks)} chunks.")

            # Step 4: Run entity extraction and generate audit markdown
            output_md_path = f"output/{file_path.stem}_knowledge.md"
            extraction_results = self.extractor.process_and_save(embedded_chunks, output_md_path=output_md_path)
            logger.info(f"📁 Audit markdown generated and saved to {output_md_path}")

            # Step 5: Synchronize with Qdrant and Neo4j databases via HybridLoader
            logger.info("💾 Synchronizing pipeline outputs to Qdrant and Neo4j databases...")
            loader = HybridLoader()
            # Step 6: Archive source PDF so it isn't processed twice
            target_path = self.processed_dir / file_path.name
            file_path.rename(target_path)
            
            # Total pipeline elapsed time
            total_elapsed = time.time() - pipeline_start_time
            minutes, seconds = divmod(total_elapsed, 60)
            
            logger.info(f"✅ Successfully processed {file_path.name} and archived to {self.processed_dir}")
            logger.info(f"🏁 Total pipeline execution time for {file_path.name}: {int(minutes)}m {seconds:.2f}s")
            logger.info("👀 Ready and waiting for next SANS PDF drop...")
            try:
                loader.load_to_qdrant(embedded_chunks)
                loader.load_to_neo4j(embedded_chunks)
                loader.load_extracted_knowledge_to_neo4j(extraction_results)
            finally:
                loader.close()
            logger.info("🎯 Database synchronization complete!")

            # Step 6: Archive source PDF so it isn't processed twice
            target_path = self.processed_dir / file_path.name
            file_path.rename(target_path)
            
            # Total pipeline elapsed time
            total_elapsed = time.time() - pipeline_start_time
            minutes, seconds = divmod(total_elapsed, 60)
            
            logger.info(f"✅ Successfully processed {file_path.name} and archived to {self.processed_dir}")
            logger.info(f"🏁 Total pipeline execution time for {file_path.name}: {int(minutes)}m {seconds:.2f}s")

        except Exception as e:
            logger.error(f"❌ Error processing pipeline for {file_path.name}: {e}", exc_info=True)


class PipelineWatcher:
    def __init__(self, watch_dir: str = "files"):
        self.container_name = "vllm-sans-extractor"
        self.model_name = "casperhansen/mistral-nemo-instruct-2407-awq"
        self.watch_dir = Path(watch_dir)
        self.processed_dir = Path("data/processed")

    def _is_server_healthy(self):
        try:
            response = requests.get("http://localhost:8000/v1/models", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _start_vllm_server(self):
        """Spawns the vLLM container if not already running."""
        if self._is_server_healthy():
            logger.info("✅ vLLM container server is already running.")
            return

        logger.info("🚀 Starting containerized vLLM server on GPU...")
        subprocess.run(["docker", "rm", "-f", self.container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        home_dir = subprocess.check_output(["echo", "$HOME"]).decode().strip()
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--gpus", "all",
            "--ipc=host",
            "-p", "8000:8000",
            "-v", f"{home_dir}/.cache/huggingface:/root/.cache/huggingface",
            "nvcr.io/nvidia/vllm:26.08-py3",
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--tensor-parallel-size", "1",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.85"
        ]

        subprocess.run(cmd, check=True)
        
        logger.info("⏳ Waiting for vLLM containerized model weights to load into VRAM...")
        start_time = time.time()
        timeout = 180
        
        while time.time() - start_time < timeout:
            if self._is_server_healthy():
                logger.info("✅ vLLM container server is online and ready!")
                return
            time.sleep(4)
            
        raise TimeoutError("Timed out waiting for containerized vLLM server initialization.")

    def _stop_vllm_server(self):
        """Stops and removes the vLLM container to free VRAM post-pipeline."""
        logger.info("🛑 Shutting down vLLM Docker container...")
        subprocess.run(["docker", "stop", self.container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", self.container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("🧹 vLLM container successfully stopped and VRAM released.")

    def run(self):
        """Starts the vLLM server, checks databases, and initiates the watchdog monitoring loop."""
        os.makedirs(self.watch_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        # 1. Start vLLM
        self._start_vllm_server()
        
        # 2. Check Database Connectivity proactively
        logger.info("🔍 Verifying Qdrant and Neo4j availability...")
        try:
            test_loader = HybridLoader()
            if not test_loader.check_health():
                raise ConnectionError("Qdrant or Neo4j is not reachable at the configured IPs/ports.")
            test_loader.close()
            logger.info("✅ Database connections verified successfully!")
        except Exception as e:
            logger.error(f"❌ Database startup check failed: {e}")
            self._stop_vllm_server()
            return

        extractor = SANSEntityExtractor(model_name=self.model_name)
        
        event_handler = SANSFileHandler(extractor, self.processed_dir)
        observer = Observer()
        observer.schedule(event_handler, path=str(self.watch_dir), recursive=False)
        
        observer.start()
        logger.info(f"👀 Watchdog active! Monitoring folder [{self.watch_dir}] for incoming SANS PDFs... (Press Ctrl+C to exit)")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("🛑 Interrupted by user. Stopping watchdog...")
        finally:
            observer.join()
            self._stop_vllm_server()
        
        # 2. Check Database Connectivity proactively
        logger.info("🔍 Verifying Qdrant and Neo4j availability...")
        try:
            test_loader = HybridLoader()
            if not test_loader.check_health():
                raise ConnectionError("Qdrant or Neo4j is not reachable at the configured IPs/ports.")
            test_loader.close()
            logger.info("✅ Database connections verified successfully!")
        except Exception as e:
            logger.error(f"❌ Database startup check failed: {e}")
            self._stop_vllm_server()
            return

        extractor = SANSEntityExtractor(model_name=self.model_name)
        
        event_handler = SANSFileHandler(extractor, self.processed_dir)
        observer = Observer()
        observer.schedule(event_handler, path=str(self.watch_dir), recursive=False)
        
        observer.start()
        logger.info(f"👀 Watchdog active! Monitoring folder [{self.watch_dir}] for incoming SANS PDFs... (Press Ctrl+C to exit)")
        # ... rest of run loop ...

# --- Direct Watcher Execution Script (Outside of Functions) ---
if __name__ == "__main__":
    watcher = PipelineWatcher()
    watcher.run()