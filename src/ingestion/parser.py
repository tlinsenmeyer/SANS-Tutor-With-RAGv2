# src/ingestion/parser.py

import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.ingestion.flow_prefect import ingestion_flow

class PipelineHandler(FileSystemEventHandler):
    def __init__(self, pdf_dir: str, md_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.md_dir = Path(md_dir)
        self._cache = {}

    def _debounced(self, path: Path):
        now = time.time()
        last = self._cache.get(path, 0)
        if now - last < 3:
            return True
        self._cache[path] = now
        return False

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if self._debounced(path):
            return

        if path.suffix.lower() == ".pdf":
            print(f"[WATCHER] New PDF detected: {path.name}")
            ingestion_flow(str(path), str(self.md_dir))

def start_pipeline(
    pdf_path=str(Path.home() / "Documents" / "files"),
    md_path=str(Path.home() / "Documents" / "processed_md")
):
    Path(pdf_path).mkdir(exist_ok=True)
    Path(md_path).mkdir(exist_ok=True)

    handler = PipelineHandler(pdf_path, md_path)
    observer = Observer()
    observer.schedule(handler, path=pdf_path, recursive=False)

    print(f"[DAEMON] Watching '{pdf_path}' for new PDFs...")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("[DAEMON] Stopping watcher.")

    observer.join()


if __name__ == "__main__":
    start_pipeline()
