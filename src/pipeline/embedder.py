import logging
from typing import List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SANSEmbedder:
    """Generates dense vector embeddings for SANS text chunks targeting GPU 1."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda:0"):
        self.model_name = model_name
        
        # Fallback to CPU safely if GPU 1 isn't present, otherwise bind to cuda:1
        if self.device_is_valid(device):
            self.device = device
        else:
            logger.warning(f"Requested device {device} is unavailable. Falling back to cpu.")
            self.device = "cpu"
            
        logger.info(f"Loading embedding model '{self.model_name}' on device: {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    @staticmethod
    def device_is_valid(device_str: str) -> bool:
        if device_str.startswith("cuda") and torch.cuda.is_available():
            try:
                device_idx = int(device_str.split(":")[-1])
                return device_idx < torch.cuda.device_count()
            except ValueError:
                return False
        return device_str == "cpu"

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts text from chunks, computes dense vector representations in batches,
        and appends the resulting vectors back into the chunk payload dictionaries.
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks on {self.device}...")
        
        texts = [chunk["text"] for chunk in chunks]
        
        # Batch encode text into dense floating-point vectors
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Unit-norm for efficient cosine similarity search in Qdrant
        )
        
        embedded_payloads = []
        for chunk, vector in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            # Convert NumPy float32 array to a standard Python list for database wire serialization
            chunk_copy["vector"] = vector.tolist()
            embedded_payloads.append(chunk_copy)
            
        logger.info("Successfully generated and attached embedding vectors to all payloads.")
        return embedded_payloads