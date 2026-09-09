# app/llm/vllm_client.py

from typing import List, Dict
from httpx import AsyncClient

class VLLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:3000/v1",
        model: str = "meta-llama/Llama-3.1-8B-Instruct-GPTQ"
    ):
        self.base_url = base_url
        self.model = model
        self.client = AsyncClient(timeout=60)

    async def chat(self, messages: List[Dict], max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        resp = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def completion(self, prompt: str, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        resp = await self.client.post(f"{self.base_url}/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    async def embed(self, text: str) -> List[float]:
        payload = {
            "model": self.model,
            "input": text
        }
        resp = await self.client.post(f"{self.base_url}/embeddings", json=payload)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": texts
        }
        resp = await self.client.post(f"{self.base_url}/embeddings", json=payload)
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]
