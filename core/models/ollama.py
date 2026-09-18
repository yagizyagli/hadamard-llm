from typing import Any, Dict, List, Optional, AsyncGenerator
import json
import httpx
from core.models.base import BaseModel
from core.exceptions import ModelProviderError

class OllamaModel(BaseModel):
    """
    Ultra-lightweight Ollama model implementation for running open-source
    models locally (e.g., llama3, mistral) via raw HTTPX requests.
    """

    def __init__(self, model_name: str = "llama3", temperature: float = 0.7, base_url: str = "http://localhost:11434", **kwargs: Any):
        super().__init__(model_name, temperature, **kwargs)
        self.endpoint = f"{base_url}/api/generate"

    def _build_payload(self, prompt: str, system_instruction: Optional[str] = None, stream: bool = False, **kwargs: Any) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                **self.config,
                **kwargs
            }
        }
        if system_instruction:
            payload["system"] = system_instruction
        return payload

    async def generate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs: Any) -> str:
        payload = self._build_payload(prompt, system_instruction, stream=False, **kwargs)
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.endpoint, json=payload, headers=headers, timeout=120.0)
                if response.status_code != 200:
                    raise ModelProviderError("Ollama", response.status_code, response.text)
                
                data = response.json()
                return data["response"]
            except httpx.HTTPError as exc:
                raise ModelProviderError("Ollama", 500, f"Local Ollama connection failed: {str(exc)}")

    async def generate_stream(self, prompt: str, system_instruction: Optional[str] = None, **kwargs: Any) -> AsyncGenerator[str, None]:
        payload = self._build_payload(prompt, system_instruction, stream=True, **kwargs)
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", self.endpoint, json=payload, headers=headers, timeout=120.0) as response:
                    if response.status_code != 200:
                        raise ModelProviderError("Ollama", response.status_code, await response.aread())
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk_data = json.loads(line)
                            yield chunk_data.get("response", "")
                        except json.JSONDecodeError:
                            continue
            except httpx.HTTPError as exc:
                raise ModelProviderError("Ollama", 500, f"Local Ollama streaming connection failed: {str(exc)}")
