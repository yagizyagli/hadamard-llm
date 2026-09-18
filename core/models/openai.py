import os
from typing import Any, Dict, List, Optional, AsyncGenerator
import httpx
from core.models.base import BaseModel
from core.exceptions import MissingAPIKeyError, ModelProviderError

class OpenAIModel(BaseModel):
    """
    Ultra-lightweight, zero-official-sdk OpenAI model implementation
    using raw high-performance HTTPX async requests.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7, api_key: Optional[str] = None, **kwargs: Any):
        super().__init__(model_name, temperature, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("OpenAI API Key not found. Set OPENAI_API_KEY env var or pass it explicitly.")
        self.base_url = "https://openai.com"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _build_payload(self, prompt: str, system_instruction: Optional[str] = None, stream: bool = False, **kwargs: Any) -> Dict[str, Any]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
            **self.config,
            **kwargs
        }
        return payload

    async def generate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs: Any) -> str:
        headers = self._build_headers()
        payload = self._build_payload(prompt, system_instruction, stream=False, **kwargs)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.base_url, json=payload, headers=headers, timeout=60.0)
                if response.status_code != 200:
                    raise ModelProviderError("OpenAI", response.status_code, response.text)
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPError as exc:
                raise ModelProviderError("OpenAI", 500, f"HTTP Request failed: {str(exc)}")

    async def generate_stream(self, prompt: str, system_instruction: Optional[str] = None, **kwargs: Any) -> AsyncGenerator[str, None]:
        headers = self._build_headers()
        payload = self._build_payload(prompt, system_instruction, stream=True, **kwargs)

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", self.base_url, json=payload, headers=headers, timeout=60.0) as response:
                    if response.status_code != 200:
                        raise ModelProviderError("OpenAI", response.status_code, await response.aread())
                    
                    async for line in response.aiter_lines():
                        if not line.strip() or line.strip() == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                import json
                                json_data = json.loads(line[6:])
                                delta = json_data["choices"][0]["delta"]
                                if "content" in delta:
                                    yield delta["content"]
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
            except httpx.HTTPError as exc:
                raise ModelProviderError("OpenAI", 500, f"HTTP Streaming Request failed: {str(exc)}")
