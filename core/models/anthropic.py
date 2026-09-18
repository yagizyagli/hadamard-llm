import os
from typing import Any, Dict, List, Optional, AsyncGenerator
import json
import httpx
from core.models.base import BaseModel
from core.exceptions import MissingAPIKeyError, ModelProviderError

class AnthropicModel(BaseModel):
    """
    Ultra-lightweight, zero-official-sdk Anthropic Claude model implementation
    using raw high-performance HTTPX async requests.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet-latest", temperature: float = 0.7, api_key: Optional[str] = None, **kwargs: Any):
        super().__init__(model_name, temperature, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("Anthropic API Key not found. Set ANTHROPIC_API_KEY env var or pass it explicitly.")
        self.base_url = "https://anthropic.com"
        self.anthropic_version = "2023-06-01"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json"
        }

    def _build_payload(self, prompt: str, system_instruction: Optional[str] = None, stream: bool = False, **kwargs: Any) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "stream": stream,
            **self.config,
            **kwargs
        }
        if system_instruction:
            payload["system"] = system_instruction
        return payload

    async def generate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs: Any) -> str:
        headers = self._build_headers()
        payload = self._build_payload(prompt, system_instruction, stream=False, **kwargs)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.base_url, json=payload, headers=headers, timeout=60.0)
                if response.status_code != 200:
                    raise ModelProviderError("Anthropic", response.status_code, response.text)
                
                data = response.json()
                return data["content"][0]["text"]
            except httpx.HTTPError as exc:
                raise ModelProviderError("Anthropic", 500, f"HTTP Request failed: {str(exc)}")

    async def generate_stream(self, prompt: str, system_instruction: Optional[str] = None, **kwargs: Any) -> AsyncGenerator[str, None]:
        headers = self._build_headers()
        payload = self._build_payload(prompt, system_instruction, stream=True, **kwargs)

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", self.base_url, json=payload, headers=headers, timeout=60.0) as response:
                    if response.status_code != 200:
                        raise ModelProviderError("Anthropic", response.status_code, await response.aread())
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            try:
                                event_data = json.loads(line[6:])
                                if event_data.get("type") == "content_block_delta":
                                    delta = event_data.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        yield delta.get("text", "")
                            except (json.JSONDecodeError, KeyError):
                                continue
            except httpx.HTTPError as exc:
                raise ModelProviderError("Anthropic", 500, f"HTTP Streaming Request failed: {str(exc)}")
