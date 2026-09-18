import os
from typing import Any, Dict, List, Optional, AsyncGenerator
import json
import httpx
from core.models.base import BaseModel
from core.exceptions import MissingAPIKeyError, ModelProviderError

class GeminiModel(BaseModel):
    """
    Ultra-lightweight, zero-official-sdk Google Gemini model implementation
    using raw high-performance HTTPX async requests.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.7, api_key: Optional[str] = None, **kwargs: Any):
        super().__init__(model_name, temperature, **kwargs)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("Gemini API Key not found. Set GEMINI_API_KEY env var or pass it explicitly.")
        # Google Gemini uses api_key as a query parameter rather than a header
        self.base_url = f"https://googleapis.com{self.model_name}"

    def _build_payload(self, prompt: str, system_instruction: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        contents = [{"parts": [{"text": prompt}]}]
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                **self.config,
                **kwargs
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
            
        return payload

    async def generate(self, prompt: str, system_instruction: Optional[str] = None, **kwargs: Any) -> str:
        url = f"{self.base_url}:generateContent?key={self.api_key}"
        payload = self._build_payload(prompt, system_instruction, **kwargs)
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=60.0)
                if response.status_code != 200:
                    raise ModelProviderError("Gemini", response.status_code, response.text)
                
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                raise ModelProviderError("Gemini", 500, f"Request or parsing failed: {str(exc)}")

    async def generate_stream(self, prompt: str, system_instruction: Optional[str] = None, **kwargs: Any) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}:streamGenerateContent?key={self.api_key}"
        payload = self._build_payload(prompt, system_instruction, **kwargs)
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", url, json=payload, headers=headers, timeout=60.0) as response:
                    if response.status_code != 200:
                        raise ModelProviderError("Gemini", response.status_code, await response.aread())
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        # Gemini stream items can arrive wrapped inside a JSON array chunk
                        clean_line = line.strip().lstrip(",[").rstrip(",]")
                        if not clean_line:
                            continue
                        try:
                            chunk_data = json.loads(clean_line)
                            text_chunk = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                            yield text_chunk
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
            except httpx.HTTPError as exc:
                raise ModelProviderError("Gemini", 500, f"HTTP Streaming Request failed: {str(exc)}")
