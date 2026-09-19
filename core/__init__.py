"""
Hadamard-LLM Core Engine
An ultra-lightweight, asynchronous LLM orchestration framework.
"""

from core.models import (
    BaseModel,
    OpenAIModel,
    AnthropicModel,
    GeminiModel,
    OllamaModel,
    GroqModel,
    DeepSeekModel,
    GrokModel,
)

from core.prompts.template import PromptTemplate
from core.memory.window import BufferWindowMemory
from core.parsers.json_parser import SimpleJsonParser

__version__ = "0.1.0"

__all__ = [
    "BaseModel",
    "OpenAIModel",
    "AnthropicModel",
    "GeminiModel",
    "OllamaModel",
    "GroqModel",
    "DeepSeekModel",
    "GrokModel",
    "PromptTemplate",
    "BufferWindowMemory",
    "SimpleJsonParser",
]
