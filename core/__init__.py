"""
Hadamard-LLM Core Engine
An ultra-lightweight, asynchronous LLM orchestration framework.
"""

from .models import (
    BaseModel,
    OpenAIModel,
    AnthropicModel,
    GeminiModel,
    OllamaModel,
    GroqModel,
    DeepSeekModel,
    GrokModel,
)

from .prompts.template import PromptTemplate
from .memory.window import BufferWindowMemory
from .parsers.json_parser import SimpleJsonParser

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
