"""
Hadamard-LLM Core Engine
An ultra-lightweight, asynchronous LLM orchestration framework.
"""

from core.models.base import BaseModel
from core.models.openai import OpenAIModel
from core.models.anthropic import AnthropicModel
from core.models.gemini import GeminiModel
from core.models.ollama import OllamaModel
from core.models.groq import GroqModel
from core.models.deepseek import DeepSeekModel
from core.models.grok import GrokModel

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
