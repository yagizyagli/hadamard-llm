from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseMemory(ABC):
    """
    Abstract Base Class for all memory providers in Hadamard-LLM.
    Ensures a unified structural workflow for chat history management.
    """

    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the memory store."""
        pass

    @abstractmethod
    def get_messages(self) -> List[Dict[str, str]]:
        """Retrieves all stored messages formatted for LLM payloads."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears the chat history entirely."""
        pass
s
