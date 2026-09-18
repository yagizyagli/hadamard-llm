from typing import Dict, List
from core.memory.base import BaseMemory


class BufferWindowMemory(BaseMemory):
    """
    A short-term sliding window memory that retains only the last 'k' interactions.
    Prevents token bloat and keeps context windows highly optimized.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Appends a message chunk. If history exceeds 2*k (turns),
        evicts the oldest interaction slice.
        """
        self.history.append({"role": role, "content": content})
        
        # k represents conversation turns (1 turn = 1 user + 1 assistant message = 2 messages)
        max_messages = self.k * 2
        if len(self.history) > max_messages:
            # Drop the oldest turn slice safely
            self.history = self.history[-max_messages:]

    def get_messages(self) -> List[Dict[str, str]]:
        """Returns the sliding window payload for the models."""
        return self.history

    def clear(self) -> None:
        """Resets the internal history sequence."""
        self.history = []
