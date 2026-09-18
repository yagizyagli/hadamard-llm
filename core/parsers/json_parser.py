import json
import re
from typing import Any, Dict, Optional


class SimpleJsonParser:
    """
    A professional, zero-dependency output parser that isolates, cleans, 
    and decodes JSON data hidden within raw LLM string responses.
    """

    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        """
        Extracts JSON from markdown code blocks or raw text boundaries,
        cleans trailing anomalies, and converts it into a Python dictionary.
        """
        if not text:
            return {}

        clean_text = text.strip()

        # Regex to strip markdown code blocks like ```json ... ``` or ``` ... ```
        markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(markdown_pattern, clean_text)
        
        if match:
            clean_text = match.group(1).strip()

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as exc:
            # Fallback strategy: Try to extract everything between first '{' and last '}'
            start_idx = clean_text.find("{")
            end_idx = clean_text.rfind("}")
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                nested_json = clean_text[start_idx : end_idx + 1]
                try:
                    return json.loads(nested_json)
                except json.JSONDecodeError:
                    pass
            
            raise ValueError(f"Failed to cleanly parse LLM response into a valid JSON object. Raw payload: {text}") from exc
