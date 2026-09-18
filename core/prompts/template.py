from typing import Any, Dict, Set
import re
from core.exceptions import PromptTemplateError


class PromptTemplate:
    """
    A lightweight, blazing-fast prompt template engine that extracts variables
    and renders prompts safely using Python's native string formatting.
    """

    def __init__(self, template: str):
        self.template = template
        self.input_variables = self._extract_variables(template)

    def _extract_variables(self, template: str) -> Set[str]:
        """
        Extracts all variables enclosed in curly braces {variable_name}.
        Uses regex to safely find double-brace escapes like {{json_format}}.
        """
        # Finds all single curly brace variables but ignores double curly braces
        pattern = r"(?<!{){([a-zA-Z0-9_]+)}(?!})"
        return set(re.findall(pattern, template))

    def render(self, **kwargs: Any) -> str:
        """
        Injects the given variables into the template string.
        Raises PromptTemplateError if any required variable is missing.
        """
        missing_vars = self.input_variables - set(kwargs.keys())
        if missing_vars:
            raise PromptTemplateError(
                f"Missing required variables for prompt rendering: {missing_vars}. "
                f"Expected: {list(self.input_variables)}"
            )

        try:
            return self.template.format(**kwargs)
        except KeyError as exc:
            raise PromptTemplateError(f"Failed to render prompt template due to key mismatch: {str(exc)}")
