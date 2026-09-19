import os
import sys
import pytest

# Force override Python's lookup path matrix to include local folders directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if os.path.join(PARENT_DIR, "core") not in sys.path:
    sys.path.insert(0, os.path.join(PARENT_DIR, "core"))

from core.prompts.template import PromptTemplate
from core.exceptions import PromptTemplateError


def test_prompt_variable_extraction():
    """Tests if variables are accurately extracted while ignoring escaped double braces."""
    template_str = "Hello {name}, welcome to {project}. JSON format should be {{this_is_escaped}}."
    prompt = PromptTemplate(template_str)
    assert prompt.input_variables == {"name", "project"}


def test_successful_prompt_rendering():
    """Tests successful injection of variables into the template."""
    template_str = "Execute quantum protocol on wire {wire_id}."
    prompt = PromptTemplate(template_str)
    rendered = prompt.render(wire_id="3")
    assert rendered == "Execute quantum protocol on wire 3."


def test_missing_variable_raises_exception():
    """Tests if missing variables correctly trigger a PromptTemplateError."""
    template_str = "Compare {model_a} with {model_b}."
    prompt = PromptTemplate(template_str)
    
    with pytest.raises(PromptTemplateError) as exc_info:
        prompt.render(model_a="OpenAI")
        
    assert "Missing required variables" in str(exc_info.value)
