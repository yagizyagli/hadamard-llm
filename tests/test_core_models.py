import os
import sys
import pytest

# Explicit path enjection override
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Directly access the source files bypassing dynamic package lookup bottlenecks
from core.models.openai import OpenAIModel
from core.exceptions import MissingAPIKeyError


def test_openai_missing_api_key_raises_error():
    """Tests if initializing a model wrapper without any API key safely triggers our custom error."""
    cached_key = os.environ.pop("OPENAI_API_KEY", None)
    
    with pytest.raises(MissingAPIKeyError):
        OpenAIModel(api_key=None)
        
    if cached_key:
        os.environ["OPENAI_API_KEY"] = cached_key


def test_openai_payload_builder():
    """Tests if our internal message and parameter router constructs payloads correctly."""
    model = OpenAIModel(model_name="gpt-4o", temperature=0.2, api_key="mock-key")
    payload = model._build_payload(prompt="Run test.", system_instruction="Be a test runner.")
    
    assert payload["model"] == "gpt-4o"
    assert payload["temperature"] == 0.2
    assert len(payload["messages"]) == 2
    assert payload["messages"]["role"] == "system"
    assert payload["messages"]["role"] == "user"
