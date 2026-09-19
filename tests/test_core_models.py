import os
import sys
import pytest

# Enforce project root injection before pulling framework modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
