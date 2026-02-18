
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from ..interface import LLMProvider
from ...config import config

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        key = api_key or config.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY not found in config or env vars.")
        self.client = OpenAI(api_key=key)

    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.2, seed: Optional[int] = None) -> str:
        """Generate a text response."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_tokens=500  # Cost guardrail
        )
        return response.choices[0].message.content or ""

    def json(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.2, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate a structured JSON response."""
        # Enforce JSON mode
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                seed=seed,
                max_tokens=800  # Cost guardrail
            )
            content = response.choices[0].message.content
            if not content:
                return {}
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback handling or log error
            return {"error": "Failed to parse JSON", "raw": content}

    def classify(self, messages: List[Dict[str, str]], model: str) -> Dict[str, bool]:
        """Classify input (e.g., relevance check)."""
        # Expectation: The prompt should ask for a JSON boolean/classification
        # This is essentially a wrapper around json() but ensuring bool types if possible
        # For simplicity, we delegate to json() and assume the prompt ensures structure
        return self.json(messages, model, temperature=0.0)
