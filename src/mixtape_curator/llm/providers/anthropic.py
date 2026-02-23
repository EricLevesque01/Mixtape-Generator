"""
Anthropic Claude LLM provider.
Implements the LLMProvider protocol using the anthropic SDK.
"""
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("mixtape_curator.llm.anthropic")

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from ..interface import LLMProvider


class AnthropicProvider(LLMProvider):
    """LLM provider using Anthropic Claude models."""
    
    def __init__(self, api_key: str):
        if not HAS_ANTHROPIC:
            raise ImportError(
                "The 'anthropic' package is required. Install with: pip install anthropic"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, messages: List[Dict[str, str]], model: str = "claude-sonnet-4-20250514",
                 temperature: float = 0.2, seed: Optional[int] = None) -> str:
        """Generate a text response using Claude."""
        # Convert messages to Anthropic format
        system_msg = None
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append(m)
        
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg
        
        try:
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generate error: {e}")
            raise
    
    def json(self, messages: List[Dict[str, str]], model: str = "claude-sonnet-4-20250514",
             temperature: float = 0.2, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate a structured JSON response using Claude."""
        # Append JSON instruction to the last message
        enhanced_messages = messages.copy()
        if enhanced_messages:
            last = enhanced_messages[-1].copy()
            last["content"] = last["content"] + "\n\nRespond with valid JSON only. No markdown, no explanation."
            enhanced_messages[-1] = last
        
        text = self.generate(enhanced_messages, model=model, temperature=temperature, seed=seed)
        
        # Parse JSON from response (handle potential markdown fencing)
        text = text.strip()
        if text.startswith("```"):
            # Remove markdown code fences
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from Anthropic response: {e}")
            logger.debug(f"Raw response: {text[:500]}")
            return {"error": f"JSON parse error: {e}", "raw": text[:500]}
    
    def classify(self, messages: List[Dict[str, str]], model: str = "claude-sonnet-4-20250514") -> Dict[str, bool]:
        """Classify input using Claude."""
        result = self.json(messages, model=model)
        # Ensure boolean values
        return {k: bool(v) for k, v in result.items() if isinstance(v, bool)}
