from typing import List, Dict, Any, Optional, Protocol

class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.2, seed: Optional[int] = None) -> str:
        """Generate a text response."""
        ...

    def json(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.2, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate a structured JSON response."""
        ...

    def classify(self, messages: List[Dict[str, str]], model: str) -> Dict[str, bool]:
        """Classify input (e.g., relevance check)."""
        ...
