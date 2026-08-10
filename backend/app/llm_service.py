"""
LLM Service Module (Groq Provider)
Re-exports explanation LLM logic from llm.py:
- explain_prediction / generate_explanation
"""

from .llm import (
    LLMResult,
    explain_prediction,
    generate_explanation,
    SYSTEM_INSTRUCTIONS,
)

__all__ = [
    "LLMResult",
    "explain_prediction",
    "generate_explanation",
    "SYSTEM_INSTRUCTIONS",
]
