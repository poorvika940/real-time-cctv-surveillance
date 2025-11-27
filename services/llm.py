"""Small LLM selection layer. Provides a generate(prompt) function and
an in-process runtime toggle for the default model.
"""
import os
from typing import Optional

DEFAULT_LLM = os.environ.get('DEFAULT_LLM', 'claude-sonnet-4.5')
_CURRENT_LLM = DEFAULT_LLM


def get_default_llm():
    return _CURRENT_LLM


def set_default_llm(model_name: str):
    global _CURRENT_LLM
    _CURRENT_LLM = model_name
    return _CURRENT_LLM


# Lazy import of provider wrappers to avoid heavy deps at import time
def generate(prompt: str, model: Optional[str] = None, **opts):
    """Generate text using the current default provider.

    If `model` is provided, it overrides the current default.
    """
    model_to_use = model or _CURRENT_LLM
    # Only Anthropic is implemented for now (claude-sonnet-4.5)
    if model_to_use and model_to_use.startswith('claude'):
        from .anthropic_client import generate as _anthropic_generate
        return _anthropic_generate(prompt, model=model_to_use, **opts)

    # unknown model -> return a safe mocked response
    return {'success': True, 'model': model_to_use, 'text': f'[mocked:{model_to_use}] {prompt[:200]}'}
