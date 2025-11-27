"""Lightweight Anthropic Claude Sonnet 4.5 client wrapper.

This module implements a minimal call to the Anthropic completion API
and a safe mocked fallback when `ANTHROPIC_API_KEY` is not available.

Do not commit your API key into source control; set the `ANTHROPIC_API_KEY`
environment variable in your runtime environment.
"""
import os
import time
import json
import requests

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
DEFAULT_MODEL = os.environ.get('DEFAULT_LLM', 'claude-sonnet-4.5')
API_URL = 'https://api.anthropic.com/v1/complete'


class AnthropicError(RuntimeError):
    pass


def generate(prompt: str, model: str = None, max_tokens: int = 512, stop_sequences=None, temperature: float = 0.0):
    """Generate text using Anthropic API or return a mocked response when no key.

    Returns a dict: {'success': bool, 'model': model, 'text': str}
    """
    model = model or DEFAULT_MODEL
    stop_sequences = stop_sequences or ["\nHuman:", "\nAssistant:"]

    if not ANTHROPIC_API_KEY:
        # Mock fallback for local testing
        return {
            'success': True,
            'model': model,
            'text': f"[mocked response for model={model}] {prompt[:200]}"
        }

    headers = {
        'Authorization': f'Bearer {ANTHROPIC_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': model,
        'prompt': prompt,
        'max_tokens_to_sample': int(max_tokens),
        'temperature': float(temperature),
        'stop_sequences': stop_sequences,
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        raise AnthropicError(f'network error: {e}')

    if r.status_code != 200:
        raise AnthropicError(f'Anthropic API error: {r.status_code} {r.text}')

    try:
        j = r.json()
    except Exception as e:
        raise AnthropicError(f'invalid json response: {e} - {r.text[:200]}')

    # The response shape may include 'completion' or 'response' depending on API versions.
    text = j.get('completion') or j.get('response') or j.get('completion_text') or j.get('text')
    if text is None:
        # best effort to extract string fields
        for k in ('completion', 'response', 'text'):
            if k in j and isinstance(j[k], str):
                text = j[k]
                break
    if text is None:
        # fallback to stringify the whole response
        text = json.dumps(j)[:1000]

    return {'success': True, 'model': model, 'text': text}
