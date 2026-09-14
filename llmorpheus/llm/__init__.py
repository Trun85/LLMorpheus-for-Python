"""LLM back-ends used to fill placeholders."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from .anthropic_client import AnthropicClient
from .base import CompletionCache, LLMClient, LLMError, LLMResponse
from .deepseek_client import DeepSeekClient
from .mock import MockClient
from .openai_client import OpenAIClient

PROVIDERS = ("openai", "anthropic", "deepseek", "mock")

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
    "deepseek": "deepseek-flash",
    "mock": "mock",
}

# Output budget when thinking is on: the reasoning text shares the budget with
# the answer, and 250 tokens is routinely exhausted before any answer appears.
THINKING_MAX_TOKENS = 4096


def build_client(
    provider: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 250,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    attempts: int = 3,
    rate_limit_ms: int = 0,
    timeout: float = 120.0,
    cache_dir: Optional[Path] = None,
    thinking: bool = False,
    secrets: Optional[Mapping[str, str]] = None,
) -> LLMClient:
    """Construct an :class:`LLMClient` from CLI-level options."""
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(
            "unknown provider {!r}; choose one of: {}".format(provider, ", ".join(PROVIDERS))
        )
    if thinking and provider != "deepseek":
        raise ValueError("--thinking is only supported by the deepseek provider")
    cache = CompletionCache(cache_dir)
    common = dict(
        model=model or DEFAULT_MODELS[provider],
        temperature=temperature,
        max_tokens=max_tokens,
        attempts=attempts,
        rate_limit_ms=rate_limit_ms,
        timeout=timeout,
        cache=cache,
        secrets=dict(secrets or {}),
    )
    if provider == "mock":
        return MockClient(**common)
    if provider == "anthropic":
        client = AnthropicClient(**common)
        if base_url:
            client.base_url = base_url
        if api_key_env:
            client.api_key_env = api_key_env
        return client
    if provider == "deepseek":
        # A custom base URL here is still DeepSeek (or a proxy for it), so the
        # key stays mandatory - unlike a bare OpenAI-compatible local server.
        deepseek = DeepSeekClient(**common, thinking=thinking)
        if base_url:
            deepseek.base_url = base_url
        if api_key_env:
            deepseek.api_key_env = api_key_env
        return deepseek
    client = OpenAIClient(**common)
    if base_url:
        client.base_url = base_url
        # A custom endpoint is usually a local server that needs no credentials.
        client.require_api_key = False
    if api_key_env:
        client.api_key_env = api_key_env
        client.require_api_key = True
    return client


__all__ = [
    "AnthropicClient",
    "CompletionCache",
    "DeepSeekClient",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "MockClient",
    "OpenAIClient",
    "PROVIDERS",
    "THINKING_MAX_TOKENS",
    "build_client",
]
