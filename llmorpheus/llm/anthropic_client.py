"""Anthropic Messages API back-end."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .base import LLMClient, LLMError, LLMResponse, api_key_from_env

DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"


@dataclass
class AnthropicClient(LLMClient):
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "ANTHROPIC_API_KEY"

    provider = "anthropic"

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info["base_url"] = self.base_url
        return info

    def preflight(self) -> None:
        api_key_from_env(self.api_key_env, self.provider, self.secrets)

    def _complete_once(self, system: str, user: str) -> LLMResponse:
        url = "{}/messages".format(self.base_url.rstrip("/"))
        headers = {
            "x-api-key": api_key_from_env(self.api_key_env, self.provider, self.secrets),
            "anthropic-version": API_VERSION,
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = self._post_json(url, headers, payload)
        if not isinstance(data, dict):
            raise LLMError("unexpected response shape: {}".format(str(data)[:300]))
        blocks = data.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            raise LLMError("no text content in response: {}".format(str(data)[:300]))
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
        )
