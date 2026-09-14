"""OpenAI-compatible chat-completions back-end.

Also covers OpenRouter, Together, Groq, vLLM, LM Studio and Ollama — anything
that speaks ``POST /v1/chat/completions`` — by pointing ``--base-url`` at it.
The paper used exactly this kind of mix (openai.com, openrouter.ai, octo.ai).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .base import LLMClient, LLMError, LLMResponse, api_key_from_env

DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass
class OpenAIClient(LLMClient):
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "OPENAI_API_KEY"
    require_api_key: bool = True
    extra_headers: Dict[str, str] = None  # type: ignore[assignment]

    provider = "openai"

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info["base_url"] = self.base_url
        return info

    def preflight(self) -> None:
        if self.require_api_key:
            api_key_from_env(self.api_key_env, self.provider, self.secrets)

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        try:
            headers["Authorization"] = "Bearer {}".format(
                api_key_from_env(self.api_key_env, self.provider, self.secrets)
            )
        except LLMError:
            # Local servers (Ollama, LM Studio, vLLM) usually need no key.
            if self.require_api_key:
                raise
        for name, value in (self.extra_headers or {}).items():
            headers[name] = value
        return headers

    def _payload(self, system: str, user: str) -> Dict[str, Any]:
        """The request body; subclasses add provider-specific fields here."""
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _complete_once(self, system: str, user: str) -> LLMResponse:
        url = "{}/chat/completions".format(self.base_url.rstrip("/"))
        data = self._post_json(url, self._headers(), self._payload(system, user))
        if not isinstance(data, dict):
            raise LLMError("unexpected response shape: {}".format(str(data)[:300]))
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise LLMError("no usable choices in response: {}".format(str(data)[:300]))
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise LLMError("no message in the first choice: {}".format(str(data)[:300]))
        text = message.get("content") or ""
        if not text.strip() and message.get("reasoning_content"):
            # Reasoning models (DeepSeek, and others using this field) can spend
            # the entire token budget thinking and return no answer at all.
            raise LLMError(
                "the model used its whole {}-token budget on reasoning and returned "
                "no answer (finish_reason={!r}); raise --max-tokens or turn thinking "
                "off".format(self.max_tokens, choices[0].get("finish_reason"))
            )
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )
