"""OpenRouter back-end.

OpenRouter speaks the OpenAI protocol, so this is the OpenAI client plus two
request fields that plain ``--base-url https://openrouter.ai/api/v1`` cannot set:

- ``reasoning``: many models it serves (DeepSeek's among them) reason by default,
  and at the tool's 250-token budget the reasoning can use all of it and leave
  no answer. It is switched off, for the reasons given in ``deepseek_client``.
- ``provider``: by default OpenRouter routes each request to whichever upstream
  it picks, and upstreams differ in price by up to ten times and in quantization.
  ``route`` pins the endpoints to use, in order, with no fallback - so every
  completion in a run comes from the same weights at a known price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .openai_client import OpenAIClient

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"


@dataclass
class OpenRouterClient(OpenAIClient):
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    route: List[str] = field(default_factory=list)

    provider = "openrouter"

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info["route"] = list(self.route)
        info["reasoning"] = False
        return info

    def _payload(self, system: str, user: str) -> Dict[str, Any]:
        payload = super()._payload(system, user)
        payload["reasoning"] = {"enabled": False}
        if self.route:
            payload["provider"] = {"order": list(self.route), "allow_fallbacks": False}
        return payload
