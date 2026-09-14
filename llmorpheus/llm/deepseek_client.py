"""DeepSeek back-end.

DeepSeek speaks the OpenAI chat-completions protocol, so this is the OpenAI
client with DeepSeek's endpoint, key variable and default model — plus explicit
control over *thinking mode*, which ``deepseek-flash`` turns on by default.

Thinking is switched **off** by default here, for three reasons that matter for
mutation testing:

- in thinking mode DeepSeek ignores ``temperature``, so the paper's temperature
  experiments (RQ3) would silently measure nothing;
- the reasoning text is drawn from the same output budget, and at the tool's
  default of 250 tokens it can consume all of it and leave no answer;
- a placeholder replacement is a one-line answer that does not benefit from
  lengthy reasoning, while every reasoning token is billed.

Pass ``--thinking`` to turn it on; the default token budget is raised to match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .openai_client import OpenAIClient

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


@dataclass
class DeepSeekClient(OpenAIClient):
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "DEEPSEEK_API_KEY"
    thinking: bool = False

    provider = "deepseek"

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info["thinking"] = self.thinking
        return info

    def _payload(self, system: str, user: str) -> Dict[str, Any]:
        payload = super()._payload(system, user)
        payload["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
        if self.thinking:
            # Accepted but ignored in thinking mode; omitting it avoids implying
            # that the temperature setting had any effect on the result.
            payload.pop("temperature", None)
        return payload
