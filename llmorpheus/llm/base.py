"""Shared plumbing for the LLM back-ends: retries, rate limiting, caching."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class LLMError(RuntimeError):
    """Raised when a completion could not be obtained."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False


@dataclass
class CompletionCache:
    """On-disk cache of completions, so re-runs do not re-pay for tokens."""

    directory: Optional[Path]

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / "{}.json".format(key)

    def get(self, key: str) -> Optional[LLMResponse]:
        if self.directory is None:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return LLMResponse(
            text=data.get("text", ""),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            cached=True,
        )

    def put(self, key: str, response: LLMResponse) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": response.text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }
        try:
            self._path(key).write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass


class RateLimiter:
    """Ensures at least ``interval_ms`` elapses between successive requests."""

    def __init__(self, interval_ms: int = 0) -> None:
        self.interval = max(0.0, interval_ms / 1000.0)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self.interval
        if sleep_for > 0:
            time.sleep(sleep_for)


@dataclass
class LLMClient:
    """Base class for chat-completion back-ends."""

    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 250
    attempts: int = 3
    rate_limit_ms: int = 0
    timeout: float = 120.0
    cache: CompletionCache = field(default_factory=lambda: CompletionCache(None))
    # Credentials read from .env. Deliberately absent from describe() - so never
    # part of the cache key or of mutants.json - and hidden from repr(), so a
    # traceback or debug print cannot reveal them.
    secrets: Dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    _limiter: RateLimiter = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    provider = "base"

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.rate_limit_ms)

    # -- public API --------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def preflight(self) -> None:
        """Fail fast on misconfiguration before any prompt is sent.

        Subclasses that need credentials should raise :class:`LLMError` here so
        the user sees one clear message rather than one per prompt.
        """
        return None

    def complete(self, system: str, user: str) -> LLMResponse:
        key = self._cache_key(system, user)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        response = self._complete_with_retries(system, user)
        self.cache.put(key, response)
        return response

    # -- to implement in subclasses ---------------------------------------
    def _complete_once(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def _cache_key(self, system: str, user: str) -> str:
        # ``describe()`` covers every setting that changes what the model returns
        # (provider, model, temperature, token budget, endpoint, thinking mode),
        # so toggling any of them can never be served a stale cached completion.
        payload = json.dumps(
            {"client": self.describe(), "system": system, "user": user},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _complete_with_retries(self, system: str, user: str) -> LLMResponse:
        last_error: Optional[BaseException] = None
        for attempt in range(1, max(1, self.attempts) + 1):
            self._limiter.wait()
            try:
                return self._complete_once(system, user)
            except LLMError as error:
                last_error = error
                if not getattr(error, "retryable", False) or attempt == self.attempts:
                    raise
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt == self.attempts:
                    raise LLMError("request failed after {} attempts: {}".format(attempt, error))
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
        raise LLMError("request failed: {}".format(last_error))

    def _post_json(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        for name, value in headers.items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = ""
            try:
                detail = error.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # pragma: no cover - best effort
                pass
            failure = LLMError("HTTP {} from {}: {}".format(error.code, url, detail))
            setattr(failure, "retryable", error.code in RETRYABLE_STATUS)
            setattr(failure, "status", error.code)
            raise failure
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            # Gateways and proxies happily answer 200 with an HTML error page.
            # Treat it as a retryable transport failure rather than letting a
            # ValueError escape and abort a run of thousands of prompts.
            preview = raw[:200].decode("utf-8", errors="replace")
            failure = LLMError("malformed (non-JSON) response from {}: {!r}".format(url, preview))
            setattr(failure, "retryable", True)
            raise failure


def api_key_from_env(
    env_var: str, provider: str, fallback: Optional[Mapping[str, str]] = None
) -> str:
    """Read a key from the environment, else from ``fallback`` (values from .env)."""
    key = os.environ.get(env_var, "").strip() or (fallback or {}).get(env_var, "").strip()
    if not key:
        raise LLMError(
            "no API key found: set {} in .env or in the environment to use the {} "
            "provider".format(env_var, provider)
        )
    return key
