"""A deterministic, offline stand-in for an LLM.

It exists so the whole pipeline — prompt generation, completion parsing, mutant
validation, test execution, reporting — can be exercised and tested without an
API key or a single token spent.  It applies crude textual rewrites to the
original fragment and formats them exactly like a real completion, including
some deliberately invalid suggestions so the discard path is exercised too.

It is *not* a substitute for a real model: it cannot suggest the kind of
context-aware, plausible-looking bugs that motivate the technique.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .base import LLMClient, LLMResponse

_FENCED = re.compile(r"```[ \t]*[A-Za-z0-9_+\-]*[ \t]*\r?\n(.*?)```", re.DOTALL)

_COMPARISONS = [
    (" == ", " != "),
    (" != ", " == "),
    (" >= ", " > "),
    (" <= ", " < "),
    (" > ", " >= "),
    (" < ", " <= "),
    (" is not ", " is "),
    (" is ", " is not "),
    (" not in ", " in "),
    (" in ", " not in "),
    (" and ", " or "),
    (" or ", " and "),
    (" + ", " - "),
    (" - ", " + "),
    (" * ", " / "),
    (" // ", " / "),
]

_CONSTANTS = ["None", "True", "False", "0", "-1", '""']


def _swap_operator(fragment: str) -> List[str]:
    results = []
    for needle, replacement in _COMPARISONS:
        if needle in fragment:
            results.append(fragment.replace(needle, replacement, 1))
    return results


def _tweak_numbers(fragment: str) -> List[str]:
    results = []
    for match in re.finditer(r"\b\d+\b", fragment):
        value = int(match.group())
        for candidate in (value + 1, value - 1, 0):
            if candidate == value:
                continue
            results.append(
                fragment[: match.start()] + str(candidate) + fragment[match.end() :]
            )
        break
    return results


def _negate(fragment: str) -> List[str]:
    stripped = fragment.strip()
    if not stripped or "\n" in stripped:
        return []
    if stripped.startswith("not "):
        return [stripped[4:]]
    return ["not ({})".format(stripped)]


def _rename_call(fragment: str) -> List[str]:
    match = re.match(r"^([A-Za-z_][A-Za-z_0-9.]*)$", fragment.strip())
    if not match:
        return []
    name = match.group(1)
    parts = name.split(".")
    swaps = {
        "min": "max", "max": "min", "len": "id", "sorted": "reversed",
        "append": "insert", "startswith": "endswith", "endswith": "startswith",
        "lower": "upper", "upper": "lower", "join": "split", "abs": "round",
        "get": "pop", "keys": "values", "values": "keys", "strip": "lstrip",
    }
    last = parts[-1]
    if last in swaps:
        parts[-1] = swaps[last]
        return [".".join(parts)]
    return []


def suggest(fragment: str, limit: int = 3) -> List[str]:
    """Produce up to ``limit`` distinct textual mutations of ``fragment``."""
    candidates: List[str] = []
    for producer in (_swap_operator, _rename_call, _tweak_numbers, _negate):
        for candidate in producer(fragment):
            text = " ".join(candidate.split())
            if text and text != fragment.strip() and text not in candidates:
                candidates.append(text)
            if len(candidates) >= limit:
                return candidates
    for constant in _CONSTANTS:
        if len(candidates) >= limit:
            break
        if constant != fragment.strip() and constant not in candidates:
            candidates.append(constant)
    return candidates[:limit]


@dataclass
class MockClient(LLMClient):
    """Formats :func:`suggest` output as if it came from a chat model."""

    provider = "mock"

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info["model"] = self.model or "mock"
        return info

    def _complete_once(self, system: str, user: str) -> LLMResponse:
        blocks = _FENCED.findall(user)
        fragment = blocks[1].strip() if len(blocks) > 1 else ""
        if not fragment and blocks:
            fragment = "None"
        options = suggest(fragment) or ["None"]
        parts = []
        for index, option in enumerate(options, start=1):
            parts.append(
                "Option {}: The PLACEHOLDER can be replaced with:\n"
                "```python\n{}\n```\n"
                "This would result in different behavior because it changes the "
                "evaluated expression.\n".format(index, option)
            )
        text = "\n".join(parts) + "\nDONE."
        return LLMResponse(
            text=text,
            prompt_tokens=len(user) // 4,
            completion_tokens=len(text) // 4,
        )
