"""Extracting mutants from LLM completions, and validating them."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..analysis.locations import Location
from ..analysis.positions import Position, SourceMap
from ..prompting.prompts import PLACEHOLDER

# Fenced code blocks, with or without a language tag, spanning one or more lines.
_FENCED_BLOCK = re.compile(r"```[ \t]*[A-Za-z0-9_+\-]*[ \t]*\r?\n(.*?)```", re.DOTALL)
_INLINE_BLOCK = re.compile(r"```[ \t]*([^`\n]+?)[ \t]*```")
_EXPLANATION = re.compile(
    r"(?:this\s+would\s+result\s+in\s+different\s+behavio(?:u)?r\s+because\s+)(.+?)(?:\n\n|\n(?=Option)|$)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Mutant:
    """A concrete, syntactically valid replacement for one placeholder."""

    id: str
    file: str
    kind: str
    start: Position
    end: Position
    original: str
    replacement: str
    explanation: str = ""
    prompt_id: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "kind": self.kind,
            "start": self.start.as_dict(),
            "end": self.end.as_dict(),
            "original": self.original,
            "replacement": self.replacement,
            "explanation": self.explanation,
            "prompt_id": self.prompt_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Mutant":
        return cls(
            id=data["id"],
            file=data["file"],
            kind=data["kind"],
            start=Position.from_dict(data["start"]),
            end=Position.from_dict(data["end"]),
            original=data["original"],
            replacement=data["replacement"],
            explanation=data.get("explanation", ""),
            prompt_id=data.get("prompt_id", ""),
        )

    def apply(self, source: str) -> str:
        """Return ``source`` with this mutant's replacement spliced in."""
        return SourceMap(source).replace(self.start, self.end, self.replacement)

    def describe(self) -> str:
        return "{}:{} [{}] {!r} -> {!r}".format(
            self.file, self.start, self.kind, self.original, self.replacement
        )


@dataclass
class GenerationStats:
    """Counts mirroring Table 2 of the paper."""

    prompts: int = 0
    completions: int = 0
    failed_prompts: int = 0
    candidates: int = 0
    invalid: int = 0
    identical: int = 0
    duplicate: int = 0
    mutants: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CandidateFilter:
    """Applies the paper's discard rules to candidate replacements.

    Candidates are rejected when they are syntactically invalid, identical to
    the original fragment, or duplicates of an already accepted mutant.  Unlike
    the JavaScript implementation — which can parse a fragment on its own — a
    Python fragment is only meaningful in context, so validity is checked by
    splicing the candidate into the whole file and parsing that.
    """

    source: str
    file: str
    allow_multiline: bool = False
    max_replacement_chars: int = 400
    _accepted: Dict[Tuple[int, int, int, int, str], None] = field(default_factory=dict)

    def _key(self, location: Location, replacement: str) -> Tuple[int, int, int, int, str]:
        return (
            location.start.line,
            location.start.column,
            location.end.line,
            location.end.column,
            replacement,
        )

    def check(self, location: Location, replacement: str) -> Tuple[bool, Optional[str]]:
        """Return ``(accepted, reason_rejected)``."""
        text = replacement.strip()
        if not text:
            return False, "invalid"
        if len(text) > self.max_replacement_chars:
            return False, "invalid"
        if PLACEHOLDER in text or "PLACEHOLDER" in text:
            return False, "invalid"
        if "\n" in text and not self.allow_multiline:
            return False, "invalid"
        if text == location.original.strip():
            return False, "identical"
        key = self._key(location, text)
        if key in self._accepted:
            return False, "duplicate"
        if not is_syntactically_valid(self.source, location, text):
            return False, "invalid"
        self._accepted[key] = None
        return True, None


def is_syntactically_valid(source: str, location: Location, replacement: str) -> bool:
    """Does splicing ``replacement`` into ``source`` still produce parsable Python?"""
    try:
        mutated = SourceMap(source).replace(location.start, location.end, replacement)
    except Exception:  # pragma: no cover - defensive
        return False
    try:
        ast.parse(mutated, filename=location.file)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True


def extract_code_blocks(completion: str) -> List[str]:
    """Pull the fenced code blocks out of an LLM completion, in order."""
    blocks = [block.strip("\n").strip() for block in _FENCED_BLOCK.findall(completion)]
    if not blocks:
        blocks = [block.strip() for block in _INLINE_BLOCK.findall(completion)]
    cleaned: List[str] = []
    for block in blocks:
        text = block.strip()
        if not text:
            continue
        # Some models restate the language tag on its own first line.
        lines = text.split("\n")
        if len(lines) > 1 and lines[0].strip().lower() in ("python", "py"):
            text = "\n".join(lines[1:]).strip()
        cleaned.append(text)
    return cleaned


def extract_explanations(completion: str) -> List[str]:
    return [" ".join(match.split()) for match in _EXPLANATION.findall(completion)]


def mutant_id(file: str, location: Location, replacement: str) -> str:
    digest = hashlib.sha1(
        "{}|{}|{}|{}|{}".format(
            file, location.kind, location.start, location.end, replacement
        ).encode("utf-8")
    ).hexdigest()[:12]
    return digest


def mutants_from_completion(
    completion: str,
    location: Location,
    candidate_filter: CandidateFilter,
    stats: GenerationStats,
) -> List[Mutant]:
    """Parse a completion into accepted mutants, updating ``stats`` in place."""
    blocks = extract_code_blocks(completion)
    explanations = extract_explanations(completion)
    accepted: List[Mutant] = []
    for index, block in enumerate(blocks):
        stats.candidates += 1
        ok, reason = candidate_filter.check(location, block)
        if not ok:
            if reason == "invalid":
                stats.invalid += 1
            elif reason == "identical":
                stats.identical += 1
            elif reason == "duplicate":
                stats.duplicate += 1
            continue
        replacement = block.strip()
        accepted.append(
            Mutant(
                id=mutant_id(location.file, location, replacement),
                file=location.file,
                kind=location.kind,
                start=location.start,
                end=location.end,
                original=location.original,
                replacement=replacement,
                explanation=explanations[index] if index < len(explanations) else "",
                prompt_id=location.id,
            )
        )
    stats.mutants += len(accepted)
    return accepted


@dataclass
class MutantSet:
    """The ``mutants.json`` payload: everything ``llmorpheus run`` needs."""

    mutants: List[Mutant]
    stats: GenerationStats
    meta: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "meta": self.meta,
            "stats": self.stats.as_dict(),
            "mutants": [mutant.as_dict() for mutant in self.mutants],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "MutantSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        stats = GenerationStats(**{
            key: value
            for key, value in data.get("stats", {}).items()
            if key in GenerationStats().__dict__
        })
        return cls(
            mutants=[Mutant.from_dict(item) for item in data.get("mutants", [])],
            stats=stats,
            meta=data.get("meta", {}),
        )

    def by_file(self) -> Dict[str, List[Mutant]]:
        grouped: Dict[str, List[Mutant]] = {}
        for mutant in self.mutants:
            grouped.setdefault(mutant.file, []).append(mutant)
        return grouped


def group_by_kind(mutants: Iterable[Mutant]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for mutant in mutants:
        counts[mutant.kind] = counts.get(mutant.kind, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def file_digest(source: str) -> str:
    """Fingerprint a source file, so stale mutants can be detected at run time.

    A mutant records absolute line/column offsets, which are only meaningful
    against the exact text they were generated from.
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
