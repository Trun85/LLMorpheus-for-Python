"""Turning placeholder locations into LLM prompts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..analysis.locations import Location
from ..analysis.positions import Position, SourceMap

PLACEHOLDER = "<PLACEHOLDER>"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

DEFAULT_TEMPLATE = "template-full"
DEFAULT_SYSTEM_TEMPLATE = "system-mutation-testing-expert"


@dataclass
class Prompt:
    """One instantiated prompt, ready to be sent to an LLM."""

    id: str
    location: Location
    system: str
    text: str
    template: str
    system_template: str
    context_first_line: int
    context_last_line: int

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "location": self.location.as_dict(),
            "template": self.template,
            "system_template": self.system_template,
            "context": {"first_line": self.context_first_line, "last_line": self.context_last_line},
            "system": self.system,
            "text": self.text,
        }


def available_templates() -> List[str]:
    return sorted(
        path.stem for path in TEMPLATES_DIR.glob("template-*.txt")
    )


def available_system_templates() -> List[str]:
    return sorted(path.stem for path in TEMPLATES_DIR.glob("system-*.txt"))


def load_template(name: str) -> str:
    """Load a template by bundled name or by path."""
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    for filename in (name, "{}.txt".format(name)):
        bundled = TEMPLATES_DIR / filename
        if bundled.is_file():
            return bundled.read_text(encoding="utf-8")
    raise FileNotFoundError(
        "no template named {!r}; bundled templates: {}".format(
            name, ", ".join(available_templates() + available_system_templates())
        )
    )


def render(template: str, variables: Dict[str, str]) -> str:
    """Substitute ``{{name}}`` placeholders.

    Deliberately not a general template engine: values are inserted verbatim so
    that source code containing braces is never re-interpreted.
    """
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


def build_code_with_placeholder(
    source_map: SourceMap,
    location: Location,
    max_context_lines: int,
) -> "tuple[str, int, int]":
    """Return the prompt's code window with the location replaced by a placeholder."""
    first, last = source_map.window(location.start, location.end, max_context_lines)
    window_offset = source_map.offset(Position(first, 0))
    window_text = source_map.line_text(first, last)
    relative_start = source_map.offset(location.start) - window_offset
    relative_end = source_map.offset(location.end) - window_offset
    relative_start = max(0, min(relative_start, len(window_text)))
    relative_end = max(relative_start, min(relative_end, len(window_text)))
    code = window_text[:relative_start] + PLACEHOLDER + window_text[relative_end:]
    return code.rstrip("\n"), first, last


def build_prompt(
    source_map: SourceMap,
    location: Location,
    template_text: str,
    system_text: str,
    template_name: str = DEFAULT_TEMPLATE,
    system_template_name: str = DEFAULT_SYSTEM_TEMPLATE,
    max_context_lines: int = 200,
) -> Prompt:
    code, first, last = build_code_with_placeholder(source_map, location, max_context_lines)
    text = render(template_text, {"code": code, "orig": location.original})
    return Prompt(
        id=location.id,
        location=location,
        system=system_text.strip(),
        text=text,
        template=template_name,
        system_template=system_template_name,
        context_first_line=first,
        context_last_line=last,
    )


def build_prompts(
    source: str,
    locations: Sequence[Location],
    template_name: str = DEFAULT_TEMPLATE,
    system_template_name: str = DEFAULT_SYSTEM_TEMPLATE,
    max_context_lines: int = 200,
    template_text: Optional[str] = None,
    system_text: Optional[str] = None,
) -> List[Prompt]:
    """Build one prompt per location, all against the same source file."""
    source_map = SourceMap(source)
    template_body = template_text if template_text is not None else load_template(template_name)
    system_body = system_text if system_text is not None else load_template(system_template_name)
    return [
        build_prompt(
            source_map,
            location,
            template_body,
            system_body,
            template_name=template_name,
            system_template_name=system_template_name,
            max_context_lines=max_context_lines,
        )
        for location in locations
    ]


def write_prompt_log(directory: Path, prompt: Prompt, completion: str) -> None:
    """Persist a prompt and its completion, as the paper's tool does."""
    directory.mkdir(parents=True, exist_ok=True)
    safe = prompt.id.replace(os.sep, "_").replace("#", "_")
    (directory / "{}.prompt.txt".format(safe)).write_text(
        "=== system ===\n{}\n\n=== user ===\n{}\n".format(prompt.system, prompt.text),
        encoding="utf-8",
    )
    (directory / "{}.completion.txt".format(safe)).write_text(completion, encoding="utf-8")
