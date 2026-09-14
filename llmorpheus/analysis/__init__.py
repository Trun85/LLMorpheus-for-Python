"""Static analysis: parsing sources and deciding where placeholders go.

Corresponds to the parsing half of the paper's *prompt generator*, which uses
BabelJS to find the locations at which mutations are introduced.
"""

from __future__ import annotations

from .locations import (
    ALL_KINDS,
    DEFAULT_KINDS,
    EXTENDED_KINDS,
    Location,
    find_locations,
    resolve_kinds,
    summarise_kinds,
)
from .positions import Position, SourceMap
from .project import FileSelector, SourceFile, guess_source_dirs, resolve_project_root

__all__ = [
    "ALL_KINDS",
    "DEFAULT_KINDS",
    "EXTENDED_KINDS",
    "FileSelector",
    "Location",
    "Position",
    "SourceFile",
    "SourceMap",
    "find_locations",
    "guess_source_dirs",
    "resolve_kinds",
    "resolve_project_root",
    "summarise_kinds",
]
