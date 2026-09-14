"""Turning completions into mutants: the paper's *mutant generator*."""

from __future__ import annotations

from .generator import GeneratorConfig, build_all_prompts, collect_locations, generate
from .mutants import (
    CandidateFilter,
    GenerationStats,
    Mutant,
    MutantSet,
    extract_code_blocks,
    extract_explanations,
    file_digest,
    group_by_kind,
    is_syntactically_valid,
    mutants_from_completion,
)

__all__ = [
    "CandidateFilter",
    "GenerationStats",
    "GeneratorConfig",
    "Mutant",
    "MutantSet",
    "build_all_prompts",
    "collect_locations",
    "extract_code_blocks",
    "extract_explanations",
    "file_digest",
    "generate",
    "group_by_kind",
    "is_syntactically_valid",
    "mutants_from_completion",
]
