"""Prompt construction: the paper's *prompt generator*."""

from __future__ import annotations

from .prompts import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_TEMPLATE,
    PLACEHOLDER,
    Prompt,
    available_system_templates,
    available_templates,
    build_code_with_placeholder,
    build_prompt,
    build_prompts,
    load_template,
    render,
    write_prompt_log,
)

__all__ = [
    "DEFAULT_SYSTEM_TEMPLATE",
    "DEFAULT_TEMPLATE",
    "PLACEHOLDER",
    "Prompt",
    "available_system_templates",
    "available_templates",
    "build_code_with_placeholder",
    "build_prompt",
    "build_prompts",
    "load_template",
    "render",
    "write_prompt_log",
]
