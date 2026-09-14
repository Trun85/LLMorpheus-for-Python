"""LLM-based mutation testing for Python.

A Python implementation of the technique described in "LLMorpheus: Mutation
Testing using Large Language Models" (Tip, Bell and Schäfer, arXiv:2404.09952):
placeholders are introduced at designated locations in a program's source code
and an LLM is asked what they could be replaced with.

The package is laid out along the three components of the paper's Figure 5:

``llmorpheus.analysis``
    Parses sources and decides where placeholders go.
``llmorpheus.prompting``
    Instantiates prompt templates around those placeholders.
``llmorpheus.llm``
    Talks to a model — OpenAI-compatible, Anthropic, or an offline mock.
``llmorpheus.generation``
    Filters completions down to valid, non-duplicate mutants.
``llmorpheus.execution``
    Applies mutants, runs the test suite, classifies and reports.
"""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    """Expose the public API lazily, so ``import llmorpheus`` stays cheap."""
    from importlib import import_module

    sources = {
        "Location": "llmorpheus.analysis",
        "Position": "llmorpheus.analysis",
        "SourceMap": "llmorpheus.analysis",
        "find_locations": "llmorpheus.analysis",
        "Prompt": "llmorpheus.prompting",
        "build_prompts": "llmorpheus.prompting",
        "GeneratorConfig": "llmorpheus.generation",
        "Mutant": "llmorpheus.generation",
        "MutantSet": "llmorpheus.generation",
        "generate": "llmorpheus.generation",
        "RunSummary": "llmorpheus.execution",
        "RunnerConfig": "llmorpheus.execution",
        "run_mutants": "llmorpheus.execution",
        "text_summary": "llmorpheus.execution",
        "write_html_report": "llmorpheus.execution",
        "build_client": "llmorpheus.llm",
    }
    if name in sources:
        return getattr(import_module(sources[name]), name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


__all__ = [
    "GeneratorConfig",
    "Location",
    "Mutant",
    "MutantSet",
    "Position",
    "Prompt",
    "RunSummary",
    "RunnerConfig",
    "SourceMap",
    "__version__",
    "build_client",
    "build_prompts",
    "find_locations",
    "generate",
    "run_mutants",
    "text_summary",
    "write_html_report",
]
