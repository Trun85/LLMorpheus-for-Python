"""The prompt generator and mutant generator, wired together.

Corresponds to the first two components of Figure 5 in the paper: source files
are parsed, placeholders are inserted at designated locations, an LLM is asked
what each placeholder could be replaced with, and the suggestions are filtered
down to syntactically valid, non-duplicate mutants.
"""

from __future__ import annotations

import datetime as _dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..console import Console, Timer, format_duration
from ..llm import LLMClient, LLMError
from ..analysis.locations import Location, find_locations, resolve_kinds, summarise_kinds
from .mutants import (
    CandidateFilter,
    GenerationStats,
    Mutant,
    MutantSet,
    file_digest,
    group_by_kind,
    mutants_from_completion,
)
from ..analysis.project import FileSelector, SourceFile
from ..prompting.prompts import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_TEMPLATE,
    Prompt,
    build_prompts,
    load_template,
    write_prompt_log,
)


# HTTP statuses meaning the credentials themselves were refused. Unlike a
# malformed or rate-limited response, these fail every prompt the same way.
AUTHENTICATION_FAILURES = (401, 403)


@dataclass
class GeneratorConfig:
    """Everything that controls *which* mutants get generated."""

    project_root: Path
    sources: Sequence[str] = ()
    include: Sequence[str] = ()
    exclude: Sequence[str] = ()
    kinds: Sequence[str] = ()
    template: str = DEFAULT_TEMPLATE
    system_template: str = DEFAULT_SYSTEM_TEMPLATE
    max_context_lines: int = 200
    max_fragment_chars: int = 240
    max_fragment_lines: int = 8
    allow_multiline: bool = False
    concurrency: int = 4
    limit: Optional[int] = None
    save_prompts: bool = True
    output_dir: Optional[Path] = None


def collect_locations(
    config: GeneratorConfig, console: Optional[Console] = None
) -> Tuple[List[SourceFile], Dict[str, List[Location]]]:
    """Find every placeholder location in the selected source files."""
    console = console or Console(quiet=True)
    selector = FileSelector(
        project_root=config.project_root,
        sources=config.sources,
        include=config.include,
        exclude=config.exclude,
    )
    files = selector.find()
    kinds = resolve_kinds(config.kinds)
    per_file: Dict[str, List[Location]] = {}
    for source_file in files:
        locations = find_locations(
            source_file.source,
            source_file.relative_path,
            kinds=kinds,
            max_fragment_chars=config.max_fragment_chars,
            max_fragment_lines=config.max_fragment_lines,
        )
        if locations:
            per_file[source_file.relative_path] = locations
    return files, per_file


def build_all_prompts(
    files: Sequence[SourceFile],
    per_file: Dict[str, List[Location]],
    config: GeneratorConfig,
) -> Tuple[List[Prompt], Dict[str, str]]:
    """Instantiate one prompt per location; also returns file sources by path."""
    template_text = load_template(config.template)
    system_text = load_template(config.system_template)
    sources = {item.relative_path: item.source for item in files}
    prompts: List[Prompt] = []
    for source_file in files:
        locations = per_file.get(source_file.relative_path)
        if not locations:
            continue
        prompts.extend(
            build_prompts(
                source_file.source,
                locations,
                template_name=config.template,
                system_template_name=config.system_template,
                max_context_lines=config.max_context_lines,
                template_text=template_text,
                system_text=system_text,
            )
        )
    if config.limit is not None:
        prompts = prompts[: max(0, config.limit)]
    return prompts, sources


def generate(
    config: GeneratorConfig,
    client: LLMClient,
    console: Optional[Console] = None,
) -> MutantSet:
    """Run the full generation pipeline and return the resulting mutant set."""
    console = console or Console()
    client.preflight()
    with Timer() as timer:
        files, per_file = collect_locations(config, console)
        prompts, sources = build_all_prompts(files, per_file, config)

        location_total = sum(len(items) for items in per_file.values())
        console.step(
            "{} source file(s), {} placeholder location(s), {} prompt(s)".format(
                len(per_file), location_total, len(prompts)
            )
        )

        stats = GenerationStats(prompts=len(prompts))
        filters = {
            path: CandidateFilter(
                source=source,
                file=path,
                allow_multiline=config.allow_multiline,
            )
            for path, source in sources.items()
        }
        prompt_dir = (config.output_dir / "prompts") if config.output_dir else None

        mutants: List[Mutant] = []
        failures: List[str] = []
        if prompts:
            workers = max(1, config.concurrency)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_complete, client, prompt): prompt for prompt in prompts
                }
                done = 0
                for future in as_completed(futures):
                    prompt = futures[future]
                    done += 1
                    console.progress(done, len(prompts), "prompts")
                    try:
                        response = future.result()
                    except Exception as error:  # one bad prompt must not end the run...
                        status = getattr(error, "status", None)
                        if status in AUTHENTICATION_FAILURES:
                            # ...but a rejected key fails every prompt identically, so
                            # stop rather than send the rest only to be refused.
                            for pending in futures:
                                pending.cancel()
                            raise LLMError(
                                "the provider rejected the API key (HTTP {}), so no further "
                                "prompts were sent. Check {} - `llmorpheus config` shows "
                                "where it is read from. Provider said: {}".format(
                                    status, getattr(client, "api_key_env", "the API key"), error
                                )
                            ) from error
                        stats.failed_prompts += 1
                        failures.append(
                            "{}: {}: {}".format(prompt.id, type(error).__name__, error)
                        )
                        continue
                    stats.completions += 1
                    stats.prompt_tokens += response.prompt_tokens
                    stats.completion_tokens += response.completion_tokens
                    if prompt_dir is not None and config.save_prompts:
                        write_prompt_log(prompt_dir, prompt, response.text)
                    mutants.extend(
                        mutants_from_completion(
                            response.text,
                            prompt.location,
                            filters[prompt.location.file],
                            stats,
                        )
                    )

    if prompts and stats.failed_prompts == len(prompts):
        raise LLMError(
            "every one of the {} prompt(s) failed; the first error was - {}".format(
                len(prompts), failures[0]
            )
        )
    for message in failures[:5]:
        console.warn("prompt failed - {}".format(message))
    if len(failures) > 5:
        console.warn("... and {} more failed prompt(s)".format(len(failures) - 5))

    mutants.sort(key=lambda item: (item.file, item.start, item.end, item.replacement))
    meta = {
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(config.project_root),
        "sources": list(config.sources),
        "kinds": sorted(resolve_kinds(config.kinds)),
        "template": config.template,
        "system_template": config.system_template,
        "max_context_lines": config.max_context_lines,
        "llm": client.describe(),
        "duration_seconds": round(timer.seconds, 2),
        "files": sorted(per_file.keys()),
        # Mutants carry absolute line/column offsets, so ``llmorpheus run`` uses
        # these to refuse a mutant set generated against different source text.
        "file_digests": {
            path: file_digest(sources[path]) for path in sorted(per_file.keys()) if path in sources
        },
        "locations_by_kind": summarise_kinds(
            [item for items in per_file.values() for item in items]
        ),
        "mutants_by_kind": group_by_kind(mutants),
    }

    console.step(
        "{} mutant(s) from {} candidate(s) in {} "
        "(discarded: {} invalid, {} identical, {} duplicate)".format(
            stats.mutants,
            stats.candidates,
            format_duration(timer.seconds),
            stats.invalid,
            stats.identical,
            stats.duplicate,
        )
    )
    return MutantSet(mutants=mutants, stats=stats, meta=meta)


def _complete(client: LLMClient, prompt: Prompt):
    return client.complete(prompt.system, prompt.text)
