"""Command-line interface for LLMorpheus-py."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from . import __version__
from .analysis import ALL_KINDS, guess_source_dirs, resolve_project_root, summarise_kinds
from .console import Console
from .execution import (
    DEFAULT_SCRUBBED_ENV,
    RunnerConfig,
    RunSummary,
    resolve_test_command,
    run_mutants,
    surviving_mutants,
    text_summary,
    write_html_report,
)
from .generation import GeneratorConfig, MutantSet, collect_locations, generate, group_by_kind
from .llm import PROVIDERS, THINKING_MAX_TOKENS, build_client
from .prompting import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_TEMPLATE,
    available_system_templates,
    available_templates,
)
from .settings import (
    Settings,
    SettingsError,
    list_subjects,
    load_settings,
    parse_env_assignments,
)

WORK_DIR = ".llmorpheus"

# Append-style options cannot take their defaults via ``set_defaults``: argparse
# appends command-line values to a list default instead of replacing it, so
# ``--src x`` would silently add to the subject's sources.
_LIST_DESTS = ("src", "include", "exclude", "kinds")


# --------------------------------------------------------------------------
# argument plumbing
# --------------------------------------------------------------------------
def _split_list(values: Optional[Sequence[str]]) -> List[str]:
    result: List[str] = []
    for value in values or ():
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _add_quiet_argument(parser: argparse.ArgumentParser) -> None:
    """Accept ``-q`` after the subcommand as well as before it.

    ``argparse.SUPPRESS`` keeps the subparser from overwriting a ``-q`` that was
    given before the subcommand with its own default.
    """
    parser.add_argument(
        "-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help="less output"
    )


def _add_config_arguments(parser: argparse.ArgumentParser, *, suppress: bool = True) -> None:
    """``--subject`` and ``--env-file``, accepted before or after the subcommand.

    They are read by a pre-parse in :func:`parse_arguments`, because the settings
    they select supply the defaults for every other option. They are declared
    here too, so the real parser accepts them and ``--help`` documents them.
    """
    extra = {"default": argparse.SUPPRESS} if suppress else {}
    group = parser.add_argument_group("configuration")
    group.add_argument(
        "--subject",
        metavar="NAME",
        help="per-project settings: subjects/NAME.conf, or a path to a .conf file",
        **extra
    )
    group.add_argument(
        "--env-file",
        metavar="FILE",
        help="settings file to use instead of the repository's .env",
        **extra
    )


def _add_project_arguments(parser: argparse.ArgumentParser) -> None:
    _add_quiet_argument(parser)
    _add_config_arguments(parser)
    group = parser.add_argument_group("project")
    group.add_argument(
        "-C", "--project", default=".", metavar="DIR", help="project root (default: .)"
    )
    group.add_argument(
        "--src",
        action="append",
        metavar="PATH",
        help="file or directory to mutate; repeatable, comma-separated allowed "
        "(default: src/ and top-level packages)",
    )
    group.add_argument(
        "--include", action="append", metavar="GLOB", help="only mutate files matching GLOB"
    )
    group.add_argument(
        "--exclude", action="append", metavar="GLOB", help="never mutate files matching GLOB"
    )


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("placeholders and prompts")
    group.add_argument(
        "--kinds",
        action="append",
        metavar="KIND",
        help="placeholder kinds: 'default' (the paper's scheme), 'all', or explicit "
        "names such as if-condition,call-argument",
    )
    group.add_argument(
        "--template", default=DEFAULT_TEMPLATE, help="prompt template name or path"
    )
    group.add_argument(
        "--system-template",
        default=DEFAULT_SYSTEM_TEMPLATE,
        help="system prompt template name or path",
    )
    group.add_argument(
        "--max-context-lines",
        type=int,
        default=200,
        help="lines of source around a placeholder to include (default: 200)",
    )
    group.add_argument("--max-fragment-chars", type=int, default=240)
    group.add_argument("--max-fragment-lines", type=int, default=8)
    group.add_argument(
        "--allow-multiline",
        action="store_true",
        help="accept multi-line replacements (off by default, as in the paper)",
    )
    group.add_argument(
        "--limit", type=int, default=None, metavar="N", help="use at most N prompts"
    )
    group.add_argument(
        "--no-save-prompts", action="store_true", help="do not write prompts/completions to disk"
    )


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("LLM")
    group.add_argument("--provider", default="openai", choices=list(PROVIDERS))
    group.add_argument("--model", default=None, help="model identifier")
    group.add_argument("--temperature", type=float, default=0.0)
    group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="output token budget (default: 250, or {} with --thinking)".format(
            THINKING_MAX_TOKENS
        ),
    )
    group.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, …)",
    )
    group.add_argument("--api-key-env", default=None, metavar="VAR")
    group.add_argument(
        "--attempts", type=int, default=3, help="retries per prompt on rate limits/errors"
    )
    group.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        metavar="MS",
        help="minimum milliseconds between requests",
    )
    group.add_argument("--request-timeout", type=float, default=120.0, metavar="SECONDS")
    group.add_argument("--concurrency", type=int, default=4, help="parallel LLM requests")
    group.add_argument("--no-cache", action="store_true", help="do not reuse cached completions")
    group.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="deepseek only: enable thinking mode (off by default; ignores --temperature "
        "and raises the default --max-tokens to {})".format(THINKING_MAX_TOKENS),
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("test execution")
    group.add_argument(
        "--test-command",
        default=None,
        metavar="CMD",
        help='how to run the tests (default: "python -m pytest -x -q"); a relative '
        "interpreter such as .venv/bin/python is relative to the project root",
    )
    group.add_argument(
        "--test-env",
        action="append",
        metavar="KEY=VALUE",
        help="environment variable for the test command; repeatable, overrides "
        "LLMORPHEUS_TEST_ENV",
    )
    group.add_argument(
        "--test-timeout", type=float, default=None, metavar="SECONDS", help="per-mutant timeout"
    )
    group.add_argument("--timeout-factor", type=float, default=3.0)
    group.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="mutants to evaluate in parallel; >1 copies the project per worker",
    )
    group.add_argument(
        "--skip-baseline", action="store_true", help="do not verify the suite is green first"
    )
    group.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="SCORE",
        help="exit non-zero if the mutation score is below SCORE",
    )


def _config_selection(argv: Sequence[str]) -> Dict[str, Optional[str]]:
    """Find ``--subject`` / ``--env-file`` wherever they appear on the command line."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--subject")
    pre.add_argument("--env-file")

    def _silent(message: str) -> None:
        raise SystemExit(2)

    pre.error = _silent  # type: ignore[assignment]  # the real parser reports errors
    try:
        known, _ = pre.parse_known_args(list(argv))
    except SystemExit:
        return {"subject": None, "env_file": None}
    return {"subject": known.subject, "env_file": known.env_file}


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    repo_root: Optional[Path] = None,
) -> argparse.Namespace:
    """Parse the command line on top of ``.env`` and subject-file defaults."""
    argv = list(sys.argv[1:] if argv is None else argv)
    selection = _config_selection(argv)
    settings = load_settings(
        env_file=selection["env_file"],
        subject=selection["subject"],
        environ=environ,
        repo_root=repo_root,
    )
    args = build_parser(settings).parse_args(argv)
    _apply_list_settings(args, settings)
    args.settings = settings
    return args


def _apply_list_settings(args: argparse.Namespace, settings: Settings) -> None:
    for dest in _LIST_DESTS:
        if hasattr(args, dest) and getattr(args, dest) is None:
            value = settings.list_default(dest)
            if value:
                setattr(args, dest, list(value))  # type: ignore[call-overload]
    if hasattr(args, "test_env"):
        merged = dict(settings.list_default("test_env") or {})  # type: ignore[call-overload]
        merged.update(_test_env(args))
        args.test_env = merged


def _test_env(args: argparse.Namespace) -> Dict[str, str]:
    value = getattr(args, "test_env", None)
    if isinstance(value, dict):
        return dict(value)
    merged: Dict[str, str] = {}
    for item in value or ():
        merged.update(parse_env_assignments(item, "--test-env"))
    return merged


def _output_dir(project_root: Path, requested: Optional[str]) -> Path:
    if requested:
        path = Path(requested).expanduser()
        return path if path.is_absolute() else project_root / path
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return project_root / WORK_DIR / "run-{}".format(stamp)


def _link_latest(project_root: Path, output_dir: Path) -> None:
    latest = project_root / WORK_DIR / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.symlink_to(output_dir.resolve(), target_is_directory=True)
    except OSError:
        pass


def _project_root(args: argparse.Namespace) -> Path:
    try:
        return resolve_project_root(args.project)
    except NotADirectoryError:
        settings = getattr(args, "settings", None)
        if settings is None or settings.subject_name is None:
            raise
        repo = settings.get("LLMORPHEUS_REPO")
        version = settings.get("LLMORPHEUS_VERSION")
        hint = "see subjects/README.md"
        if repo and version:
            hint = "clone it first:  git clone --branch {} --depth 1 {} {}".format(
                version, repo, shlex.quote(str(args.project))
            )
        raise NotADirectoryError(
            "subject {!r}: {} does not exist - {}".format(
                settings.subject_name, args.project, hint
            )
        )


def _git(root: Path, *arguments: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root)] + list(arguments),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _subject_status(
    args: argparse.Namespace, project_root: Path, console: Console
) -> Dict[str, object]:
    """Confirm a subject's checkout matches its pinned version; record provenance."""
    settings = getattr(args, "settings", None)
    if settings is None or settings.subject_name is None:
        return {}
    version = settings.get("LLMORPHEUS_VERSION")
    info: Dict[str, object] = {
        "name": settings.subject_name,
        "config": str(settings.subject_file),
        "repo": settings.get("LLMORPHEUS_REPO"),
        "version": version,
    }
    toplevel = _git(project_root, "rev-parse", "--show-toplevel")
    # An enclosing repository (e.g. this tool's own) is not the subject's checkout.
    if toplevel is None or os.path.realpath(toplevel) != os.path.realpath(str(project_root)):
        if version:
            console.warn(
                "{} is not a git checkout, so it cannot be confirmed to be version {}".format(
                    project_root, version
                )
            )
        return info
    commit = _git(project_root, "rev-parse", "HEAD")
    tag = _git(project_root, "describe", "--tags", "--exact-match", "HEAD")
    info.update(commit=commit, tag=tag)
    if version and tag != version:
        console.warn(
            "{} pins version {}, but {} is checked out at {}".format(
                settings.subject_file.name, version, project_root, tag or (commit or "?")[:12]
            )
        )
    modified = bool(_git(project_root, "status", "--porcelain", "--untracked-files=no"))
    info["modified"] = modified
    if modified:
        console.warn(
            "tracked files in {} have local modifications, so results will not reflect "
            "version {} - inspect with: git -C {} diff".format(
                project_root, version or "the pinned version", shlex.quote(str(project_root))
            )
        )
    return info


def _announce_settings(args: argparse.Namespace, console: Console) -> None:
    settings = getattr(args, "settings", None)
    if settings is None:
        return
    parts = []
    if settings.subject_file is not None:
        parts.append("subject {}".format(settings.subject_file))
    if settings.env_file is not None:
        parts.append(".env {}".format(settings.env_file))
    if parts:
        console.info("    settings: {}".format(", ".join(parts)))


def _generator_config(args: argparse.Namespace, project_root: Path, output_dir: Optional[Path]):
    sources = _split_list(args.src) or guess_source_dirs(project_root)
    return GeneratorConfig(
        project_root=project_root,
        sources=sources,
        include=_split_list(args.include),
        exclude=_split_list(args.exclude),
        kinds=_split_list(args.kinds),
        template=args.template,
        system_template=args.system_template,
        max_context_lines=args.max_context_lines,
        max_fragment_chars=args.max_fragment_chars,
        max_fragment_lines=args.max_fragment_lines,
        allow_multiline=args.allow_multiline,
        # ``locations`` does not take the LLM options, so fall back to the default.
        concurrency=getattr(args, "concurrency", 4),
        limit=args.limit,
        save_prompts=not args.no_save_prompts,
        output_dir=output_dir,
    )


def _client(args: argparse.Namespace, project_root: Path):
    cache_dir = None if args.no_cache else project_root / WORK_DIR / "cache"
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = THINKING_MAX_TOKENS if args.thinking else 250
    settings = getattr(args, "settings", None)
    secrets: Dict[str, str] = {}
    if settings is not None:
        secrets = settings.secrets(extra_names=[args.api_key_env] if args.api_key_env else ())
    return build_client(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=max_tokens,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        attempts=args.attempts,
        rate_limit_ms=args.rate_limit,
        timeout=args.request_timeout,
        cache_dir=cache_dir,
        thinking=args.thinking,
        secrets=secrets,
    )


def _runner_config(args: argparse.Namespace, project_root: Path) -> RunnerConfig:
    command, use_shell = resolve_test_command(args.test_command, project_root)
    api_key_env = getattr(args, "api_key_env", None)
    settings = getattr(args, "settings", None)
    if api_key_env is None and settings is not None:
        api_key_env = settings.get("LLMORPHEUS_API_KEY_ENV")
    scrub = tuple(DEFAULT_SCRUBBED_ENV) + ((str(api_key_env),) if api_key_env else ())
    return RunnerConfig(
        project_root=project_root,
        test_command=command,
        use_shell=use_shell,
        timeout=args.test_timeout,
        timeout_factor=args.timeout_factor,
        jobs=max(1, args.jobs),
        skip_baseline=args.skip_baseline,
        env=_test_env(args),
        scrub_env=scrub,
    )


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def command_locations(args: argparse.Namespace, console: Console) -> int:
    project_root = _project_root(args)
    config = _generator_config(args, project_root, None)
    files, per_file = collect_locations(config, console)
    all_locations = [item for items in per_file.values() for item in items]
    if args.json:
        print(json.dumps([item.as_dict() for item in all_locations], indent=2))
        return 0
    for path in sorted(per_file):
        console.info("{} ({} locations)".format(path, len(per_file[path])))
        for location in per_file[path]:
            console.info("    {}".format(location.describe()))
    console.info("")
    console.step(
        "{} location(s) across {} file(s) of {} scanned".format(
            len(all_locations), len(per_file), len(files)
        )
    )
    for kind, count in summarise_kinds(all_locations).items():
        console.info("    {:<26} {}".format(kind, count))
    return 0


def command_generate(args: argparse.Namespace, console: Console) -> int:
    project_root = _project_root(args)
    _announce_settings(args, console)
    subject = _subject_status(args, project_root, console)
    client = _client(args, project_root)
    # Fail on a missing API key before touching the project: creating the run
    # directory would repoint .llmorpheus/latest at an empty run.
    client.preflight()
    output_dir = _output_dir(project_root, args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    _link_latest(project_root, output_dir)

    config = _generator_config(args, project_root, output_dir)
    mutant_set = generate(config, client, console)
    if subject:
        mutant_set.meta["subject"] = subject
    path = output_dir / "mutants.json"
    mutant_set.write(path)
    console.info("    wrote {}".format(path))
    for kind, count in group_by_kind(mutant_set.mutants).items():
        console.info("    {:<26} {}".format(kind, count))
    return 0


def command_run(args: argparse.Namespace, console: Console) -> int:
    project_root = _project_root(args)
    _announce_settings(args, console)
    subject = _subject_status(args, project_root, console)
    mutants_path = Path(args.mutants).expanduser()
    if not mutants_path.is_absolute():
        mutants_path = project_root / mutants_path
    if not mutants_path.is_file():
        console.error("no mutants file at {}".format(mutants_path))
        return 2
    mutant_set = MutantSet.read(mutants_path)
    if not mutant_set.mutants:
        console.error("the mutants file contains no mutants")
        return 2

    output_dir = _output_dir(project_root, args.out or str(mutants_path.parent))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    summary = run_mutants(
        mutant_set, _runner_config(args, project_root), console, results_path=results_path
    )
    if subject:
        summary.meta["subject"] = subject
    summary.write(results_path)
    console.info("    wrote {}".format(results_path))
    return _finish(args, console, summary, output_dir, project_root)


def command_report(args: argparse.Namespace, console: Console) -> int:
    project_root = _project_root(args)
    results_path = Path(args.results).expanduser()
    if not results_path.is_absolute():
        results_path = project_root / results_path
    if not results_path.is_file():
        console.error("no results file at {}".format(results_path))
        return 2
    summary = RunSummary.read(results_path)
    output_dir = _output_dir(project_root, args.out or str(results_path.parent))
    return _finish(args, console, summary, output_dir, project_root)


def command_all(args: argparse.Namespace, console: Console) -> int:
    project_root = _project_root(args)
    _announce_settings(args, console)
    subject = _subject_status(args, project_root, console)
    # Check everything that can fail cheaply before spending tokens or touching
    # the project: a missing virtualenv or API key should fail now, not after
    # every prompt has been paid for or .llmorpheus/latest has been repointed.
    runner_config = _runner_config(args, project_root)
    client = _client(args, project_root)
    client.preflight()
    output_dir = _output_dir(project_root, args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    _link_latest(project_root, output_dir)

    config = _generator_config(args, project_root, output_dir)
    mutant_set = generate(config, client, console)
    if subject:
        mutant_set.meta["subject"] = subject
    mutant_set.write(output_dir / "mutants.json")
    if not mutant_set.mutants:
        console.error("no mutants were generated; nothing to run")
        return 1

    results_path = output_dir / "results.json"
    summary = run_mutants(mutant_set, runner_config, console, results_path=results_path)
    if subject:
        summary.meta["subject"] = subject
    summary.write(results_path)
    return _finish(args, console, summary, output_dir, project_root)


def command_config(args: argparse.Namespace, console: Console) -> int:
    settings: Settings = args.settings
    console.step("settings sources, highest precedence first")
    console.info("    1. command-line flags")
    for index, layer in enumerate(settings.layers, start=2):
        console.info("    {}. {}".format(index, settings.describe_layer(layer)))
    console.info("    {}. built-in defaults".format(len(settings.layers) + 2))
    console.step("effective values")
    for key, value, source in settings.rows():
        console.info("    {:<27} {:<46} {}".format(key, value, source))
    names = list_subjects(settings.subjects_dir)
    console.step("subjects in {}".format(settings.subjects_dir))
    console.info("    {}".format(", ".join(names) if names else "(none)"))
    return 0


def _finish(
    args: argparse.Namespace,
    console: Console,
    summary: RunSummary,
    output_dir: Path,
    project_root: Path,
) -> int:
    console.info("")
    console.info(text_summary(summary))
    survivors = surviving_mutants(summary, limit=args.show_survivors)
    if survivors:
        console.info("")
        console.step("surviving mutants (first {})".format(len(survivors)))
        for result in survivors:
            console.info("    {}".format(result.mutant.describe()))
            if result.mutant.explanation:
                console.info("        {}".format(result.mutant.explanation))
    html_path = Path(args.html).expanduser() if args.html else output_dir / "report.html"
    if not html_path.is_absolute():
        html_path = project_root / html_path
    write_html_report(summary, html_path, project_root=project_root)
    console.info("")
    console.step("HTML report: {}".format(html_path))
    if args.fail_under is not None and summary.mutation_score < args.fail_under:
        console.error(
            "mutation score {:.2f}% is below the required {:.2f}%".format(
                summary.mutation_score, args.fail_under
            )
        )
        return 1
    return 0


# --------------------------------------------------------------------------
def build_parser(settings: Optional[Settings] = None) -> argparse.ArgumentParser:
    """The argument parser; ``settings`` (from .env / a subject file) supply defaults."""
    parser = argparse.ArgumentParser(
        prog="llmorpheus",
        description="LLM-based mutation testing for Python, after Tip, Bell and "
        "Schäfer's LLMorpheus (arXiv:2404.09952).",
    )
    parser.add_argument("--version", action="version", version="llmorpheus-py {}".format(__version__))
    parser.add_argument("-q", "--quiet", action="store_true")
    _add_config_arguments(parser, suppress=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    locations = subparsers.add_parser(
        "locations", help="list the placeholder locations that would be prompted about"
    )
    _add_project_arguments(locations)
    _add_generation_arguments(locations)
    locations.add_argument("--json", action="store_true", help="emit JSON")
    locations.set_defaults(handler=command_locations)

    generate_parser = subparsers.add_parser(
        "generate", help="prompt an LLM and write mutants.json"
    )
    _add_project_arguments(generate_parser)
    _add_generation_arguments(generate_parser)
    _add_llm_arguments(generate_parser)
    generate_parser.add_argument("--out", default=None, metavar="DIR", help="output directory")
    generate_parser.set_defaults(handler=command_generate)

    run_parser = subparsers.add_parser("run", help="apply mutants and run the test suite")
    _add_project_arguments(run_parser)
    _add_run_arguments(run_parser)
    run_parser.add_argument(
        "--mutants", default="{}/latest/mutants.json".format(WORK_DIR), metavar="FILE"
    )
    run_parser.add_argument("--out", default=None, metavar="DIR")
    run_parser.add_argument("--html", default=None, metavar="FILE")
    run_parser.add_argument("--show-survivors", type=int, default=20, metavar="N")
    run_parser.set_defaults(handler=command_run)

    report_parser = subparsers.add_parser("report", help="re-render a report from results.json")
    _add_project_arguments(report_parser)
    report_parser.add_argument(
        "--results", default="{}/latest/results.json".format(WORK_DIR), metavar="FILE"
    )
    report_parser.add_argument("--out", default=None, metavar="DIR")
    report_parser.add_argument("--html", default=None, metavar="FILE")
    report_parser.add_argument("--show-survivors", type=int, default=20, metavar="N")
    report_parser.add_argument("--fail-under", type=float, default=None, metavar="SCORE")
    report_parser.set_defaults(handler=command_report)

    all_parser = subparsers.add_parser("all", help="generate, run and report in one go")
    _add_project_arguments(all_parser)
    _add_generation_arguments(all_parser)
    _add_llm_arguments(all_parser)
    _add_run_arguments(all_parser)
    all_parser.add_argument("--out", default=None, metavar="DIR")
    all_parser.add_argument("--html", default=None, metavar="FILE")
    all_parser.add_argument("--show-survivors", type=int, default=20, metavar="N")
    all_parser.set_defaults(handler=command_all)

    config_parser = subparsers.add_parser(
        "config", help="show the effective settings and where each value comes from"
    )
    _add_quiet_argument(config_parser)
    _add_config_arguments(config_parser)
    config_parser.set_defaults(handler=command_config)

    templates = subparsers.add_parser("templates", help="list bundled prompt templates and kinds")
    _add_quiet_argument(templates)
    templates.set_defaults(handler=command_templates)

    if settings is not None:
        defaults = settings.scalar_defaults()
        for subparser in (locations, generate_parser, run_parser, report_parser, all_parser):
            dests = {action.dest for action in subparser._actions}
            subparser.set_defaults(
                **{dest: value for dest, value in defaults.items() if dest in dests}
            )
    return parser


def command_templates(args: argparse.Namespace, console: Console) -> int:
    console.step("prompt templates")
    for name in available_templates():
        console.info("    {}".format(name))
    console.step("system prompt templates")
    for name in available_system_templates():
        console.info("    {}".format(name))
    console.step("placeholder kinds")
    for name in ALL_KINDS:
        console.info("    {}".format(name))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_arguments(argv)
    except SettingsError as error:
        Console().error(str(error))
        return 2
    console = Console(quiet=args.quiet)
    try:
        return int(args.handler(args, console))
    except KeyboardInterrupt:
        console.error("interrupted")
        return 130
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as error:
        console.error(str(error))
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
