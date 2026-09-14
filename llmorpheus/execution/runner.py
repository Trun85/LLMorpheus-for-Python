"""Applying mutants and classifying them by running the project's test suite.

This is the role played by the customised StrykerJS in the paper: it takes a
precomputed ``mutants.json``, applies one mutant at a time, runs the tests, and
records whether each mutant was *killed*, *survived*, or *timed out*.
"""

from __future__ import annotations

import atexit
import datetime as _dt
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

from ..analysis.positions import SourceMap
from ..analysis.project import DEFAULT_EXCLUDE_DIRS
from ..console import Console, format_duration
from ..generation.mutants import Mutant, MutantSet, file_digest

KILLED = "killed"
SURVIVED = "survived"
TIMEOUT = "timeout"
ERROR = "error"

DEFAULT_TEST_COMMAND = [sys.executable, "-m", "pytest", "-x", "-q"]

# LLM credentials are never needed by the suites under test, and a mutated run
# whose failing test prints its environment would otherwise copy them into
# results.json and the HTML report. Any other ``*_API_KEY`` is dropped as well.
DEFAULT_SCRUBBED_ENV = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
)

# Files we restore on the way out even if the process is interrupted.
_DIRTY_FILES: Dict[Path, str] = {}
_DIRTY_LOCK = threading.Lock()


def _remember(path: Path, original: str) -> None:
    with _DIRTY_LOCK:
        _DIRTY_FILES.setdefault(path, original)


def _forget(path: Path) -> None:
    with _DIRTY_LOCK:
        _DIRTY_FILES.pop(path, None)


def restore_all() -> List[Path]:
    """Put every mutated file back. Registered with ``atexit`` and signals.

    Returns the paths that could *not* be restored, so callers can say so out
    loud rather than leaving mutated code on disk silently.
    """
    with _DIRTY_LOCK:
        pending = dict(_DIRTY_FILES)
        _DIRTY_FILES.clear()
    failed: List[Path] = []
    for path, original in pending.items():
        try:
            path.write_text(original, encoding="utf-8")
        except OSError:
            failed.append(path)
    return failed


atexit.register(restore_all)


def _install_signal_handlers() -> None:
    def handler(signum, frame):  # pragma: no cover - interactive path
        restore_all()
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


@dataclass
class RunnerConfig:
    project_root: Path
    test_command: Sequence[str] = tuple(DEFAULT_TEST_COMMAND)
    use_shell: bool = False
    timeout: Optional[float] = None
    timeout_factor: float = 3.0
    minimum_timeout: float = 30.0
    jobs: int = 1
    skip_baseline: bool = False
    output_tail_chars: int = 2000
    env: Dict[str, str] = field(default_factory=dict)
    scrub_env: Sequence[str] = DEFAULT_SCRUBBED_ENV


class TestOutcome(NamedTuple):
    """One execution of the test command."""

    __test__ = False  # not a pytest test class, despite the name

    exit_code: Optional[int]
    output: str
    duration: float
    timed_out: bool


@dataclass
class MutantResult:
    mutant: Mutant
    status: str
    duration: float
    exit_code: Optional[int] = None
    output: str = ""

    def as_dict(self) -> dict:
        data = self.mutant.as_dict()
        data.update(
            {
                "status": self.status,
                "duration": round(self.duration, 3),
                "exit_code": self.exit_code,
                "output": self.output,
            }
        )
        return data


@dataclass
class RunSummary:
    results: List[MutantResult]
    baseline_duration: float
    timeout: float
    total_duration: float
    meta: Dict[str, object] = field(default_factory=dict)

    def counts(self) -> Dict[str, int]:
        counts = {KILLED: 0, SURVIVED: 0, TIMEOUT: 0, ERROR: 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    @property
    def mutation_score(self) -> float:
        counts = self.counts()
        detected = counts[KILLED] + counts[TIMEOUT]
        total = detected + counts[SURVIVED]
        return (detected / total * 100.0) if total else 0.0

    def as_dict(self) -> dict:
        return {
            "meta": self.meta,
            "summary": {
                "counts": self.counts(),
                "mutation_score": round(self.mutation_score, 2),
                "baseline_duration": round(self.baseline_duration, 3),
                "timeout": round(self.timeout, 2),
                "total_duration": round(self.total_duration, 2),
            },
            "results": [result.as_dict() for result in self.results],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "RunSummary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results = []
        for item in data.get("results", []):
            results.append(
                MutantResult(
                    mutant=Mutant.from_dict(item),
                    status=item.get("status", ERROR),
                    duration=float(item.get("duration", 0.0)),
                    exit_code=item.get("exit_code"),
                    output=item.get("output", ""),
                )
            )
        summary = data.get("summary", {})
        return cls(
            results=results,
            baseline_duration=float(summary.get("baseline_duration", 0.0)),
            timeout=float(summary.get("timeout", 0.0)),
            total_duration=float(summary.get("total_duration", 0.0)),
            meta=data.get("meta", {}),
        )


class TestRunner:
    """Runs the configured test command inside a given directory."""

    __test__ = False  # not a pytest test class, despite the name

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config

    def _environment(self) -> Dict[str, str]:
        env = {
            name: value
            for name, value in os.environ.items()
            if name not in self.config.scrub_env and not name.endswith("_API_KEY")
        }
        # Never leave stale bytecode behind for a mutated module.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Explicitly configured test variables always apply.
        env.update(self.config.env)
        return env

    def run(self, cwd: Path, timeout: Optional[float]) -> TestOutcome:
        command: object
        if self.config.use_shell:
            command = " ".join(self.config.test_command)
        else:
            command = list(self.config.test_command)
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=self.config.use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self._environment(),
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate(process)
            output, _ = process.communicate()
            # A timeout is reported by its own flag, never by an exit code: a
            # process killed by a signal already yields a negative returncode
            # (SIGHUP is -1), which would otherwise be misread as a timeout.
            return TestOutcome(
                exit_code=process.returncode,
                output=_tail(output, self.config.output_tail_chars),
                duration=time.monotonic() - started,
                timed_out=True,
            )
        return TestOutcome(
            exit_code=process.returncode,
            output=_tail(output, self.config.output_tail_chars),
            duration=time.monotonic() - started,
            timed_out=False,
        )


def _terminate(process: "subprocess.Popen") -> None:
    """Kill the whole process group so pytest's children die too."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:  # pragma: no cover - Windows has no process groups to signal
            process.kill()
        return
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        pass
    try:
        process.kill()
    except (OSError, AttributeError):
        pass


def _tail(output: Optional[bytes], limit: int) -> str:
    if not output:
        return ""
    text = output.decode("utf-8", errors="replace")
    return text if len(text) <= limit else "…\n" + text[-limit:]


class Workspace:
    """A directory in which mutants are applied, one at a time."""

    def __init__(self, root: Path, files: Iterable[str]) -> None:
        self.root = root
        self._originals: Dict[str, str] = {}
        for relative in files:
            path = self.root / relative
            if path.is_file():
                self._originals[relative] = path.read_text(encoding="utf-8")

    def original(self, relative: str) -> Optional[str]:
        return self._originals.get(relative)

    def apply(self, mutant: Mutant) -> Optional[str]:
        """Apply ``mutant``; returns ``None``, or why it could not be applied.

        The span is re-checked against the file before writing. A mutant records
        absolute line/column offsets captured during generation, so any edit to
        the file in between would otherwise splice the replacement into an
        unrelated region and produce a confidently wrong killed/survived verdict.
        """
        original = self._originals.get(mutant.file)
        if original is None:
            return "file not found: {}".format(mutant.file)
        current = SourceMap(original).text(mutant.start, mutant.end)
        if current != mutant.original:
            return (
                "{} has changed since the mutants were generated: expected {!r} at "
                "{}, found {!r}. Re-run `llmorpheus generate`.".format(
                    mutant.file, mutant.original, mutant.start, current
                )
            )
        path = self.root / mutant.file
        _remember(path, original)
        path.write_text(mutant.apply(original), encoding="utf-8")
        _purge_bytecode(path)
        return None

    def restore(self, mutant: Mutant) -> None:
        original = self._originals.get(mutant.file)
        if original is None:
            return
        path = self.root / mutant.file
        # Only drop the emergency-restore entry once the file is genuinely back:
        # forgetting it after a failed write would strip restore_all()'s ability
        # to recover, leaving mutated code on disk silently.
        path.write_text(original, encoding="utf-8")
        _purge_bytecode(path)
        _forget(path)


def _purge_bytecode(path: Path) -> None:
    cache = path.parent / "__pycache__"
    if not cache.is_dir():
        return
    for compiled in cache.glob("{}.*.pyc".format(path.stem)):
        try:
            compiled.unlink()
        except OSError:
            pass


def _copy_project(project_root: Path, destination: Path) -> Path:
    ignore = shutil.ignore_patterns(*DEFAULT_EXCLUDE_DIRS, "*.pyc")
    shutil.copytree(project_root, destination, ignore=ignore, symlinks=True)
    return destination


def run_mutants(
    mutant_set: MutantSet,
    config: RunnerConfig,
    console: Optional[Console] = None,
    results_path: Optional[Path] = None,
) -> RunSummary:
    """Execute the test suite once per mutant and classify the outcome."""
    console = console or Console()
    _install_signal_handlers()
    runner = TestRunner(config)

    stale = _stale_files(mutant_set, config.project_root)
    skipped: List[MutantResult] = []
    mutants: List[Mutant] = []
    for mutant in mutant_set.mutants:
        if mutant.file in stale:
            skipped.append(
                MutantResult(
                    mutant=mutant,
                    status=ERROR,
                    duration=0.0,
                    output="{} has changed since the mutants were generated".format(mutant.file),
                )
            )
        else:
            mutants.append(mutant)
    if stale:
        console.warn(
            "skipping {} mutant(s): {} changed since generation - re-run "
            "`llmorpheus generate`".format(len(skipped), ", ".join(sorted(stale)))
        )
    files = sorted({mutant.file for mutant in mutants})

    baseline_duration = 0.0
    if not config.skip_baseline:
        console.step("running the test suite unmutated (baseline)")
        baseline = runner.run(config.project_root, None)
        baseline_duration = baseline.duration
        if baseline.timed_out or baseline.exit_code != 0:
            console.error(
                "the unmutated test suite failed (exit code {}). "
                "Mutation testing needs a green suite; last output:\n{}".format(
                    baseline.exit_code, baseline.output
                )
            )
            raise RuntimeError("baseline test run failed")
        console.info("    baseline passed in {}".format(format_duration(baseline_duration)))

    timeout = config.timeout
    if timeout is None:
        timeout = max(config.minimum_timeout, math.ceil(baseline_duration * config.timeout_factor))

    console.step(
        "running {} mutant(s) with a {} timeout, {} job(s)".format(
            len(mutants), format_duration(timeout), config.jobs
        )
    )

    started = time.monotonic()
    results: List[MutantResult] = []
    lock = threading.Lock()
    temporary_roots: List[Path] = []

    def record(result: MutantResult) -> None:
        with lock:
            results.append(result)
            done = len(results)
            console.progress(done, len(mutants), _progress_suffix(results))
            if results_path is not None and (done % 10 == 0 or done == len(mutants)):
                _write_partial(results_path, results, baseline_duration, timeout, started, mutant_set)

    try:
        if config.jobs <= 1 or len(mutants) <= 1:
            workspace = Workspace(config.project_root, files)
            for mutant in mutants:
                record(_evaluate(mutant, workspace, runner, timeout))
        else:
            buckets = _split(mutants, config.jobs)
            with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
                futures = []
                for index, bucket in enumerate(buckets):
                    root = Path(tempfile.mkdtemp(prefix="llmorpheus-w{}-".format(index)))
                    temporary_roots.append(root)
                    worker_root = _copy_project(config.project_root, root / config.project_root.name)
                    worker_config = RunnerConfig(**{**config.__dict__, "project_root": worker_root})
                    worker_config.env = dict(config.env)
                    worker_config.env.setdefault(
                        "PYTHONPATH",
                        os.pathsep.join(
                            filter(None, [str(worker_root), os.environ.get("PYTHONPATH", "")])
                        ),
                    )
                    futures.append(
                        pool.submit(
                            _evaluate_bucket,
                            bucket,
                            worker_root,
                            files,
                            TestRunner(worker_config),
                            timeout,
                            record,
                        )
                    )
                for future in as_completed(futures):
                    future.result()
    finally:
        unrestored = restore_all()
        if unrestored:
            console.error(
                "could not restore {} - these file(s) still contain mutated code: {}".format(
                    "them" if len(unrestored) > 1 else "it",
                    ", ".join(str(path) for path in unrestored),
                )
            )
        for root in temporary_roots:
            shutil.rmtree(root, ignore_errors=True)

    total_duration = time.monotonic() - started
    results.extend(skipped)
    results.sort(key=lambda item: (item.mutant.file, item.mutant.start, item.mutant.replacement))
    summary = RunSummary(
        results=results,
        baseline_duration=baseline_duration,
        timeout=timeout,
        total_duration=total_duration,
        meta=_run_meta(config, mutant_set),
    )
    counts = summary.counts()
    console.step(
        "killed {}, survived {}, timed out {}, errored {} - mutation score {:.2f}% in {}".format(
            counts[KILLED],
            counts[SURVIVED],
            counts[TIMEOUT],
            counts[ERROR],
            summary.mutation_score,
            format_duration(total_duration),
        )
    )
    return summary


def _evaluate_bucket(
    mutants: Sequence[Mutant],
    root: Path,
    files: Sequence[str],
    runner: TestRunner,
    timeout: float,
    record,
) -> None:
    workspace = Workspace(root, files)
    for mutant in mutants:
        record(_evaluate(mutant, workspace, runner, timeout))


def _evaluate(
    mutant: Mutant, workspace: Workspace, runner: TestRunner, timeout: float
) -> MutantResult:
    problem = workspace.apply(mutant)
    if problem is not None:
        return MutantResult(mutant=mutant, status=ERROR, duration=0.0, output=problem)
    try:
        outcome = runner.run(workspace.root, timeout)
    except OSError as error:
        return MutantResult(mutant=mutant, status=ERROR, duration=0.0, output=str(error))
    finally:
        workspace.restore(mutant)
    if outcome.timed_out:
        return MutantResult(
            mutant=mutant,
            status=TIMEOUT,
            duration=outcome.duration,
            exit_code=outcome.exit_code,
            output=outcome.output,
        )
    status = SURVIVED if outcome.exit_code == 0 else KILLED
    return MutantResult(
        mutant=mutant,
        status=status,
        duration=outcome.duration,
        exit_code=outcome.exit_code,
        output=outcome.output,
    )


def _stale_files(mutant_set: MutantSet, project_root: Path) -> Set[str]:
    """Files whose contents no longer match what they were when generated.

    ``mutants.json`` records a digest per file precisely so that a stale mutant
    set is caught here rather than being spliced into shifted offsets.
    """
    digests = (mutant_set.meta or {}).get("file_digests") or {}
    if not isinstance(digests, dict):
        return set()
    stale: Set[str] = set()
    for relative, expected in digests.items():
        path = project_root / relative
        try:
            actual = file_digest(path.read_text(encoding="utf-8"))
        except OSError:
            actual = None
        if actual != expected:
            stale.add(relative)
    return stale


def _split(items: Sequence[Mutant], buckets: int) -> List[List[Mutant]]:
    count = max(1, min(buckets, len(items)))
    result: List[List[Mutant]] = [[] for _ in range(count)]
    for index, item in enumerate(items):
        result[index % count].append(item)
    return [bucket for bucket in result if bucket]


def _progress_suffix(results: Sequence[MutantResult]) -> str:
    counts = {KILLED: 0, SURVIVED: 0, TIMEOUT: 0, ERROR: 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return "killed {} / survived {}".format(counts[KILLED], counts[SURVIVED])


def _run_meta(config: RunnerConfig, mutant_set: MutantSet) -> Dict[str, object]:
    return {
        "run_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(config.project_root),
        "test_command": list(config.test_command),
        "jobs": config.jobs,
        "generation": mutant_set.meta,
        "generation_stats": mutant_set.stats.as_dict(),
    }


def _write_partial(
    path: Path,
    results: Sequence[MutantResult],
    baseline_duration: float,
    timeout: float,
    started: float,
    mutant_set: MutantSet,
) -> None:
    partial = RunSummary(
        results=list(results),
        baseline_duration=baseline_duration,
        timeout=timeout,
        total_duration=time.monotonic() - started,
        meta={"partial": True, "generation": mutant_set.meta},
    )
    partial.write(path)


def parse_test_command(value: Optional[str]) -> Tuple[List[str], bool]:
    """Turn a ``--test-command`` string into an argv list.

    Returns ``(argv, use_shell)``; shell mode is selected when the command
    contains shell metacharacters, so ``"pytest && echo done"`` still works.
    """
    if not value:
        return list(DEFAULT_TEST_COMMAND), False
    if any(character in value for character in "|&;<>$`"):
        return [value], True
    return shlex.split(value), False


def resolve_test_command(value: Optional[str], project_root: Path) -> Tuple[List[str], bool]:
    """Parse a configured test command and anchor it at the project root.

    - ``{project}`` is replaced by the (shell-quoted) project root.
    - A relative executable such as ``.venv/bin/python`` is made absolute
      against the project root. Tests run with the project as working
      directory, but ``--jobs`` workers run in copies that exclude ``.venv``, so
      a relative interpreter would not exist there.
    - ``os.path.abspath`` is used, never ``resolve()``: a virtualenv's python is
      a symlink to the base interpreter, and following it would run the tests
      outside the virtualenv, without the project's dependencies.
    """
    if value:
        value = value.replace("{project}", shlex.quote(str(project_root)))
    command, use_shell = parse_test_command(value)
    if not use_shell and command:
        executable = command[0]
        if "/" in executable and not os.path.isabs(executable):
            command[0] = os.path.abspath(os.path.join(str(project_root), executable))
        if os.path.isabs(command[0]) and not os.path.exists(command[0]):
            raise FileNotFoundError(
                "test command executable not found: {} - create the project's virtual "
                "environment first".format(command[0])
            )
    return command, use_shell
