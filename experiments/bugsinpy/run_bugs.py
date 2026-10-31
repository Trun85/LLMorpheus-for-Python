#!/usr/bin/env python3
"""Run LLMorpheus over BugsInPy++ bugs, one fixed version per bug.

Runs inside the llmorpheus-bugsinpy container (see README.md). For each selected
bug: check out the fixed version, build its conda env the way bugsinpy-testall
does, make the bug's test file green by deselecting tests that already fail,
mutate only the source files the real patch touched, then re-run the
bug-revealing tests against every mutant that lands on a patched line.

One JSON file per bug under results/bugs/, so an interrupted run resumes where
it stopped and never pays twice for a finished bug.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from llmorpheus.execution.runner import Workspace, _copy_project  # noqa: E402
from llmorpheus.generation.mutants import Mutant  # noqa: E402

BUGSINPY = Path("/home/user/BugsInPy")
WORK = Path("/work")
RESULTS = HERE / "results"
CONDA = "/opt/conda/bin/conda"

# Small, quick suites first, so a budget that runs out still covers the most bugs.
PROJECT_ORDER = [
    "PySnooper", "cookiecutter", "httpie", "sanic", "tqdm", "thefuck", "fastapi",
    "black", "spacy", "tornado", "youtube-dl", "luigi", "scrapy", "ansible",
    "keras", "matplotlib", "pandas",
]
# Their tests start servers or bind fixed ports: parallel workers would collide
# and report false kills.
SERIAL_PROJECTS = {"sanic", "tornado", "scrapy", "luigi", "httpie", "fastapi"}
# deepinfra/fp8/flex for deepseek/deepseek-v4.1-flash, USD per token (2026-10-20).
PRICE_IN, PRICE_OUT = 0.112e-6, 0.336e-6
BUDGET_MARKERS = ("HTTP 402", "Insufficient", "limit exceeded", "Key limit", "credits")


def log(message: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{}] {}".format(stamp, message), flush=True)


# -- shell -------------------------------------------------------------------
class Shell:
    """Runs commands, appending their full output to one bug's log file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, command, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None, label: str = "") -> Tuple[Optional[int], str, float, bool]:
        """Returns (exit code, output, seconds, timed out)."""
        shell = isinstance(command, str)
        started = time.monotonic()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n$ {}  [{}]\n".format(command if shell else " ".join(map(shlex.quote, command)), label))
        process = subprocess.Popen(
            ["bash", "-c", command] if shell else command,
            cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        timed_out = False
        try:
            raw, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError:
                pass
            raw, _ = process.communicate()
        output = (raw or b"").decode("utf-8", errors="replace")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(output[-200_000:])
            handle.write("\n[exit {} in {:.1f}s{}]\n".format(
                process.returncode, time.monotonic() - started, ", TIMED OUT" if timed_out else ""))
        return process.returncode, output, time.monotonic() - started, timed_out


class Skip(Exception):
    def __init__(self, stage: str, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.stage, self.reason, self.detail = stage, reason, detail[-3000:]


class BudgetExhausted(Exception):
    pass


class NeedsBudget(Exception):
    """The next bug is estimated to cost more than the budget left."""

    def __init__(self, estimate: float) -> None:
        super().__init__(estimate)
        self.estimate = estimate


# -- BugsInPy metadata -------------------------------------------------------
def read_info(path: Path) -> Dict[str, str]:
    info = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r'\s*([A-Za-z_]+)\s*=\s*"?(.*?)"?\s*$', line)
        if match:
            info[match.group(1)] = match.group(2)
    return info


def bugsinpy_index() -> Dict[Tuple[str, int], str]:
    """BugsInPy++'s own reproduction of each fixed version: pass, fail or error."""
    index = {}
    path = BUGSINPY / "new-conda-fixed.csv"
    for line in path.read_text().splitlines()[1:] if path.exists() else ():
        parts = line.strip().split(",")
        if len(parts) == 4 and parts[1].isdigit():
            index[(parts[0], int(parts[1]))] = parts[3]
    return index


def all_bugs() -> Dict[str, List[int]]:
    bugs = {}
    for project in PROJECT_ORDER:
        folder = BUGSINPY / "projects" / project / "bugs"
        bugs[project] = sorted(int(p.name) for p in folder.iterdir() if p.name.isdigit())
    return bugs


def select_bugs(fraction: float, seed: int, projects: Optional[List[str]]) -> List[Tuple[str, int]]:
    """ceil(fraction * n) bugs per project, drawn with a per-project seeded RNG."""
    chosen = []
    for project, ids in all_bugs().items():
        if projects and project not in projects:
            continue
        count = min(len(ids), math.ceil(len(ids) * fraction))
        sample = random.Random("{}-{}".format(seed, project)).sample(ids, count)
        chosen.extend((project, bug) for bug in sorted(sample))
    return chosen


def parse_patch(text: str) -> Dict[str, List[dict]]:
    """Per file, its hunks: lines added in the fixed version and lines removed from the buggy one."""
    files: Dict[str, List[dict]] = {}
    current = None
    hunk = None
    new_line = 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            current = line.split(" b/", 1)[1].strip() if " b/" in line else None
            hunk = None
            if current:
                files.setdefault(current, [])
            continue
        header = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if header and current:
            new_line = int(header.group(1))
            hunk = {"added": [], "deleted_at": [], "removed": []}
            files[current].append(hunk)
            continue
        if hunk is None or line.startswith(("+++", "---", "\\")):
            continue
        if line.startswith("+"):
            hunk["added"].append(new_line)
            new_line += 1
        elif line.startswith("-"):
            hunk["removed"].append(line[1:])
            hunk["deleted_at"].append(new_line)
        else:
            new_line += 1
    return files


def fix_lines(hunks: List[dict]) -> Set[int]:
    """Lines of the fixed file that carry the fix; a pure deletion marks the lines around it."""
    lines: Set[int] = set()
    for hunk in hunks:
        if hunk["added"]:
            lines.update(hunk["added"])
        else:
            for line in hunk["deleted_at"]:
                lines.update((line - 1, line))
    return lines


def fix_scopes(source: str, lines: List[int], window: int = 40) -> List[Tuple[int, int]]:
    """Line ranges to mutate around a fix: the innermost function holding each fixed
    line, else the innermost class, else `window` lines either side. Merged."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    scopes = []
    for node in ast.walk(tree) if tree else ():
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            scopes.append((start, node.end_lineno, isinstance(node, ast.ClassDef)))
    ranges = []
    for line in lines:
        holding = [s for s in scopes if s[0] <= line <= s[1]]
        functions = [s for s in holding if not s[2]] or holding
        if functions:
            start, end, _ = min(functions, key=lambda s: s[1] - s[0])
            ranges.append((start, end))
        else:
            ranges.append((max(1, line - window), line + window))
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return (any(p in ("test", "tests", "testing") for p in parts[:-1])
            or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


# -- budget ------------------------------------------------------------------
def openrouter_key() -> Optional[str]:
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def _openrouter(path: str, key: str) -> dict:
    request = urllib.request.Request("https://openrouter.ai/api/v1/" + path,
                                     headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response).get("data", {})


def key_status() -> Dict[str, Optional[float]]:
    """What can still be spent: the key's own limit, capped by the account's credits.

    .env is re-read every time, so swapping in a new key takes effect at once.
    """
    status: Dict[str, Optional[float]] = {"limit_remaining": None, "usage": None, "available": None}
    key = openrouter_key()
    if not key:
        return status
    try:  # a status check must never stop the run
        data = _openrouter("key", key)
        status["limit_remaining"], status["usage"] = data.get("limit_remaining"), data.get("usage")
    except Exception as error:
        log("    could not read the key's balance: {}".format(error))
        return status
    bounds = [status["limit_remaining"]] if status["limit_remaining"] is not None else []
    try:
        credits = _openrouter("credits", key)
        bounds.append(float(credits["total_credits"]) - float(credits["total_usage"]))
    except Exception:
        pass
    status["available"] = min(bounds) if bounds else None
    return status


def cost_per_prompt(project: Optional[str] = None) -> float:
    """Measured USD per prompt, with a 20% margin.

    Prompts carry the file around the placeholder, so their size - and price -
    differs by project: use the project's own average once it has 200 prompts,
    else the average over all projects, else a prior.
    """
    totals: Dict[Optional[str], List[float]] = {None: [0.0, 0.0]}
    for path in (RESULTS / "bugs").glob("*.json"):
        record = json.loads(path.read_text())
        generation = record.get("generation") or {}
        if generation.get("prompts") and record.get("est_cost_usd"):
            for key in (None, record.get("project")):
                totals.setdefault(key, [0.0, 0.0])
                totals[key][0] += record["est_cost_usd"]
                totals[key][1] += generation["prompts"]
    for key in (project, None):
        cost, prompts = totals.get(key, (0.0, 0.0))
        if prompts >= 200:
            return cost / prompts * 1.2
    return 0.0003 * 1.2


# -- one bug -----------------------------------------------------------------
class BugRun:
    def __init__(self, project: str, bug: int, args: argparse.Namespace) -> None:
        self.project, self.bug, self.args = project, bug, args
        self.name = "{}_{}".format(project, bug)
        self.bug_dir = BUGSINPY / "projects" / project / "bugs" / str(bug)
        self.info = read_info(self.bug_dir / "bug.info")
        self.project_info = read_info(BUGSINPY / "projects" / project / "project.info")
        self.workdir = WORK / "bugs" / self.name
        self.root = self.workdir / project
        self.env_name = self.name
        self.env_bin = Path("/opt/conda/envs") / self.env_name / "bin"
        self.python = self.env_bin / "python"
        self.shell = Shell(RESULTS / "logs" / "{}.log".format(self.name))
        self.timings: Dict[str, float] = {}
        self.record: Dict[str, object] = {
            "project": project, "bug_id": bug, "name": self.name,
            "python_version": self.info.get("python_version"),
            "buggy_commit": self.info.get("buggy_commit_id"),
            "fixed_commit": self.info.get("fixed_commit_id"),
            "test_files": [t for t in self.info.get("test_file", "").split(";") if t],
            "bug_tests": [l.strip() for l in (self.bug_dir / "run_test.sh").read_text().splitlines() if l.strip()],
            "github_url": self.project_info.get("github_url"),
            "bugsinpy_fixed_result": bugsinpy_index().get((project, bug)),
        }

    # environment for every test command: the bug's env first on PATH, as `conda activate` would
    def test_env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "CONDA_PREFIX", "CONDA_DEFAULT_ENV") and not k.endswith("_API_KEY")}
        env.update({
            "PATH": "{}:{}".format(self.env_bin, os.environ.get("PATH", "")),
            "CONDA_PREFIX": str(self.env_bin.parent), "CONDA_DEFAULT_ENV": self.env_name,
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        })
        if self.project == "matplotlib":
            env["MPLLOCALFREETYPE"] = "1"
        return env

    def timed(self, stage: str, started: float) -> None:
        self.timings[stage] = round(time.monotonic() - started, 1)

    # 1. checkout -----------------------------------------------------------
    def checkout(self) -> None:
        started = time.monotonic()
        url = (self.project_info.get("github_url") or "").strip()
        mirror = WORK / "mirrors" / "{}.git".format(self.project)
        if url and not mirror.exists():
            mirror.parent.mkdir(parents=True, exist_ok=True)
            code, out, _, _ = self.shell.run(["git", "clone", "--bare", "--quiet", url, str(mirror)],
                                             timeout=3600, label="mirror")
            if code != 0:
                shutil.rmtree(mirror, ignore_errors=True)
        for use_mirror in ([True, False] if mirror.exists() else [False]):
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir.mkdir(parents=True)
            git_config = ""
            if use_mirror:
                git_config = "git config --global url.{}.insteadOf {} && ".format(
                    shlex.quote(str(mirror)), shlex.quote(url))
            command = ("{}bugsinpy-checkout -p {} -i {} -v 1 -w {}; status=$?; "
                       "git config --global --unset-all url.{}.insteadOf; exit $status").format(
                git_config, shlex.quote(self.project), self.bug, shlex.quote(str(self.workdir)),
                shlex.quote(str(mirror)))
            code, out, _, _ = self.shell.run(command, timeout=3600, label="checkout")
            if code == 0 and (self.root / "bugsinpy_bug.info").exists():
                head = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()
                if head == self.info.get("buggy_commit_id"):
                    self.timed("checkout", started)
                    return
        raise Skip("checkout", "bugsinpy-checkout failed", out)

    # 2. environment ----------------------------------------------------------
    def build_env(self) -> None:
        started = time.monotonic()
        self.shell.run([CONDA, "env", "remove", "-n", self.env_name, "-y"], timeout=600, label="env remove")
        version = self.info.get("python_version", "")
        code, out, _, timed_out = self.shell.run(
            [CONDA, "create", "-n", self.env_name, "-y", "python={}".format(version), "pytest"],
            timeout=3600, label="conda create")
        if code != 0 or not self.python.exists():
            raise Skip("environment", "conda could not create Python {}".format(version), out)
        compile_command = "bugsinpy-compile -w {}".format(shlex.quote(str(self.root)))
        code, out, _, timed_out = self.shell.run(compile_command, cwd=self.root, timeout=7200,
                                                 label="bugsinpy-compile")
        if code != 0 and not timed_out:
            # pip builds each sdist with the newest build backend, which has moved on
            # since BugsInPy++ was verified: flit_core 4 dropped [tool.flit.metadata]
            # (fastapi), setuptools 71 needs a newer `packaging` than old pins give.
            # PIP_CONSTRAINT reaches pip's isolated build environments too.
            constraints = WORK / "build-constraints.txt"
            constraints.write_text("flit_core<4\nsetuptools<71\n")
            env = dict(os.environ, PIP_CONSTRAINT=str(constraints))
            code, retry, _, timed_out = self.shell.run(compile_command, cwd=self.root, env=env,
                                                       timeout=7200, label="bugsinpy-compile (constrained)")
            out = out[-3000:] + "\n--- retried with flit_core<4, setuptools<71 ---\n" + retry
            self.record["build_constraints"] = code == 0
        if code != 0:
            raise Skip("environment", "bugsinpy-compile failed" + (" (timed out)" if timed_out else ""), out)
        version_out = subprocess.run([str(self.python), "-m", "pytest", "--version"], capture_output=True,
                                     text=True, env=self.test_env(), cwd=str(self.root))
        self.record["pytest_version"] = (version_out.stdout + version_out.stderr).strip().splitlines()[-1:] or ["?"]
        self.timed("environment", started)

    # 3. the bug-revealing tests must pass on the fixed version ----------------
    def bug_tests_pass(self, root: Path, timeout: float) -> Tuple[bool, str]:
        outputs = []
        for command in self.record["bug_tests"]:
            code, out, _, timed_out = self.shell.run(command, cwd=root, env=self.test_env(),
                                                     timeout=timeout, label="bug test")
            outputs.append(out[-1500:])
            if code != 0 or timed_out:
                return False, "\n".join(outputs)
        return True, "\n".join(outputs)

    def check_bug_tests(self) -> None:
        started = time.monotonic()
        passed, out = self.bug_tests_pass(self.root, timeout=1800)
        if not passed and self.restore_pins():
            passed, retry = self.bug_tests_pass(self.root, timeout=1800)
            out = out + "\n--- after restoring requirements.txt pins ---\n" + retry
        self.timed("bug_tests_fixed", started)
        self.bug_test_seconds = max(5.0, self.timings["bug_tests_fixed"])
        if not passed:
            raise Skip("bug tests", "the bug-revealing test fails on the fixed version", out)

    def restore_pins(self) -> bool:
        """Put back the versions requirements.txt pins; True if anything changed.

        bugsinpy-compile installs pytest-cov and pytest-json-report after the bug's
        requirements, and they drag pytest past its pin - thefuck pins 3.10.1 and
        ends up on 7.4, whose removal of Node.get_marker breaks its conftest. Only
        used when the env as compiled cannot run the bug's own test.
        """
        lines = (self.root / "bugsinpy_requirements.txt").read_text(errors="replace").splitlines()
        pins = [l.strip() for l in lines if re.match(r"^[A-Za-z0-9_.\-\[\]]+==[^\s;#]+$", l.strip())]
        if not pins:
            return False
        wanted = {re.split(r"[=\[]", p)[0].lower().replace("_", "-") for p in pins}
        pinned = self.workdir / "pins.txt"
        pinned.write_text("\n".join(pins) + "\n")
        self.shell.run([str(self.python), "-m", "pip", "install", "--no-deps", "-r", str(pinned)],
                       env=self.test_env(), timeout=3600, label="restore pins")
        extras = [p for p in ("pytest-json-report", "pytest-metadata", "pytest-cov") if p not in wanted]
        self.shell.run([str(self.python), "-m", "pip", "uninstall", "-y", *extras],
                       env=self.test_env(), timeout=600, label="drop compile plugins")
        version = subprocess.run([str(self.python), "-m", "pytest", "--version"], capture_output=True,
                                 text=True, env=self.test_env(), cwd=str(self.root))
        self.record["pins_restored"] = True
        self.record["pytest_version"] = (version.stdout + version.stderr).strip().splitlines()[-1:] or ["?"]
        return True

    # 4. baseline: make the bug's test file green -----------------------------
    def pytest_command(self, deselect: List[str], fail_fast: bool, report: bool) -> List[str]:
        command = [str(self.python), "-m", "pytest", *self.record["test_files"], "-q", "-p", "no:cacheprovider"]
        if fail_fast:
            command.append("-x")
        if report:
            command.append("-rfE")
        for node in deselect:
            command += ["--deselect", node]
        return command

    def baseline(self) -> None:
        started = time.monotonic()
        missing = [t for t in self.record["test_files"] if not (self.root / t).exists()]
        if missing:
            raise Skip("baseline", "test file missing: {}".format(", ".join(missing)))
        # bug.info's test_file also lists inputs the tests read - black's tests/data/*.py -
        # which pytest would import as test modules and fail on. Keep what pytest collects.
        kept, dropped = [], []
        for test_file in self.record["test_files"]:
            code, _, _, _ = self.shell.run(
                [str(self.python), "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", test_file],
                cwd=self.root, env=self.test_env(), timeout=600, label="collect")
            (kept if code == 0 else dropped).append(test_file)
        if dropped:
            self.record["test_files_dropped"] = dropped
            self.record["test_files"] = kept
        if not kept:
            raise Skip("baseline", "pytest collects no tests from any test file: {}".format(", ".join(dropped)))
        code, out, seconds, timed_out = self.shell.run(
            self.pytest_command([], fail_fast=False, report=True), cwd=self.root, env=self.test_env(),
            timeout=3600, label="baseline")
        if timed_out:
            raise Skip("baseline", "the test file takes over an hour", out)
        if code == 5:
            raise Skip("baseline", "pytest collected no tests from the test file", out)
        failing = []
        for line in out.splitlines():
            # pytest < 5 writes FAIL in the short summary, later versions FAILED.
            match = re.match(r"^(FAILED|FAIL|ERROR) (.+?)(?: - .*)?$", line.strip())
            if match:
                failing.append(match.group(2).strip())
        broken_files = [f for f in failing if "::" not in f]
        if broken_files and set(broken_files) >= set(self.record["test_files"]):
            raise Skip("baseline", "the test file does not import on the fixed version", out)
        deselect = sorted(set(f for f in failing if "::" in f))
        for attempt in (1, 2):
            code, out, seconds, timed_out = self.shell.run(
                self.pytest_command(deselect, fail_fast=True, report=True), cwd=self.root,
                env=self.test_env(), timeout=3600, label="baseline after deselect")
            if code == 0:
                break
            for line in out.splitlines():
                match = re.match(r"^(FAILED|FAIL|ERROR) (.+?)(?: - .*)?$", line.strip())
                if match and "::" in match.group(2):
                    deselect = sorted(set(deselect) | {match.group(2).strip()})
        else:
            raise Skip("baseline", "the test file is still red after deselecting {} test(s)".format(len(deselect)), out)
        if code != 0:
            raise Skip("baseline", "the test file is still red after deselecting {} test(s)".format(len(deselect)), out)
        summary = re.findall(r"(\d+) (passed|failed|skipped|deselected|error|xfailed|xpassed)", out)
        self.record["baseline_counts"] = {kind: int(n) for n, kind in summary}
        self.record["deselected_tests"] = deselect
        self.record["baseline_seconds"] = round(seconds, 2)
        self.deselect = deselect
        self.timed("baseline", started)

    # 5. which files to mutate ------------------------------------------------
    def choose_files(self) -> None:
        patch = parse_patch((self.bug_dir / "bug_patch.txt").read_text(encoding="utf-8", errors="replace"))
        files = [f for f in patch if f.endswith(".py") and not is_test_path(f) and (self.root / f).is_file()]
        if not files:
            raise Skip("files", "the patch touches no non-test Python source file")
        self.patch = patch
        self.record["mutated_files"] = files
        self.record["fix_lines"] = {f: sorted(fix_lines(patch[f])) for f in files}

    # 5b. narrow huge files, then estimate the cost, before the env is built -----
    def selection_args(self) -> List[str]:
        args: List[str] = []
        for path in self.record["mutated_files"]:
            args += ["--src", path]
        for path, ranges in (self.record.get("focus_ranges") or {}).items():
            args += ["--lines={}:{}-{}".format(path, start, end) for start, end in ranges]
        return args

    def count_locations(self, label: str) -> Tuple[int, Dict[str, int]]:
        command = [sys.executable, "-m", "llmorpheus", "locations", "-C", str(self.root),
                   "--env-file", str(REPO / ".env"), *self.selection_args()]
        _, out, _, _ = self.shell.run(command, cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO)),
                                      timeout=600, label=label)
        total = re.search(r"==> (\d+) location", out)
        per_file = {m.group(1): int(m.group(2))
                    for m in re.finditer(r"^(\S.*?) \((\d+) locations?\)$", out, re.MULTILINE)}
        return (int(total.group(1)) if total else 0), per_file

    def focus(self) -> None:
        """Files over FOCUS_THRESHOLD locations are mutated only in the scopes the fix touches."""
        _, per_file = self.count_locations("locations")
        self.record["full_file_locations"] = per_file
        ranges = {}
        for path, count in per_file.items():
            if count > self.args.focus_threshold and self.record["fix_lines"].get(path):
                source = (self.root / path).read_text(encoding="utf-8", errors="replace")
                ranges[path] = fix_scopes(source, self.record["fix_lines"][path])
        self.record["focus_ranges"] = ranges
        self.record["scope"] = {path: ("fix scopes" if path in ranges else "whole file") for path in per_file}

    def estimate(self, available: Optional[float]) -> None:
        """One prompt per placeholder location; stop here if the budget cannot cover it."""
        prompts, _ = self.count_locations("locations (selected)")
        if self.args.max_prompts:
            prompts = min(prompts, self.args.max_prompts)
        estimate = round(prompts * cost_per_prompt(self.project), 5)
        self.record["estimated_prompts"], self.record["estimated_cost_usd"] = prompts, estimate
        if available is not None and estimate + self.args.budget_floor > available:
            raise NeedsBudget(estimate)

    # 6. may mutants run in parallel copies? ------------------------------------
    def choose_jobs(self) -> int:
        if self.args.jobs <= 1 or self.project in SERIAL_PROJECTS:
            return 1
        # A worker tests a copy of the project. If the tests import the package from
        # somewhere else (an install outside the tree), a copy's mutant would never
        # run and would survive falsely. Plant an import-time error in a copy: the
        # tests must notice it, or the bug runs serially.
        copy = Path("/tmp") / "canary-{}".format(self.name)
        shutil.rmtree(copy, ignore_errors=True)
        _copy_project(self.root, copy)
        target = copy / self.record["mutated_files"][0]
        target.write_text('raise RuntimeError("llmorpheus canary")\n' + target.read_text(encoding="utf-8"),
                          encoding="utf-8")
        code, _, _, _ = self.shell.run(self.pytest_command(self.deselect, True, False), cwd=copy,
                                       env=self.test_env(), timeout=3600, label="canary")
        shutil.rmtree(copy, ignore_errors=True)
        jobs = self.args.jobs if code != 0 else 1
        self.record["canary_detected"] = code != 0
        return jobs

    # 7. llmorpheus -------------------------------------------------------------
    def mutate(self) -> None:
        started = time.monotonic()
        jobs = self.choose_jobs()
        self.record["jobs"] = jobs
        out_dir = self.root / ".llmorpheus" / "bugsinpy"
        cache = self.root / ".llmorpheus" / "cache"
        cache.parent.mkdir(parents=True, exist_ok=True)
        shared = WORK / "llm-cache"
        shared.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            cache.symlink_to(shared, target_is_directory=True)
        test_command = " ".join(shlex.quote(part) for part in self.pytest_command(self.deselect, True, False))
        command = [sys.executable, "-m", "llmorpheus", "all", "-C", str(self.root),
                   "--env-file", str(REPO / ".env"), "--out", str(out_dir), "--no-save-prompts",
                   "--test-command", test_command, "--jobs", str(jobs),
                   "--concurrency", str(self.args.concurrency), *self.selection_args()]
        for key in ("PATH", "PYTHONNOUSERSITE", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "MPLLOCALFREETYPE"):
            if key in self.test_env():
                command += ["--test-env", "{}={}".format(key, self.test_env()[key])]
        if self.args.max_prompts:
            command += ["--limit", str(self.args.max_prompts)]
        env = dict(os.environ, PYTHONPATH=str(REPO))
        code, out, seconds, timed_out = self.shell.run(command, cwd=REPO, env=env,
                                                       timeout=self.args.max_hours * 3600, label="llmorpheus")
        self.record["llmorpheus_exit"] = code
        self.record["llmorpheus_timed_out"] = timed_out
        self.timed("llmorpheus", started)
        raw = RESULTS / "raw" / self.name
        raw.mkdir(parents=True, exist_ok=True)
        for name in ("mutants.json", "results.json", "report.html"):
            if (out_dir / name).exists():
                shutil.copy2(out_dir / name, raw / name)
        if any(marker in out for marker in BUDGET_MARKERS) and not (raw / "results.json").exists():
            raise BudgetExhausted(out[-2000:])
        if not (raw / "mutants.json").exists():
            raise Skip("llmorpheus", "llmorpheus produced no mutants.json (exit {})".format(code), out)
        mutants = json.loads((raw / "mutants.json").read_text())
        stats = mutants.get("stats", {})
        self.record["locations"] = sum((mutants.get("meta", {}).get("locations_by_kind") or {}).values())
        self.record["generation"] = stats
        self.record["est_cost_usd"] = round(stats.get("prompt_tokens", 0) * PRICE_IN
                                            + stats.get("completion_tokens", 0) * PRICE_OUT, 5)
        if stats.get("prompts") and stats.get("failed_prompts") == stats.get("prompts"):
            if any(marker in out for marker in BUDGET_MARKERS):
                raise BudgetExhausted(out[-2000:])
            raise Skip("llmorpheus", "every prompt failed", out)
        if not (raw / "results.json").exists():
            if stats.get("mutants", 0) == 0:
                self.record["summary"] = {"counts": {}, "mutation_score": None}
                self.results = []
                return
            raise Skip("llmorpheus", "mutants were generated but never run (exit {})".format(code), out)
        results = json.loads((raw / "results.json").read_text())
        self.record["summary"] = results.get("summary", {})
        self.record["partial"] = bool(results.get("meta", {}).get("partial")) or timed_out
        self.results = results.get("results", [])

    # 8. coupling with the real bug -------------------------------------------------
    def coupling(self) -> None:
        started = time.monotonic()
        rows = []
        lines_by_file = {f: set(v) for f, v in self.record["fix_lines"].items()}
        on_fix = [r for r in self.results
                  if r["file"] in lines_by_file
                  and lines_by_file[r["file"]] & set(range(r["start"]["line"], r["end"]["line"] + 1))]
        workspace = Workspace(self.root, sorted({r["file"] for r in on_fix}))
        timeout = max(30.0, 3 * self.bug_test_seconds)
        for result in on_fix:
            mutant = Mutant.from_dict(result)
            original = workspace.original(mutant.file) or ""
            mutated_region, removed = self.regions(mutant, original)
            problem = workspace.apply(mutant)
            if problem:
                failed, detail = None, problem
            else:
                try:
                    passed, detail = self.bug_tests_pass(self.root, timeout)
                    failed = not passed
                finally:
                    workspace.restore(mutant)
            normal = re.sub(r"\s+", "", mutated_region)
            textual = bool(normal) and any(normal == re.sub(r"\s+", "", text) for text in removed if text.strip())
            rows.append({"id": result["id"], "bug_test_failed": failed, "textual_match": textual,
                         "mutated_region": mutated_region, "buggy_lines": [t for t in removed if t.strip()][:12]})
        self.record["coupling"] = rows
        self.timed("coupling", started)

    def regions(self, mutant: Mutant, original: str) -> Tuple[str, List[str]]:
        """The mutated source lines, and the buggy-version lines removed by hunks touching them."""
        mutated = mutant.apply(original)
        before, after = original.splitlines(), mutated.splitlines()
        delta = len(after) - len(before)
        region = "\n".join(after[mutant.start.line - 1: mutant.end.line + delta])
        span = set(range(mutant.start.line, mutant.end.line + 1))
        removed: List[str] = []
        for hunk in self.patch.get(mutant.file, []):
            if span & fix_lines([hunk]):
                removed += hunk["removed"]
                if len(hunk["removed"]) > 1:
                    removed.append("".join(hunk["removed"]))
        return region, removed

    # -- driver ----------------------------------------------------------------
    def run(self, available: Optional[float]) -> Dict[str, object]:
        started = time.monotonic()
        self.record["started_at"] = dt.datetime.now().isoformat(timespec="seconds")
        try:
            self.checkout()
            self.choose_files()
            self.focus()
            self.estimate(available)
            self.build_env()
            self.check_bug_tests()
            self.baseline()
            self.mutate()
            self.coupling()
            self.record["status"] = "partial" if self.record.get("partial") else "done"
        except Skip as skip:
            self.record.update(status="skipped", stage=skip.stage, reason=skip.reason, detail=skip.detail)
        finally:
            self.record["timings"] = self.timings
            self.record["total_seconds"] = round(time.monotonic() - started, 1)
            self.record["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        return self.record

    def cleanup(self) -> None:
        if self.args.keep:
            return
        self.shell.run([CONDA, "env", "remove", "-n", self.env_name, "-y"], timeout=900, label="cleanup env")
        self.shell.run([CONDA, "clean", "--tarballs", "-y"], timeout=900, label="cleanup tarballs")
        shutil.rmtree(self.workdir, ignore_errors=True)


def merge_results(record: Dict[str, object], results: List[dict]) -> None:
    coupling = {row["id"]: row for row in record.get("coupling", [])}
    mutants = []
    for result in results:
        row = {key: result.get(key) for key in
               ("id", "file", "kind", "start", "end", "original", "replacement", "status", "duration", "explanation")}
        extra = coupling.get(result["id"])
        row["on_fix"] = extra is not None
        if extra:
            row.update(bug_test_failed=extra["bug_test_failed"], textual_match=extra["textual_match"],
                       mutated_region=extra["mutated_region"], buggy_lines=extra["buggy_lines"])
        mutants.append(row)
    record["mutants"] = mutants
    record.pop("coupling", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fraction", type=float, default=1 / 3, help="share of each project's bugs to run")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--projects", nargs="*", help="only these projects")
    parser.add_argument("--bugs", nargs="*", help="only these bugs, as project_id")
    parser.add_argument("--jobs", type=int, default=3, help="parallel mutant workers where safe")
    parser.add_argument("--concurrency", type=int, default=8, help="parallel LLM requests")
    parser.add_argument("--max-prompts", type=int, default=0, help="per-bug prompt cap (0: none)")
    parser.add_argument("--focus-threshold", type=int, default=800,
                        help="files with more locations are mutated only in the fix's scopes")
    parser.add_argument("--max-hours", type=float, default=6.0, help="per-bug llmorpheus wall-clock cap")
    parser.add_argument("--budget-floor", type=float, default=0.10, help="keep this many USD in reserve")
    parser.add_argument("--poll-minutes", type=float, default=5.0, help="balance check interval while paused")
    parser.add_argument("--keep", action="store_true", help="keep envs and checkouts")
    parser.add_argument("--list", action="store_true", help="print the selection and exit")
    args = parser.parse_args()

    # selection.json always holds the whole sample, so the export can show what has
    # not run yet; --projects and --bugs only narrow what this invocation runs.
    full = select_bugs(args.fraction, args.seed, None)
    selection = [(p, b) for p, b in full if not args.projects or p in args.projects]
    if args.bugs:
        wanted = set(args.bugs)
        selection = [(p, b) for p, b in full if "{}_{}".format(p, b) in wanted] or [
            (name.rsplit("_", 1)[0], int(name.rsplit("_", 1)[1])) for name in args.bugs]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bugs").mkdir(exist_ok=True)
    (RESULTS / "selection.json").write_text(json.dumps({
        "fraction": args.fraction, "seed": args.seed, "bugs": ["{}_{}".format(p, b) for p, b in full],
        "bugsinpy_commit": subprocess.run(["git", "-C", str(BUGSINPY), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
        "llmorpheus_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip(),
        "price_per_token_usd": {"input": PRICE_IN, "output": PRICE_OUT},
    }, indent=2) + "\n")
    if args.list:
        for project, bug in selection:
            print("{}_{}".format(project, bug))
        print("{} bugs".format(len(selection)))
        return 0

    log("{} bug(s) selected; results in {}".format(len(selection), RESULTS))
    index = 0
    while index < len(selection):
        if (RESULTS / "STOP").exists():
            log("results/STOP found - stopping between bugs")
            return 0
        project, bug = selection[index]
        name = "{}_{}".format(project, bug)
        target = RESULTS / "bugs" / "{}.json".format(name)
        if target.exists():
            index += 1
            continue
        status = key_status()
        available = status["available"]
        if available is not None and available < args.budget_floor:
            wait_for_budget(name, args.budget_floor, None, args)
            continue
        log("[{}/{}] {}  (${} available)".format(index + 1, len(selection), name,
                                                 "?" if available is None else "{:.4f}".format(available)))
        run = BugRun(project, bug, args)
        try:
            record = run.run(available)
        except NeedsBudget as need:
            run.cleanup()
            wait_for_budget(name, need.estimate + args.budget_floor, need.estimate, args)
            continue
        except BudgetExhausted:
            # Completions already paid for are cached, so the retry only pays for the rest.
            log("    the budget ran out during {}; it will be retried".format(name))
            run.cleanup()
            estimate = run.record.get("estimated_cost_usd") or 0.0
            wait_for_budget(name, estimate + args.budget_floor, estimate, args)
            continue
        except Exception:
            record = dict(run.record, status="skipped", stage="orchestrator", reason="unexpected error",
                          detail=traceback.format_exc()[-3000:])
        merge_results(record, getattr(run, "results", []))
        after = key_status()
        record["key_usage_before"], record["key_usage_after"] = status["usage"], after["usage"]
        record["cost_per_prompt_used"] = round(cost_per_prompt(project), 7)
        target.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
        summary = record.get("summary") or {}
        log("    {} {}{}  mutants={} score={} est=${}  {:.0f}s".format(
            record["status"], record.get("stage", ""), (": " + record["reason"]) if record.get("reason") else "",
            len(record.get("mutants", [])), summary.get("mutation_score"), record.get("est_cost_usd"),
            record.get("total_seconds", 0)))
        run.cleanup()
        index += 1
    log("selection finished")
    return 0


def export_workbook() -> None:
    code = subprocess.run([sys.executable, str(HERE / "export_excel.py")], capture_output=True, text=True)
    log("    " + (code.stdout.strip() or code.stderr.strip()[-500:]))


def wait_for_budget(name: str, needed: float, estimate: Optional[float], args: argparse.Namespace) -> None:
    """Wrap up - export the workbook, say what is missing - then wait for a top-up."""
    available = key_status()["available"]
    shortfall = max(0.0, needed - (available or 0.0))
    note = RESULTS / "WAITING_FOR_BUDGET.txt"
    note.write_text("\n".join([
        "Paused {} before bug {}.".format(dt.datetime.now().isoformat(timespec="seconds"), name),
        "Estimated cost of that bug: {}".format("${:.4f}".format(estimate) if estimate is not None else "-"),
        "Available now (key limit, capped by account credits): {}".format(
            "?" if available is None else "${:.4f}".format(available)),
        "Needed to start it: ${:.4f} - top up, or raise the key's limit, by about ${:.2f}.".format(needed, shortfall),
        "The run checks every {:g} minutes and resumes by itself; a new key in .env is picked up too.".format(
            args.poll_minutes),
        "To stop waiting instead: touch experiments/bugsinpy/results/STOP",
    ]) + "\n")
    log("WAITING FOR BUDGET before {}: needs ${:.4f}, {} available - top up by ~${:.2f}".format(
        name, needed, "?" if available is None else "${:.4f}".format(available), shortfall))
    export_workbook()
    while True:
        time.sleep(args.poll_minutes * 60)
        if (RESULTS / "STOP").exists():
            log("results/STOP found while waiting - exiting")
            sys.exit(0)
        available = key_status()["available"]
        if available is not None and available >= needed:
            note.unlink(missing_ok=True)
            log("budget restored (${:.4f} available) - resuming with {}".format(available, name))
            return


if __name__ == "__main__":
    sys.exit(main())
