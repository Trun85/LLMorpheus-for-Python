"""Regression tests for the failure modes found in code review.

Each test here pins a bug that produced *silently wrong results* rather than a
crash, which is the dangerous kind for a tool that prints a mutation score.
"""

import os
import stat
import sys
from pathlib import Path

import pytest

from llmorpheus.analysis.locations import IF_CONDITION, find_locations
from llmorpheus.analysis.project import FileSelector
from llmorpheus.execution import runner as runner_module
from llmorpheus.execution.runner import (
    ERROR,
    KILLED,
    SURVIVED,
    TIMEOUT,
    MutantResult,
    RunnerConfig,
    TestOutcome,
    TestRunner,
    Workspace,
    _evaluate,
    _stale_files,
    restore_all,
    run_mutants,
)
from llmorpheus.generation.mutants import GenerationStats, Mutant, MutantSet, file_digest

MODULE = 'def classify(value):\n    if value > 10:\n        return "big"\n    return "none"\n'


def make_mutant(source: str, replacement: str = "value > 11") -> Mutant:
    location = [item for item in find_locations(source, "mod.py") if item.kind == IF_CONDITION][0]
    return Mutant(
        id="m1",
        file="mod.py",
        kind=location.kind,
        start=location.start,
        end=location.end,
        original=location.original,
        replacement=replacement,
    )


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(MODULE, encoding="utf-8")
    (tmp_path / "test_mod.py").write_text(
        'from mod import classify\n\n\ndef test_big():\n    assert classify(11) == "big"\n',
        encoding="utf-8",
    )
    (tmp_path / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent))\n",
        encoding="utf-8",
    )
    return tmp_path


# -- stale mutants ---------------------------------------------------------
def test_apply_refuses_a_span_that_no_longer_matches(project: Path):
    """Editing the file after generation must not splice into shifted offsets."""
    mutant = make_mutant(MODULE)
    (project / "mod.py").write_text('"""Added later."""\n\n\n' + MODULE, encoding="utf-8")

    workspace = Workspace(project, ["mod.py"])
    problem = workspace.apply(mutant)

    assert problem is not None
    assert "has changed since the mutants were generated" in problem
    # The file is untouched: no bare `value > 11` statement was written into it.
    assert (project / "mod.py").read_text(encoding="utf-8").count("value > 11") == 0


def test_stale_files_are_detected_from_the_recorded_digest(project: Path):
    mutant = make_mutant(MODULE)
    mutant_set = MutantSet(
        mutants=[mutant],
        stats=GenerationStats(),
        meta={"file_digests": {"mod.py": file_digest(MODULE)}},
    )
    assert _stale_files(mutant_set, project) == set()

    (project / "mod.py").write_text(MODULE + "\n# touched\n", encoding="utf-8")
    assert _stale_files(mutant_set, project) == {"mod.py"}


def test_run_skips_stale_mutants_instead_of_scoring_them(project: Path):
    mutant = make_mutant(MODULE)
    mutant_set = MutantSet(
        mutants=[mutant],
        stats=GenerationStats(),
        meta={"file_digests": {"mod.py": file_digest(MODULE)}},
    )
    (project / "mod.py").write_text('"""Added later."""\n\n\n' + MODULE, encoding="utf-8")
    before = (project / "mod.py").read_text(encoding="utf-8")

    summary = run_mutants(
        mutant_set,
        RunnerConfig(
            project_root=project,
            test_command=[sys.executable, "-m", "pytest", "-x", "-q"],
            timeout=60,
        ),
    )

    assert [result.status for result in summary.results] == [ERROR]
    assert summary.mutation_score == 0.0  # errors are excluded, not counted as killed
    assert (project / "mod.py").read_text(encoding="utf-8") == before


# -- sources outside the project root --------------------------------------
def test_source_outside_the_project_root_is_rejected(tmp_path: Path):
    project = tmp_path / "proj"
    outside = tmp_path / "lib"
    project.mkdir()
    outside.mkdir()
    (outside / "mod.py").write_text(MODULE, encoding="utf-8")

    selector = FileSelector(project_root=project, sources=[str(outside)])
    with pytest.raises(ValueError) as error:
        selector.find()
    assert "outside the project root" in str(error.value)


# -- timeout classification ------------------------------------------------
class _FakeRunner:
    def __init__(self, outcome: TestOutcome) -> None:
        self.outcome = outcome

    def run(self, cwd, timeout):
        return self.outcome


def test_negative_exit_code_is_killed_not_timed_out(project: Path):
    """A process killed by a signal (SIGHUP gives -1) is a killed mutant."""
    workspace = Workspace(project, ["mod.py"])
    outcome = TestOutcome(exit_code=-1, output="", duration=0.1, timed_out=False)

    result = _evaluate(make_mutant(MODULE), workspace, _FakeRunner(outcome), 60)

    assert result.status == KILLED


def test_timeout_is_reported_by_its_own_flag(project: Path):
    workspace = Workspace(project, ["mod.py"])
    outcome = TestOutcome(exit_code=-9, output="", duration=60.0, timed_out=True)

    result = _evaluate(make_mutant(MODULE), workspace, _FakeRunner(outcome), 60)

    assert result.status == TIMEOUT


def test_zero_exit_code_survives(project: Path):
    workspace = Workspace(project, ["mod.py"])
    outcome = TestOutcome(exit_code=0, output="", duration=0.1, timed_out=False)

    result = _evaluate(make_mutant(MODULE), workspace, _FakeRunner(outcome), 60)

    assert result.status == SURVIVED


def test_real_timeout_is_classified_as_timeout(tmp_path: Path):
    runner = TestRunner(
        RunnerConfig(
            project_root=tmp_path,
            test_command=[sys.executable, "-c", "import time; time.sleep(30)"],
        )
    )
    outcome = runner.run(tmp_path, timeout=1.0)
    assert outcome.timed_out is True


# -- restore safety --------------------------------------------------------
def test_failed_restore_keeps_the_emergency_entry(project: Path):
    """A restore that cannot write must stay recoverable by restore_all()."""
    mutant = make_mutant(MODULE)
    workspace = Workspace(project, ["mod.py"])
    path = project / "mod.py"
    assert workspace.apply(mutant) is None
    assert path in runner_module._DIRTY_FILES

    os.chmod(path, stat.S_IRUSR)  # read-only: the restoring write will fail
    try:
        with pytest.raises(OSError):
            workspace.restore(mutant)
        # Crucially still registered, so the atexit/signal hook can retry.
        assert path in runner_module._DIRTY_FILES
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    assert restore_all() == []
    assert path.read_text(encoding="utf-8") == MODULE


def test_restore_all_reports_what_it_could_not_restore(project: Path):
    mutant = make_mutant(MODULE)
    workspace = Workspace(project, ["mod.py"])
    path = project / "mod.py"
    workspace.apply(mutant)

    os.chmod(path, stat.S_IRUSR)
    try:
        assert restore_all() == [path]
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        path.write_text(MODULE, encoding="utf-8")
