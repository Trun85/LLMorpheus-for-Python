"""End-to-end tests: generate with the mock provider, then run and report."""

import json
import sys
from pathlib import Path

import pytest

from llmorpheus.generation.generator import GeneratorConfig, generate
from llmorpheus.llm import build_client
from llmorpheus.execution.report import text_summary, write_html_report
from llmorpheus.execution.runner import KILLED, SURVIVED, RunnerConfig, run_mutants

MODULE = """def classify(value):
    if value > 10:
        return "big"
    if value > 0:
        return "small"
    return "none"
"""

# Covers the ``> 10`` boundary but not the ``> 0`` one, so some mutants survive.
TESTS = """from mod import classify


def test_big():
    assert classify(11) == "big"


def test_boundary_is_not_big():
    assert classify(10) == "small"


def test_none():
    assert classify(-5) == "none"
"""


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(MODULE, encoding="utf-8")
    (tmp_path / "test_mod.py").write_text(TESTS, encoding="utf-8")
    (tmp_path / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent))\n",
        encoding="utf-8",
    )
    return tmp_path


def generate_for(project: Path):
    config = GeneratorConfig(
        project_root=project,
        sources=["mod.py"],
        concurrency=1,
        save_prompts=False,
    )
    return generate(config, build_client("mock"))


def test_generation_produces_valid_mutants(project: Path):
    mutant_set = generate_for(project)
    assert mutant_set.mutants
    assert mutant_set.stats.mutants == len(mutant_set.mutants)
    source = (project / "mod.py").read_text(encoding="utf-8")
    for mutant in mutant_set.mutants:
        compile(mutant.apply(source), "mod.py", "exec")  # every mutant parses
        assert mutant.replacement.strip() != mutant.original.strip()


def test_run_classifies_and_restores_the_source(project: Path):
    mutant_set = generate_for(project)
    before = (project / "mod.py").read_text(encoding="utf-8")
    summary = run_mutants(
        mutant_set,
        RunnerConfig(
            project_root=project,
            test_command=[sys.executable, "-m", "pytest", "-x", "-q"],
            timeout=60,
        ),
    )
    assert (project / "mod.py").read_text(encoding="utf-8") == before
    counts = summary.counts()
    assert counts[KILLED] > 0
    assert counts[SURVIVED] > 0
    assert 0 < summary.mutation_score < 100
    assert len(summary.results) == len(mutant_set.mutants)


def test_reports_are_written(project: Path, tmp_path: Path):
    mutant_set = generate_for(project)
    summary = run_mutants(
        mutant_set,
        RunnerConfig(
            project_root=project,
            test_command=[sys.executable, "-m", "pytest", "-x", "-q"],
            timeout=60,
        ),
    )
    assert "mutation" not in text_summary(summary).lower() or "total" in text_summary(summary)

    results_path = tmp_path / "results.json"
    summary.write(results_path)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["summary"]["counts"][KILLED] > 0

    html_path = tmp_path / "report.html"
    write_html_report(summary, html_path, project_root=project)
    html = html_path.read_text(encoding="utf-8")
    assert "__DATA__" not in html
    assert "mutation report" in html


def test_baseline_failure_is_reported(project: Path):
    (project / "test_mod.py").write_text(
        "def test_broken():\n    assert False\n", encoding="utf-8"
    )
    mutant_set = generate_for(project)
    with pytest.raises(RuntimeError):
        run_mutants(
            mutant_set,
            RunnerConfig(
                project_root=project,
                test_command=[sys.executable, "-m", "pytest", "-x", "-q"],
                timeout=60,
            ),
        )
