#!/usr/bin/env python3
"""Export results/bugs/*.json (written by run_bugs.py) to one Excel workbook."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")
STATUS_FILLS = {
    "done": "E2EFDA", "partial": "FFF2CC", "skipped": "FCE4D6", "not run": "EDEDED",
    "killed": "E2EFDA", "survived": "FCE4D6", "timeout": "FFF2CC", "error": "EDEDED",
}


def clean(value):
    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)
        return value if len(value) <= 32000 else value[:32000] + "…"
    if isinstance(value, (list, tuple)):
        return clean("; ".join(str(v) for v in value))
    if isinstance(value, dict):
        return clean(json.dumps(value))
    return value


def write_sheet(book, title, columns, rows, status_column=None, wrap=()):
    sheet = book.create_sheet(title)
    sheet.append([name for name, _ in columns])
    for cell in sheet[1]:
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for row in rows:
        sheet.append([clean(row.get(key)) for _, key in columns])
    for index, (name, key) in enumerate(columns, 1):
        letter = get_column_letter(index)
        longest = max([len(str(name))] + [len(str(clean(r.get(key)) or "")) for r in rows[:500]])
        sheet.column_dimensions[letter].width = min(60 if key in wrap else 40, max(8, longest + 2))
        if key in wrap:
            for cell in sheet[letter][1:]:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        if key.endswith("_pct"):
            for cell in sheet[letter][1:]:
                cell.number_format = "0.0"
        if key.endswith("_usd"):
            for cell in sheet[letter][1:]:
                cell.number_format = "0.0000"
    if status_column:
        column = [key for _, key in columns].index(status_column) + 1
        for cells in sheet.iter_rows(min_row=2, min_col=column, max_col=column):
            colour = STATUS_FILLS.get(str(cells[0].value))
            if colour:
                cells[0].fill = PatternFill("solid", fgColor=colour)
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = sheet.dimensions
    return sheet


def bug_row(name, record):
    if record is None:
        project, bug = name.rsplit("_", 1)
        return {"project": project, "bug_id": int(bug), "status": "not run",
                "reason": "chưa chạy tới (hết ngân sách hoặc đang chạy)"}
    summary = record.get("summary") or {}
    counts = summary.get("counts") or {}
    generation = record.get("generation") or {}
    mutants = record.get("mutants") or []
    on_fix = [m for m in mutants if m.get("on_fix")]
    behavioural = [m for m in on_fix if m.get("bug_test_failed")]
    textual = [m for m in on_fix if m.get("textual_match")]
    timings = record.get("timings") or {}
    return {
        "project": record["project"], "bug_id": record["bug_id"], "status": record.get("status"),
        "stage": record.get("stage"), "reason": record.get("reason"),
        "python_version": record.get("python_version"), "fixed_commit": (record.get("fixed_commit") or "")[:12],
        "mutated_files": record.get("mutated_files"), "test_files": record.get("test_files"),
        "test_files_dropped": record.get("test_files_dropped"),
        "scope": "; ".join("{}: {}".format(f, s) for f, s in (record.get("scope") or {}).items()) or None,
        "focus_ranges": "; ".join("{}: {}".format(f, ", ".join("{}-{}".format(a, b) for a, b in r))
                                  for f, r in (record.get("focus_ranges") or {}).items()) or None,
        "full_locations": sum((record.get("full_file_locations") or {}).values()) or None,
        "deselected": len(record.get("deselected_tests") or []),
        "baseline_passed": (record.get("baseline_counts") or {}).get("passed"),
        "baseline_seconds": record.get("baseline_seconds"), "jobs": record.get("jobs"),
        "locations": record.get("locations"), "prompts": generation.get("prompts"),
        "failed_prompts": generation.get("failed_prompts"), "candidates": generation.get("candidates"),
        "mutants": len(mutants) if mutants else generation.get("mutants"),
        "killed": counts.get("killed"), "survived": counts.get("survived"),
        "timeout": counts.get("timeout"), "errored": counts.get("error"),
        "score_pct": summary.get("mutation_score"),
        "on_fix": len(on_fix) if record.get("status") in ("done", "partial") else None,
        "on_fix_killed": sum(1 for m in on_fix if m.get("status") in ("killed", "timeout")),
        "behavioural": len(behavioural), "textual": len(textual),
        "reproduced_behavioural": bool(behavioural) if on_fix else None,
        "reproduced_textual": bool(textual) if on_fix else None,
        "prompt_tokens": generation.get("prompt_tokens"), "completion_tokens": generation.get("completion_tokens"),
        "est_cost_usd": record.get("est_cost_usd"),
        "env_seconds": timings.get("environment"), "llmorpheus_seconds": timings.get("llmorpheus"),
        "coupling_seconds": timings.get("coupling"), "total_seconds": record.get("total_seconds"),
        "partial": record.get("partial"), "detail": record.get("detail"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "bugsinpy_llmorpheus.xlsx")
    args = parser.parse_args()

    selection = json.loads((RESULTS / "selection.json").read_text())
    records = {}
    for path in sorted((RESULTS / "bugs").glob("*.json")):
        record = json.loads(path.read_text())
        records[record["name"]] = record
    names = list(selection["bugs"]) + [n for n in records if n not in selection["bugs"]]
    rows = [bug_row(name, records.get(name)) for name in names]

    book = Workbook()
    book.remove(book.active)

    summary_columns = [
        ("Project", "project"), ("Bug", "bug_id"), ("Status", "status"), ("Stage", "stage"),
        ("Reason", "reason"), ("Python", "python_version"), ("Fixed commit", "fixed_commit"),
        ("Mutated files", "mutated_files"), ("Scope", "scope"), ("Focus line ranges", "focus_ranges"),
        ("Locations in whole files", "full_locations"), ("Test files", "test_files"),
        ("Test files dropped (not collectable)", "test_files_dropped"),
        ("Deselected tests", "deselected"), ("Baseline tests passed", "baseline_passed"),
        ("Baseline (s)", "baseline_seconds"), ("Jobs", "jobs"), ("Locations", "locations"),
        ("Prompts", "prompts"), ("Failed prompts", "failed_prompts"), ("Candidates", "candidates"),
        ("Mutants", "mutants"), ("Killed", "killed"), ("Survived", "survived"), ("Timeout", "timeout"),
        ("Errored", "errored"), ("Mutation score (%)", "score_pct"), ("Mutants on fix lines", "on_fix"),
        ("On-fix killed by test file", "on_fix_killed"), ("On-fix failing bug test", "behavioural"),
        ("On-fix textual match", "textual"), ("Bug reproduced (behavioural)", "reproduced_behavioural"),
        ("Bug reproduced (textual)", "reproduced_textual"), ("Prompt tokens", "prompt_tokens"),
        ("Completion tokens", "completion_tokens"), ("Est. cost (USD)", "est_cost_usd"),
        ("Env build (s)", "env_seconds"), ("LLMorpheus (s)", "llmorpheus_seconds"),
        ("Coupling (s)", "coupling_seconds"), ("Total (s)", "total_seconds"), ("Partial", "partial"),
    ]
    write_sheet(book, "Summary", summary_columns, rows, status_column="status")

    projects = defaultdict(lambda: defaultdict(float))
    for row in rows:
        p = projects[row["project"]]
        p["selected"] += 1
        p[row["status"]] += 1
        for key in ("mutants", "killed", "survived", "timeout", "on_fix", "behavioural", "textual",
                    "est_cost_usd"):
            p[key] += row.get(key) or 0
        p["reproduced"] += 1 if row.get("reproduced_behavioural") else 0
    project_rows = []
    for name, p in projects.items():
        detected, total = p["killed"] + p["timeout"], p["killed"] + p["timeout"] + p["survived"]
        project_rows.append({
            "project": name, "selected": int(p["selected"]), "done": int(p["done"] + p["partial"]),
            "skipped": int(p["skipped"]), "not_run": int(p["not run"]), "mutants": int(p["mutants"]),
            "killed": int(p["killed"]), "survived": int(p["survived"]), "timeout": int(p["timeout"]),
            "score_pct": round(100.0 * detected / total, 2) if total else None, "on_fix": int(p["on_fix"]),
            "behavioural": int(p["behavioural"]), "textual": int(p["textual"]),
            "reproduced": int(p["reproduced"]), "est_cost_usd": round(p["est_cost_usd"], 4),
        })
    write_sheet(book, "By project", [
        ("Project", "project"), ("Bugs selected", "selected"), ("Done", "done"), ("Skipped", "skipped"),
        ("Not run", "not_run"), ("Mutants", "mutants"), ("Killed", "killed"), ("Survived", "survived"),
        ("Timeout", "timeout"), ("Mutation score (%)", "score_pct"), ("Mutants on fix lines", "on_fix"),
        ("On-fix failing bug test", "behavioural"), ("On-fix textual match", "textual"),
        ("Bugs reproduced (behavioural)", "reproduced"), ("Est. cost (USD)", "est_cost_usd"),
    ], project_rows)

    mutant_rows, coupling_rows = [], []
    for name in names:
        record = records.get(name)
        for m in (record or {}).get("mutants", []):
            row = {
                "project": record["project"], "bug_id": record["bug_id"], "id": m.get("id"),
                "file": m.get("file"), "line": (m.get("start") or {}).get("line"),
                "column": (m.get("start") or {}).get("column"), "end_line": (m.get("end") or {}).get("line"),
                "kind": m.get("kind"), "original": m.get("original"), "replacement": m.get("replacement"),
                "status": m.get("status"), "duration": m.get("duration"), "on_fix": m.get("on_fix"),
                "bug_test_failed": m.get("bug_test_failed"), "textual_match": m.get("textual_match"),
                "explanation": m.get("explanation"), "mutated_region": m.get("mutated_region"),
                "buggy_lines": "\n".join(m.get("buggy_lines") or []),
            }
            mutant_rows.append(row)
            if m.get("on_fix"):
                coupling_rows.append(row)
    write_sheet(book, "Mutants", [
        ("Project", "project"), ("Bug", "bug_id"), ("Mutant ID", "id"), ("File", "file"), ("Line", "line"),
        ("Column", "column"), ("End line", "end_line"), ("Kind", "kind"), ("Original", "original"),
        ("Replacement", "replacement"), ("Status", "status"), ("Duration (s)", "duration"),
        ("On fix lines", "on_fix"), ("Bug test failed", "bug_test_failed"), ("Textual match", "textual_match"),
        ("Explanation", "explanation"),
    ], mutant_rows, status_column="status", wrap=("explanation",))
    write_sheet(book, "Coupling", [
        ("Project", "project"), ("Bug", "bug_id"), ("Mutant ID", "id"), ("File", "file"), ("Line", "line"),
        ("Kind", "kind"), ("Original", "original"), ("Replacement", "replacement"),
        ("Status (test file)", "status"), ("Bug test failed (behavioural)", "bug_test_failed"),
        ("Matches buggy code (textual)", "textual_match"), ("Mutated line(s)", "mutated_region"),
        ("Buggy line(s) from the patch", "buggy_lines"),
    ], coupling_rows, status_column="status", wrap=("mutated_region", "buggy_lines"))
    write_sheet(book, "Skipped", [
        ("Project", "project"), ("Bug", "bug_id"), ("Status", "status"), ("Stage", "stage"),
        ("Reason", "reason"), ("Detail (log tail)", "detail"),
    ], [r for r in rows if r["status"] in ("skipped", "not run")], status_column="status", wrap=("detail",))

    total_cost = sum(r.get("est_cost_usd") or 0 for r in rows)
    usage = [r.get("key_usage_after") for r in records.values() if r.get("key_usage_after") is not None]
    info = [
        ("Generated at", dt.datetime.now().isoformat(timespec="seconds")),
        ("Bugs selected", len(selection["bugs"])),
        ("Selection", "ceil({:.3f} × số bug) mỗi project, chọn ngẫu nhiên, seed {}".format(
            selection["fraction"], selection["seed"])),
        ("BugsInPy++ commit", selection.get("bugsinpy_commit")),
        ("LLMorpheus commit", selection.get("llmorpheus_commit")),
        ("Model", "deepseek/deepseek-v4.1-flash qua OpenRouter, route deepinfra/fp8/flex, reasoning tắt, temperature 0"),
        ("Price (USD / 1M tokens)", "input {:.3f}, output {:.3f}".format(
            selection["price_per_token_usd"]["input"] * 1e6, selection["price_per_token_usd"]["output"] * 1e6)),
        ("Est. cost, all bugs (USD)", round(total_cost, 4)),
        ("Key usage reported by OpenRouter (USD)", max(usage) if usage else None),
        ("Version tested", "Bản fixed của BugsInPy: commit buggy + các file đã sửa và file test lấy từ commit fixed"),
        ("Files mutated", "Chỉ các file .py không phải test mà bug_patch.txt sửa"),
        ("Scope", "File có hơn 800 vị trí: chỉ mutate hàm nhỏ nhất chứa mỗi dòng fix (không có thì lớp, "
                  "không có nữa thì ±40 dòng); file nhỏ hơn: mutate cả file. Xem cột Scope / Focus line ranges"),
        ("Test command", "pytest trên test_file của bug, -x; các test đỏ sẵn ở baseline bị --deselect (xem sheet Summary)"),
        ("Mutation score", "(killed + timeout) / (killed + timeout + survived)"),
        ("On fix lines", "Mutant chạm dòng mà bản vá thêm vào; với hunk chỉ xoá dòng, là hai dòng quanh chỗ xoá"),
        ("Bug test failed (behavioural)", "Mutant làm fail test bắt bug trong run_test.sh, như bản buggy"),
        ("Textual match", "Dòng sau mutate trùng (bỏ khoảng trắng) với một dòng bản buggy đã bị xoá bởi hunk đó"),
    ]
    sheet = book.create_sheet("Run info")
    for key, value in info:
        sheet.append([key, clean(value)])
    sheet.column_dimensions["A"].width, sheet.column_dimensions["B"].width = 38, 110
    for cell in sheet["A"]:
        cell.font = Font(bold=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    book.save(args.out)
    done = sum(1 for r in rows if r["status"] in ("done", "partial"))
    print("wrote {} - {} bug(s): {} done, {} skipped, {} not run; {} mutants".format(
        args.out, len(rows), done, sum(1 for r in rows if r["status"] == "skipped"),
        sum(1 for r in rows if r["status"] == "not run"), len(mutant_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
