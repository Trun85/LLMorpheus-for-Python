"""Rendering results: a terminal summary and a self-contained HTML report."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..console import format_duration
from .runner import ERROR, KILLED, SURVIVED, TIMEOUT, MutantResult, RunSummary

def text_summary(summary: RunSummary) -> str:
    """A compact table, in the spirit of Table 2 of the paper."""
    counts = summary.counts()
    per_file: Dict[str, Dict[str, int]] = {}
    for result in summary.results:
        bucket = per_file.setdefault(
            result.mutant.file, {KILLED: 0, SURVIVED: 0, TIMEOUT: 0, ERROR: 0}
        )
        bucket[result.status] += 1

    width = max([len("file")] + [len(name) for name in per_file]) if per_file else 4
    lines = [
        "{:<{w}}  {:>7} {:>8} {:>8} {:>7} {:>7}".format(
            "file", "mutants", "killed", "survived", "timeout", "score", w=width
        ),
        "-" * (width + 42),
    ]
    for name in sorted(per_file):
        bucket = per_file[name]
        total = sum(bucket.values())
        detected = bucket[KILLED] + bucket[TIMEOUT]
        denominator = detected + bucket[SURVIVED]
        score = (detected / denominator * 100.0) if denominator else 0.0
        lines.append(
            "{:<{w}}  {:>7} {:>8} {:>8} {:>7} {:>6.1f}%".format(
                name, total, bucket[KILLED], bucket[SURVIVED], bucket[TIMEOUT], score, w=width
            )
        )
    lines.append("-" * (width + 42))
    lines.append(
        "{:<{w}}  {:>7} {:>8} {:>8} {:>7} {:>6.1f}%".format(
            "total",
            len(summary.results),
            counts[KILLED],
            counts[SURVIVED],
            counts[TIMEOUT],
            summary.mutation_score,
            w=width,
        )
    )
    if counts[ERROR]:
        lines.append("{} mutant(s) could not be evaluated".format(counts[ERROR]))
    lines.append(
        "total run time {} (baseline {})".format(
            format_duration(summary.total_duration), format_duration(summary.baseline_duration)
        )
    )
    return "\n".join(lines)


def surviving_mutants(summary: RunSummary, limit: Optional[int] = None) -> List[MutantResult]:
    survivors = [result for result in summary.results if result.status == SURVIVED]
    survivors.sort(key=lambda item: (item.mutant.file, item.mutant.start))
    return survivors if limit is None else survivors[:limit]


def _mutated_line(source_lines: Sequence[str], result: MutantResult) -> str:
    mutant = result.mutant
    index = mutant.start.line - 1
    if index < 0 or index >= len(source_lines):
        return mutant.replacement
    line = source_lines[index]
    if mutant.start.line != mutant.end.line:
        return line[: mutant.start.column] + mutant.replacement + " …"
    return line[: mutant.start.column] + mutant.replacement + line[mutant.end.column :]


def _payload(summary: RunSummary, project_root: Optional[Path]) -> dict:
    cache: Dict[str, List[str]] = {}

    def lines_for(relative: str) -> List[str]:
        if relative not in cache:
            text = ""
            if project_root is not None:
                path = project_root / relative
                if path.is_file():
                    try:
                        text = path.read_text(encoding="utf-8")
                    except OSError:
                        text = ""
            cache[relative] = text.splitlines()
        return cache[relative]

    items = []
    for result in summary.results:
        mutant = result.mutant
        source_lines = lines_for(mutant.file)
        index = mutant.start.line - 1
        original_line = source_lines[index] if 0 <= index < len(source_lines) else mutant.original
        items.append(
            {
                "id": mutant.id,
                "file": mutant.file,
                "kind": mutant.kind,
                "line": mutant.start.line,
                "column": mutant.start.column,
                "status": result.status,
                "duration": round(result.duration, 2),
                "original": mutant.original,
                "replacement": mutant.replacement,
                "explanation": mutant.explanation,
                "originalLine": original_line.rstrip("\n"),
                "mutatedLine": _mutated_line(source_lines, result).rstrip("\n"),
            }
        )
    counts = summary.counts()
    return {
        "generatedAt": str(summary.meta.get("run_at", "")),
        "score": round(summary.mutation_score, 2),
        "counts": counts,
        "totalDuration": summary.total_duration,
        "baselineDuration": summary.baseline_duration,
        "timeout": summary.timeout,
        "meta": summary.meta,
        "mutants": items,
    }


def write_html_report(
    summary: RunSummary, path: Path, project_root: Optional[Path] = None, title: str = "LLMorpheus"
) -> Path:
    """Write a single self-contained HTML file, with no external resources."""
    data = _payload(summary, project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = _TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__DATA__", json.dumps(data).replace("</", "<\\/")
    )
    path.write_text(document, encoding="utf-8")
    return path


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ mutation report</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #fbfbfd; --panel: #ffffff; --ink: #16181d; --muted: #666c7a;
    --line: #e3e5ea; --killed: #2f7d4f; --survived: #c2371f; --timeout: #a2701a;
    --error: #6b7280; --accent: #3b5bdb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --panel: #1c1f25; --ink: #e8eaed; --muted: #9aa2b1;
      --line: #2c313a; --killed: #58c98a; --survived: #ff8a72; --timeout: #e2b04a;
      --error: #9aa2b1; --accent: #8ea2ff;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { padding: 28px 32px 20px; border-bottom: 1px solid var(--line); }
  h1 { margin: 0 0 4px; font-size: 20px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); font-size: 13px; }
  .cards { display: flex; flex-wrap: wrap; gap: 12px; padding: 20px 32px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
          padding: 14px 18px; min-width: 128px; }
  .card .n { font-size: 24px; font-weight: 620; letter-spacing: -0.02em; }
  .card .l { color: var(--muted); font-size: 12px; text-transform: uppercase;
             letter-spacing: 0.06em; }
  .bar { height: 10px; border-radius: 999px; overflow: hidden; display: flex;
         margin: 0 32px 20px; border: 1px solid var(--line); background: var(--panel); }
  .bar span { display: block; height: 100%; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
              padding: 0 32px 16px; }
  select, input { background: var(--panel); color: var(--ink); border: 1px solid var(--line);
                  border-radius: 8px; padding: 7px 10px; font: inherit; }
  input { min-width: 220px; }
  main { padding: 0 32px 48px; }
  .file { margin-bottom: 22px; border: 1px solid var(--line); border-radius: 10px;
          background: var(--panel); overflow: hidden; }
  .file > h2 { margin: 0; padding: 12px 16px; font-size: 13px; font-weight: 600;
               border-bottom: 1px solid var(--line); display: flex; gap: 10px;
               align-items: baseline; }
  .file > h2 .score { margin-left: auto; color: var(--muted); font-weight: 500; }
  .m { padding: 12px 16px; border-bottom: 1px solid var(--line); }
  .m:last-child { border-bottom: 0; }
  .m .head { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
             margin-bottom: 8px; }
  .tag { font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase;
         border-radius: 999px; padding: 2px 9px; border: 1px solid currentColor; }
  .killed { color: var(--killed); } .survived { color: var(--survived); }
  .timeout { color: var(--timeout); } .error { color: var(--error); }
  .loc { color: var(--muted); font-size: 12px; }
  .kind { color: var(--muted); font-size: 12px; }
  pre { margin: 0; padding: 8px 10px; border-radius: 8px; overflow-x: auto;
        font: 12.5px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
        background: color-mix(in srgb, var(--ink) 5%, transparent); }
  pre.minus { border-left: 3px solid var(--muted); }
  pre.plus { border-left: 3px solid var(--survived); margin-top: 4px; }
  .why { color: var(--muted); font-size: 12.5px; margin-top: 6px; }
  .empty { color: var(--muted); padding: 30px 0; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__ mutation report</h1>
  <div class="sub" id="subtitle"></div>
</header>
<div class="cards" id="cards"></div>
<div class="bar" id="bar"></div>
<div class="controls">
  <select id="status">
    <option value="all">All statuses</option>
    <option value="survived">Survived</option>
    <option value="killed">Killed</option>
    <option value="timeout">Timed out</option>
    <option value="error">Errored</option>
  </select>
  <select id="kind"><option value="all">All placeholder kinds</option></select>
  <input id="search" type="search" placeholder="Filter by file or code…">
</div>
<main id="out"></main>
<script>
const DATA = __DATA__;
const STATUSES = ["killed", "survived", "timeout", "error"];
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function renderHeader() {
  const c = DATA.counts;
  const total = DATA.mutants.length;
  document.getElementById("subtitle").textContent =
    `${total} mutants · mutation score ${DATA.score}% · ` +
    `run took ${Math.round(DATA.totalDuration)}s · generated ${DATA.generatedAt || "—"}`;
  const cards = [["Mutation score", DATA.score + "%"], ["Mutants", total],
    ["Killed", c.killed], ["Survived", c.survived], ["Timed out", c.timeout]];
  if (c.error) cards.push(["Errored", c.error]);
  document.getElementById("cards").innerHTML = cards.map(([l, n]) =>
    `<div class="card"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
  const colors = {killed: "var(--killed)", survived: "var(--survived)",
                  timeout: "var(--timeout)", error: "var(--error)"};
  document.getElementById("bar").innerHTML = STATUSES.filter((s) => c[s])
    .map((s) => `<span style="width:${(c[s] / total) * 100}%;background:${colors[s]}" title="${s}: ${c[s]}"></span>`)
    .join("");
  const kinds = [...new Set(DATA.mutants.map((m) => m.kind))].sort();
  document.getElementById("kind").innerHTML =
    '<option value="all">All placeholder kinds</option>' +
    kinds.map((k) => `<option value="${k}">${k}</option>`).join("");
}

const fileTotals = new Map();
const fileScores = new Map();
for (const m of DATA.mutants) {
  fileTotals.set(m.file, (fileTotals.get(m.file) || 0) + 1);
  const s = fileScores.get(m.file) || {detected: 0, denom: 0};
  if (m.status === "killed" || m.status === "timeout") { s.detected++; s.denom++; }
  else if (m.status === "survived") { s.denom++; }
  fileScores.set(m.file, s);
}
function fileScore(file) {
  const s = fileScores.get(file);
  return s && s.denom ? ((s.detected / s.denom) * 100).toFixed(1) + "%" : "—";
}

function render() {
  const status = document.getElementById("status").value;
  const kind = document.getElementById("kind").value;
  const needle = document.getElementById("search").value.toLowerCase();
  const shown = DATA.mutants.filter((m) =>
    (status === "all" || m.status === status) &&
    (kind === "all" || m.kind === kind) &&
    (!needle || (m.file + m.originalLine + m.mutatedLine + m.replacement).toLowerCase().includes(needle)));
  const byFile = new Map();
  for (const m of shown) {
    if (!byFile.has(m.file)) byFile.set(m.file, []);
    byFile.get(m.file).push(m);
  }
  const out = document.getElementById("out");
  if (!shown.length) { out.innerHTML = '<p class="empty">No mutants match these filters.</p>'; return; }
  out.innerHTML = [...byFile.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([file, items]) => {
    // The per-file score always reflects the whole file, not the current filter.
    const score = fileScore(file);
    const count = items.length === fileTotals.get(file)
      ? `${items.length} mutants`
      : `${items.length} of ${fileTotals.get(file)} mutants`;
    return `<section class="file"><h2>${esc(file)}<span class="kind">${count}</span>
      <span class="score">score ${score}</span></h2>` +
      items.sort((a, b) => a.line - b.line).map((m) => `<div class="m">
        <div class="head"><span class="tag ${m.status}">${m.status}</span>
          <span class="loc">line ${m.line}</span><span class="kind">${esc(m.kind)}</span></div>
        <pre class="minus">${esc(m.originalLine || m.original)}</pre>
        <pre class="plus">${esc(m.mutatedLine || m.replacement)}</pre>
        ${m.explanation ? `<div class="why">${esc(m.explanation)}</div>` : ""}
      </div>`).join("") + "</section>";
  }).join("");
}

renderHeader();
for (const id of ["status", "kind", "search"]) {
  document.getElementById(id).addEventListener("input", render);
}
render();
</script>
</body>
</html>
"""
