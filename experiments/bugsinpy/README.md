# LLMorpheus on BugsInPy++

Runs LLMorpheus over bugs from [BugsInPy++](https://github.com/UNIST-LOFT/BugsInPyPP)
and exports the results to one Excel workbook.

For each bug, on its **fixed** version (BugsInPy's buggy commit plus the fixed
commit's changed files and test files):

1. check out with `bugsinpy-checkout`, create the conda env the way
   `bugsinpy-testall` does, and run `bugsinpy-compile`;
2. confirm the bug-revealing tests in `run_test.sh` pass;
3. run the bug's `test_file` with pytest, `--deselect` the tests that already
   fail, and confirm the rest is green;
4. mutate only the non-test `.py` files that `bug_patch.txt` touches;
5. for every mutant on a patched line, re-run the bug-revealing tests
   (behavioural coupling) and compare the mutated line with the buggy code
   (textual coupling).

## Run

From the repository root, with Docker running:

```bash
docker build --platform linux/amd64 -t llmorpheus-bugsinpy experiments/bugsinpy
```

```bash
docker run -d --name llmorpheus-bugsinpy --platform linux/amd64 -v "$PWD:/repo" -v bugsinpy-work:/work llmorpheus-bugsinpy sleep infinity
```

```bash
docker exec -d llmorpheus-bugsinpy bash -c 'cd /repo && exec python experiments/bugsinpy/run_bugs.py >> experiments/bugsinpy/results/run.log 2>&1'
```

```bash
docker exec llmorpheus-bugsinpy python /repo/experiments/bugsinpy/export_excel.py
```

The LLM settings come from `.env`. The run stops before a bug once the
OpenRouter key has less than `--budget-floor` dollars left, and a bug cut short
by the key's limit is retried on the next run. Finished bugs are never re-run:
delete `results/bugs/<project>_<id>.json` to redo one.

Useful options: `--fraction` (default 1/3 of each project's bugs, seeded),
`--projects`, `--bugs`, `--max-prompts` (per-bug prompt cap), `--max-hours`
(per-bug wall-clock cap), `--jobs`, `--keep` (keep envs and checkouts).

## Output

`results/` (gitignored): `run.log`, `selection.json`, one `bugs/<bug>.json`
per bug, `raw/<bug>/` with llmorpheus's own files, `logs/<bug>.log` with every
command's output, and `bugsinpy_llmorpheus.xlsx`.
