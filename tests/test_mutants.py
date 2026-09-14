from llmorpheus.analysis.locations import IF_CONDITION, find_locations
from llmorpheus.generation.mutants import (
    CandidateFilter,
    GenerationStats,
    Mutant,
    MutantSet,
    extract_code_blocks,
    extract_explanations,
    is_syntactically_valid,
    mutants_from_completion,
)

SOURCE = "def f(a, b):\n    if a == b:\n        return 1\n    return 0\n"

COMPLETION = """Option 1: The PLACEHOLDER can be replaced with:
```python
a != b
```
This would result in different behavior because the branch is inverted.

Option 2: The PLACEHOLDER can be replaced with:
```
a is b
```
This would result in different behavior because identity differs from equality.

Option 3: The PLACEHOLDER can be replaced with:
```python
a ==
```
This would result in different behavior because it is broken.

DONE."""


def condition_location():
    return [item for item in find_locations(SOURCE, "m.py") if item.kind == IF_CONDITION][0]


def test_extract_code_blocks_handles_tagged_and_untagged_fences():
    assert extract_code_blocks(COMPLETION) == ["a != b", "a is b", "a =="]


def test_extract_inline_fence():
    assert extract_code_blocks("try ```a < b``` instead") == ["a < b"]


def test_extract_explanations_pairs_with_blocks():
    assert len(extract_explanations(COMPLETION)) == 3


def test_syntactic_validity_is_checked_in_context():
    location = condition_location()
    assert is_syntactically_valid(SOURCE, location, "a != b")
    assert not is_syntactically_valid(SOURCE, location, "a ==")
    # Valid as a fragment on its own, but not in this position.
    assert not is_syntactically_valid(SOURCE, location, "a = 1")


def test_completion_is_filtered_into_mutants():
    location = condition_location()
    stats = GenerationStats()
    filter_ = CandidateFilter(source=SOURCE, file="m.py")
    mutants = mutants_from_completion(COMPLETION, location, filter_, stats)
    assert [mutant.replacement for mutant in mutants] == ["a != b", "a is b"]
    assert stats.candidates == 3
    assert stats.invalid == 1
    assert stats.mutants == 2
    assert mutants[0].explanation.startswith("the branch is inverted")


def test_identical_and_duplicate_candidates_are_discarded():
    location = condition_location()
    stats = GenerationStats()
    filter_ = CandidateFilter(source=SOURCE, file="m.py")
    completion = "```python\na == b\n```\n```python\na != b\n```\n```python\na != b\n```"
    mutants = mutants_from_completion(completion, location, filter_, stats)
    assert len(mutants) == 1
    assert stats.identical == 1
    assert stats.duplicate == 1


def test_multiline_replacements_are_rejected_by_default():
    location = condition_location()
    filter_ = CandidateFilter(source=SOURCE, file="m.py")
    assert filter_.check(location, "a != b\nprint(a)") == (False, "invalid")


def test_applying_a_mutant_splices_the_exact_span():
    location = condition_location()
    mutant = Mutant(
        id="x",
        file="m.py",
        kind=location.kind,
        start=location.start,
        end=location.end,
        original=location.original,
        replacement="a != b",
    )
    assert mutant.apply(SOURCE) == SOURCE.replace("a == b", "a != b")


def test_mutant_set_round_trips(tmp_path):
    location = condition_location()
    mutant = Mutant(
        id="x",
        file="m.py",
        kind=location.kind,
        start=location.start,
        end=location.end,
        original=location.original,
        replacement="a != b",
        explanation="why",
    )
    original = MutantSet(mutants=[mutant], stats=GenerationStats(mutants=1), meta={"k": "v"})
    path = tmp_path / "mutants.json"
    original.write(path)
    restored = MutantSet.read(path)
    assert restored.mutants[0].as_dict() == mutant.as_dict()
    assert restored.stats.mutants == 1
    assert restored.meta == {"k": "v"}
