from llmorpheus.analysis.locations import (
    CALL_ARGUMENT,
    CALL_ARGUMENTS,
    CALL_CALLEE,
    COMPREHENSION_CONDITION,
    EXTENDED_KINDS,
    FOR_ITER,
    FOR_TARGET,
    IF_CONDITION,
    RETURN_VALUE,
    WHILE_CONDITION,
    find_locations,
    resolve_kinds,
)
from llmorpheus.analysis.positions import SourceMap


def kinds_in(source, **kwargs):
    return {location.kind for location in find_locations(source, "m.py", **kwargs)}


def originals_of(source, kind, **kwargs):
    return [
        location.original
        for location in find_locations(source, "m.py", **kwargs)
        if location.kind == kind
    ]


def test_finds_the_paper_placeholder_kinds():
    source = (
        "def f(xs):\n"
        "    while len(xs) > 0:\n"
        "        if xs[0] == 1:\n"
        "            xs.pop(0, None)\n"
        "    for x in xs:\n"
        "        print(x)\n"
        "    return [y for y in xs if y > 2]\n"
    )
    found = kinds_in(source)
    assert {IF_CONDITION, WHILE_CONDITION, FOR_TARGET, FOR_ITER, CALL_CALLEE, CALL_ARGUMENT,
            CALL_ARGUMENTS, COMPREHENSION_CONDITION} <= found
    assert RETURN_VALUE not in found  # not part of the default scheme


def test_extended_kinds_add_return_values():
    source = "def f():\n    return 1 + 2\n"
    assert RETURN_VALUE in kinds_in(source, kinds=["all"])


def test_spans_are_exact():
    source = "if a == b:\n    pass\n"
    (location,) = [
        item for item in find_locations(source, "m.py") if item.kind == IF_CONDITION
    ]
    assert location.original == "a == b"
    assert SourceMap(source).text(location.start, location.end) == "a == b"


def test_argument_list_span_covers_every_argument():
    source = "f(a, b=1, **rest)\n"
    assert originals_of(source, CALL_ARGUMENTS) == ["a, b=1, **rest"]


def test_single_positional_argument_is_not_duplicated_as_argument_list():
    assert originals_of("f(a)\n", CALL_ARGUMENTS) == []
    assert originals_of("f(a)\n", CALL_ARGUMENT) == ["a"]


def test_non_ascii_source_positions_are_character_accurate():
    source = 'if naïve == "café":\n    pass\n'
    (location,) = [
        item for item in find_locations(source, "m.py") if item.kind == IF_CONDITION
    ]
    assert location.original == 'naïve == "café"'


def test_fstring_internals_are_not_mutated():
    source = 'x = f"{compute(1)} and {other(2)}"\n'
    assert find_locations(source, "m.py") == []


def test_unparsable_file_yields_nothing():
    assert find_locations("def (:\n", "m.py") == []


def test_oversized_fragments_are_skipped():
    source = "if {}:\n    pass\n".format(" or ".join("value{}".format(i) for i in range(80)))
    assert IF_CONDITION not in kinds_in(source)


def test_resolve_kinds_rejects_unknown_names():
    assert resolve_kinds(["all"]) == set(EXTENDED_KINDS)
    assert resolve_kinds([IF_CONDITION]) == {IF_CONDITION}
    try:
        resolve_kinds(["nonsense"])
    except ValueError as error:
        assert "nonsense" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_line_ranges_keep_only_locations_starting_inside_them(tmp_path):
    from llmorpheus.generation.generator import GeneratorConfig, collect_locations

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = f(1)\ny = g(2)\nz = h(3)\n")
    (tmp_path / "pkg" / "b.py").write_text("w = k(4)\n")

    def lines_with_locations(lines):
        config = GeneratorConfig(project_root=tmp_path, sources=["pkg"], lines=lines)
        _, per_file = collect_locations(config)
        return {path: sorted({loc.start.line for loc in locs}) for path, locs in per_file.items()}

    assert lines_with_locations([]) == {"pkg/a.py": [1, 2, 3], "pkg/b.py": [1]}
    # A file with ranges is narrowed; one without keeps everything.
    assert lines_with_locations(["pkg/a.py:2-3"]) == {"pkg/a.py": [2, 3], "pkg/b.py": [1]}
    assert lines_with_locations(["pkg/a.py:1", "pkg/a.py:3-3"]) == {"pkg/a.py": [1, 3], "pkg/b.py": [1]}


def test_line_ranges_reject_malformed_specs():
    import pytest

    from llmorpheus.generation.generator import parse_line_ranges

    assert parse_line_ranges(["a/b.py:5-9"]) == {"a/b.py": [(5, 9)]}
    for bad in ("a.py", "a.py:x-3", "a.py:9-5", ":1-2"):
        with pytest.raises(ValueError):
            parse_line_ranges([bad])


def test_form_feed_and_unicode_separators_do_not_shift_lines():
    """str.splitlines() breaks at these, Python's tokenizer does not."""
    body = "result = compute(total, rate)\nif result > limit:\n    notify(user)\n"
    plain = find_locations("x = 1  # ab\n" + body, "m.py")
    for odd in ("\x0c", "\u2028", "\x1c", "\x85"):
        located = find_locations("x = 1  # a{}b\n".format(odd) + body, "m.py")
        assert [(item.kind, item.start, item.original) for item in located] == [
            (item.kind, item.start, item.original) for item in plain
        ], repr(odd)


def test_line_ranges_must_name_a_selected_file(tmp_path):
    import pytest
    from llmorpheus.generation.generator import GeneratorConfig, collect_locations

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("if x:\n    f(1)\n", encoding="utf-8")

    def run(lines):
        config = GeneratorConfig(project_root=tmp_path, sources=["pkg"], lines=lines)
        return collect_locations(config)[1]

    with pytest.raises(ValueError, match="pkg/typo.py"):
        run(["pkg/typo.py:1-2"])
    # Equivalent spellings of the same file are accepted.
    assert list(run(["./pkg//a.py:1"])) == ["pkg/a.py"]
    assert list(run(["{}:1".format(tmp_path / "pkg" / "a.py")])) == ["pkg/a.py"]
