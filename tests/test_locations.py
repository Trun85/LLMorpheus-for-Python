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
