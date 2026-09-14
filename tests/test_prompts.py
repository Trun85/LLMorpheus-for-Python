from llmorpheus.analysis.locations import IF_CONDITION, find_locations
from llmorpheus.analysis.positions import Position, SourceMap
from llmorpheus.prompting.prompts import PLACEHOLDER, build_code_with_placeholder, build_prompts, render

SOURCE = "def f(a, b):\n    if a == b:\n        return 1\n    return 0\n"


def test_render_inserts_values_verbatim():
    assert render("x {{a}} y", {"a": "{{b}}"}) == "x {{b}} y"


def test_placeholder_replaces_only_the_location():
    location = [item for item in find_locations(SOURCE, "m.py") if item.kind == IF_CONDITION][0]
    code, first, last = build_code_with_placeholder(SourceMap(SOURCE), location, 200)
    assert "if {}:".format(PLACEHOLDER) in code
    assert "a == b" not in code
    assert (first, last) == (1, 4)


def test_context_window_is_capped():
    source = "\n".join("x = {}".format(i) for i in range(500)) + "\nif x == 1:\n    pass\n"
    location = [
        item for item in find_locations(source, "m.py") if item.kind == IF_CONDITION
    ][0]
    code, first, last = build_code_with_placeholder(SourceMap(source), location, 20)
    assert last - first + 1 == 20
    assert PLACEHOLDER in code


def test_prompt_contains_the_original_fragment():
    locations = [item for item in find_locations(SOURCE, "m.py") if item.kind == IF_CONDITION]
    (prompt,) = build_prompts(SOURCE, locations)
    assert "a == b" in prompt.text
    assert PLACEHOLDER in prompt.text
    assert "mutation testing" in prompt.system.lower()


def test_window_is_centred_on_the_location():
    source = "\n".join("line{}".format(i) for i in range(100))
    smap = SourceMap(source)
    first, last = smap.window(Position(50, 0), Position(50, 3), 10)
    assert last - first + 1 == 10
    assert first <= 50 <= last
