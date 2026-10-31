"""LLM back-ends, exercised against a local fake API server (no network, no key)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from llmorpheus.cli import _client, build_parser
from llmorpheus.llm import THINKING_MAX_TOKENS, DeepSeekClient, LLMError, build_client

ANSWER = "Option 1: The PLACEHOLDER can be replaced with:\n```python\na != b\n```\n"
OK_REPLY = {
    "choices": [{"message": {"role": "assistant", "content": ANSWER}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
}


@pytest.fixture()
def fake_api():
    captured = []
    replies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            captured.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": json.loads(self.rfile.read(length)),
                }
            )
            body = json.dumps(replies.pop(0) if replies else OK_REPLY).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(
        url="http://127.0.0.1:{}".format(server.server_address[1]),
        captured=captured,
        replies=replies,
    )
    server.shutdown()
    server.server_close()


def test_deepseek_defaults():
    client = build_client("deepseek")
    assert isinstance(client, DeepSeekClient)
    assert client.model == "deepseek-flash"
    assert client.base_url == "https://api.deepseek.com"
    assert client.api_key_env == "DEEPSEEK_API_KEY"
    assert client.thinking is False


def test_deepseek_needs_its_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(LLMError) as error:
        build_client("deepseek").preflight()
    assert "DEEPSEEK_API_KEY" in str(error.value)


def test_deepseek_request_turns_thinking_off(fake_api, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client = build_client("deepseek", base_url=fake_api.url, temperature=0.25)

    response = client.complete("system prompt", "user prompt")

    (request,) = fake_api.captured
    assert request["path"] == "/chat/completions"
    assert request["authorization"] == "Bearer sk-test"
    assert request["body"]["model"] == "deepseek-flash"
    assert request["body"]["thinking"] == {"type": "disabled"}
    assert request["body"]["temperature"] == 0.25
    assert request["body"]["max_tokens"] == 250
    assert "a != b" in response.text
    assert (response.prompt_tokens, response.completion_tokens) == (12, 7)


def test_deepseek_thinking_mode_drops_the_ignored_temperature(fake_api, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client = build_client("deepseek", base_url=fake_api.url, thinking=True, max_tokens=4096)

    client.complete("s", "u")

    body = fake_api.captured[0]["body"]
    assert body["thinking"] == {"type": "enabled"}
    assert "temperature" not in body


def test_reasoning_without_an_answer_is_a_clear_error(fake_api, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    fake_api.replies.append(
        {
            "choices": [
                {
                    "message": {"content": "", "reasoning_content": "Let me think about…"},
                    "finish_reason": "length",
                }
            ]
        }
    )
    client = build_client("deepseek", base_url=fake_api.url, thinking=True, attempts=1)

    with pytest.raises(LLMError) as error:
        client.complete("s", "u")
    assert "--max-tokens" in str(error.value)


def test_thinking_is_part_of_the_cache_key():
    off = build_client("deepseek")
    on = build_client("deepseek", thinking=True)
    assert off._cache_key("s", "u") != on._cache_key("s", "u")


def test_thinking_is_rejected_for_other_providers():
    with pytest.raises(ValueError):
        build_client("openai", thinking=True)


def test_cli_raises_the_token_budget_only_when_thinking(tmp_path):
    parser = build_parser()
    plain = _client(parser.parse_args(["generate", "--provider", "deepseek", "--no-cache"]), tmp_path)
    thinking = _client(
        parser.parse_args(["generate", "--provider", "deepseek", "--thinking", "--no-cache"]),
        tmp_path,
    )
    explicit = _client(
        parser.parse_args(
            ["generate", "--provider", "deepseek", "--thinking", "--max-tokens", "900", "--no-cache"]
        ),
        tmp_path,
    )
    assert plain.max_tokens == 250 and plain.thinking is False
    assert thinking.max_tokens == THINKING_MAX_TOKENS and thinking.thinking is True
    assert explicit.max_tokens == 900


def test_openrouter_turns_reasoning_off_and_pins_the_route(fake_api, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client = build_client("openrouter", base_url=fake_api.url, route=["deepinfra/fp8/flex"])

    response = client.complete("s", "u")

    (request,) = fake_api.captured
    assert request["authorization"] == "Bearer sk-or-test"
    assert request["body"]["model"] == "deepseek/deepseek-v4.1-flash"
    assert request["body"]["reasoning"] == {"enabled": False}
    assert request["body"]["provider"] == {"order": ["deepinfra/fp8/flex"], "allow_fallbacks": False}
    assert "a != b" in response.text


def test_openrouter_without_a_route_lets_openrouter_choose(fake_api, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    build_client("openrouter", base_url=fake_api.url).complete("s", "u")
    assert "provider" not in fake_api.captured[0]["body"]


def test_openrouter_reasoning_without_an_answer_is_a_clear_error(fake_api, monkeypatch):
    """OpenRouter names the field `reasoning`; an empty answer must not pass as zero mutants."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    fake_api.replies.append(
        {"choices": [{"message": {"content": "", "reasoning": "Let me think…"}, "finish_reason": "length"}]}
    )
    client = build_client("openrouter", base_url=fake_api.url, attempts=1)

    with pytest.raises(LLMError) as error:
        client.complete("s", "u")
    assert "reasoning" in str(error.value)


def test_route_is_part_of_the_cache_key():
    a = build_client("openrouter", route=["deepinfra/fp8/flex"])
    b = build_client("openrouter", route=["openinference/fp4"])
    assert a._cache_key("s", "u") != b._cache_key("s", "u")


def test_route_is_rejected_for_other_providers():
    with pytest.raises(ValueError):
        build_client("openai", route=["deepinfra/fp8"])


def test_cli_route_accepts_repeated_and_comma_separated_endpoints(tmp_path):
    args = build_parser().parse_args(
        ["generate", "--provider", "openrouter", "--route", "a/fp8", "--route", "b,c", "--no-cache"]
    )
    assert _client(args, tmp_path).route == ["a/fp8", "b", "c"]
