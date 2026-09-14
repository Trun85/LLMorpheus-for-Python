"""A rejected API key fails every prompt the same way, so generation must stop at once."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from llmorpheus.generation.generator import GeneratorConfig, generate
from llmorpheus.llm import LLMError, build_client

# Dozens of placeholder locations, so one request per prompt would be obvious.
MODULE = "def f(a, b):\n" + "".join(
    "    if a == {0}:\n        return g(b, {0})\n".format(index) for index in range(12)
)


@pytest.fixture()
def rejecting_api():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(self.path)
            body = json.dumps(
                {
                    "error": {
                        "message": "Authentication Fails, Your api key: ****0000 is invalid",
                        "type": "authentication_error",
                    }
                }
            ).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:{}".format(server.server_address[1]), calls
    server.shutdown()
    server.server_close()


def test_rejected_api_key_stops_generation_immediately(tmp_path: Path, rejecting_api, monkeypatch):
    url, calls = rejecting_api
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-invalid-key-0000")
    (tmp_path / "mod.py").write_text(MODULE, encoding="utf-8")
    config = GeneratorConfig(
        project_root=tmp_path, sources=["mod.py"], concurrency=1, save_prompts=False
    )
    client = build_client("deepseek", base_url=url, attempts=1)

    with pytest.raises(LLMError) as error:
        generate(config, client)

    message = str(error.value)
    assert "rejected the API key (HTTP 401)" in message
    assert "DEEPSEEK_API_KEY" in message
    # One refusal is enough; the queued prompts must never be sent.
    assert len(calls) <= 2
