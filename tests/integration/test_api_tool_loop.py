"""
Integration test: the API-key tool loop through the real anthropic SDK.

A local HTTP server stands in for the Messages streaming endpoint, so this
runs offline and needs no API key. It exercises what mocks cannot: that the
SDK accepts the tool definitions, streams a tool_use, lets the loop pass the
assistant content back unchanged, and serialises the tool_result.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _start(n_in):
    return {"type": "message_start", "message": {
        "id": f"msg_{n_in}", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": n_in, "output_tokens": 1,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}}


_TOOL_TURN = [
    _start(100),
    {"type": "content_block_start", "index": 0,
     "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "input_json_delta", "partial_json": json.dumps({"path": "pkg/retry.py"})}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
     "usage": {"output_tokens": 20}},
    {"type": "message_stop"},
]
_ANSWER_TURN = [
    _start(300),
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "It is retry_with_backoff in pkg/retry.py."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
     "usage": {"output_tokens": 15}},
    {"type": "message_stop"},
]


class _StubMessagesAPI(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        data = _sse(_TOOL_TURN if len(type(self).requests) == 1 else _ANSWER_TURN)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestApiToolLoopThroughSdk(unittest.TestCase):

    def setUp(self):
        self.project = tempfile.mkdtemp(prefix="cl_project_")
        Path(self.project, "pkg").mkdir()
        Path(self.project, "pkg", "retry.py").write_text(
            "def retry_with_backoff(fn):\n    return fn\n", encoding="utf-8")
        self.cwd = os.getcwd()
        os.chdir(self.project)
        _StubMessagesAPI.requests = []
        self.server = HTTPServer(("127.0.0.1", 0), _StubMessagesAPI)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        os.chdir(self.cwd)
        shutil.rmtree(self.project, ignore_errors=True)

    def test_read_file_round_trip(self):
        print("\n  ▶ TestApiToolLoopThroughSdk.test_read_file_round_trip")
        import io
        import anthropic
        import claude_light.config as config
        from claude_light import llm

        stub = anthropic.Anthropic(api_key="sk-ant-stub",
                                   base_url=f"http://127.0.0.1:{self.server.server_port}")
        kwargs = dict(model="claude-sonnet-5-5", max_tokens=2000,
                      system=llm._build_system_blocks("SKELETON"),
                      messages=[{"role": "user", "content": [{"type": "text", "text": "Where is retry?"}]}])
        llm._apply_effort(kwargs, "medium")
        with patch.object(config, "AUTH_MODE", "API_KEY"), \
             patch.object(llm, "client", stub), \
             patch("sys.stdout", io.StringIO()):
            reply, usage, _ = llm._make_streaming_api_call(**kwargs)

        reqs = _StubMessagesAPI.requests
        self.assertEqual(len(reqs), 2)
        self.assertEqual({t["name"] for t in reqs[0]["tools"]},
                         {"read_file", "fetch_chunk", "grep", "list_files", "shell"})
        assistant, results = reqs[1]["messages"][-2], reqs[1]["messages"][-1]
        self.assertEqual(assistant["content"][0]["type"], "tool_use")
        self.assertEqual(assistant["content"][0]["id"], "toolu_1")
        result = results["content"][0]
        self.assertEqual(result["tool_use_id"], "toolu_1")
        self.assertIn("1\tdef retry_with_backoff(fn):", result["content"])
        self.assertEqual(reply, "It is retry_with_backoff in pkg/retry.py.")
        self.assertEqual((usage.input_tokens, usage.output_tokens), (400, 35))


if __name__ == "__main__":
    unittest.main()
