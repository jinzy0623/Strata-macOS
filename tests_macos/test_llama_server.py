import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from serve.backends.llama_server import Client, LlamaServerBackend, Template

# What the fake native /tokenize returns for streamed text (special tokens parsed).
TOKENIZE = {"oi": [6517], " 👋": [59720, 233], "</think>": [248069], "\n\nOlá": [271, 81920]}


class NativeServer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.body = body
        if self.path in ("/apply-template", "/tokenize"):
            data = json.dumps({"prompt": "assistant\n<think>\n"} if self.path == "/apply-template" else
                              {"tokens": TOKENIZE[body["content"]]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.server.requested.set()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.server.delay:
            self.server.release.wait(3)
        try:
            for event in self.server.events:
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), NativeServer)
        self.server.delay = False
        self.server.requested, self.server.release = threading.Event(), threading.Event()
        self.server.events = [
            {"tokens": [0], "prompt_progress": {"processed": 2, "total": 4}},
            {"tokens": [7, 8]}, {"tokens": [9]},
            {"tokens": [], "stop": True, "stop_type": "eos", "tokens_cached": 2,
             "timings": {"prompt_ms": 25, "predicted_ms": 100, "cache_n": 1}},
        ]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.client = Client(f"http://127.0.0.1:{self.server.server_port}")
        self.engine = LlamaServerBackend(self.client, {
            "default_generation_settings": {"n_ctx": 1024}, "build_info": "test"}, 9)

    def tearDown(self):
        self.server.release.set()
        self.server.shutdown()
        self.server.server_close()

    def test_progress_is_not_output_and_eos_drains_timings(self):
        output = list(self.engine.generate([1, 2], 20, {"temperature": 0,
                           "repetition_penalty": 1.1, "seed": 42}, threading.Event()))
        self.assertEqual([t for t in output if t is not None], [7, 8, 9])
        self.assertEqual(self.engine.last["prompt_ms"], 25)
        self.assertEqual(self.engine.last["decode_ms"], 100)
        self.assertEqual(self.engine.last["generated"], 3)
        self.assertEqual(self.engine.last["reused"], 1)
        self.assertEqual(self.server.body["prompt"], [1, 2])
        self.assertEqual(self.server.body["temperature"], 0)
        self.assertEqual(self.server.body["repeat_penalty"], 1.1)

    def test_length_stop_does_not_invent_eos(self):
        self.server.events = [{"tokens": [7]}, {"stop": True, "stop_type": "limit"}]
        self.assertEqual([t for t in self.engine.generate([1], 1, {}, threading.Event())
                          if t is not None], [7])

    def test_streamed_text_wins_over_lossy_token_field(self):
        # llama-server holds back a chunk whose text ends mid-UTF-8 and then reports only the newest token,
        # so ids of held-back tokens (an emoji's first byte, or a whole speculative step) vanish from
        # "tokens". The streamed "content" is complete: the adapter must yield ids for exactly that text.
        # "tokens_predicted" counts every sampled token: a jump past len(tokens) marks the lossy chunks.
        self.server.events = [
            {"content": "oi", "tokens": [6517], "tokens_predicted": 1},
            {"content": " 👋", "tokens": [233], "tokens_predicted": 3},
            {"content": "</think>", "tokens": [248069], "tokens_predicted": 4},
            {"content": "\n\nOlá", "tokens": [81920], "tokens_predicted": 6},
            {"content": "", "tokens": [9], "tokens_predicted": 7},
            {"content": "", "tokens": [], "tokens_predicted": 7, "stop": True, "stop_type": "eos",
             "timings": {"prompt_ms": 5, "predicted_ms": 50, "cache_n": 0}},
        ]
        output = [t for t in self.engine.generate([1], 20, {}, threading.Event()) if t is not None]
        self.assertEqual(output, [6517, 59720, 233, 248069, 271, 81920, 9])
        self.assertEqual(self.engine.last["generated"], 7)

    def test_complete_chunks_keep_the_server_s_own_ids(self):
        # When the counter and "tokens" agree the ids are used as sampled (no /tokenize round trip: the fake
        # server has no entry for " tudo" and would answer 500), so the continuation prompt matches the cache.
        self.server.events = [
            {"content": "oi", "tokens": [6517], "tokens_predicted": 1},
            {"content": " tudo", "tokens": [53827], "tokens_predicted": 2},
            {"content": "", "tokens": [], "tokens_predicted": 2, "stop": True, "stop_type": "limit"},
        ]
        output = [t for t in self.engine.generate([1], 20, {}, threading.Event()) if t is not None]
        self.assertEqual(output, [6517, 53827])

    def test_a_server_without_the_counter_falls_back_to_the_text(self):
        self.server.events = [{"content": " 👋", "tokens": [233]}, {"stop": True, "stop_type": "limit"}]
        output = [t for t in self.engine.generate([1], 20, {}, threading.Event()) if t is not None]
        self.assertEqual(output, [59720, 233])

    def test_incomplete_stream_reports_failure(self):
        self.server.events = [{"tokens": [7]}]
        with self.assertRaisesRegex(RuntimeError, "ended before"):
            list(self.engine.generate([1], 10, {}, threading.Event()))

    def test_cancel_during_prefill_returns_promptly(self):
        self.server.delay = True
        cancel = threading.Event()
        gen = self.engine.generate([1], 10, {}, cancel)
        self.assertIsNone(next(gen))
        self.assertTrue(self.server.requested.wait(1))
        started = time.monotonic()
        cancel.set()
        self.assertEqual(list(gen), [])
        self.assertLess(time.monotonic() - started, 1.5)

    def test_template_passes_reasoning_and_normalizes_tool_arguments(self):
        template = Template(self.client)
        prompt = template.render([{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "weather", "arguments": {"city": "São Paulo"}}}]}],
            tools=[{"name": "weather", "parameters": {"type": "object"}}], reasoning_effort="low")
        self.assertTrue(template.starts_in_reasoning(prompt))
        self.assertEqual(self.server.body["chat_template_kwargs"], {"reasoning_effort": "low"})
        call = self.server.body["messages"][0]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "São Paulo"})
        self.assertEqual(self.server.body["tools"][0]["type"], "function")

    def test_finished_empty_assistant_turns_are_left_out_of_the_prompt(self):
        # Upstream #843: an earlier reply that came back empty must not be an example for the next one.
        Template(self.client).render([{"role": "user", "content": "a"}, {"role": "assistant", "content": ""},
                                      {"role": "user", "content": "b"}, {"role": "assistant", "content": ""}])
        roles = [m["role"] for m in self.server.body["messages"]]
        self.assertEqual(roles, ["user", "user", "assistant"])     # the final message is always kept

    def test_remote_endpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, "local http URL"):
            Client("http://example.com:8096")


if __name__ == "__main__":
    unittest.main()
