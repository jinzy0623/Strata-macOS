"""Use a current local llama-server with Strata's UI and token-id protocol.

The pinned Python binding predates qwen4exp and GGML type 42. This adapter
uses the installed native Metal engine, including its lazy embedding reads.
"""
from __future__ import annotations

import http.client
import json
import queue
import socket
import threading
import time
from urllib.parse import urlsplit

from serve.backends import BackendBundle
from serve.frontend import drop_empty_assistant_turns, images_of


class Client:
    def __init__(self, url):
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("llama_server_url must be a local http URL")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("llama_server_url must contain only a local host and port")
        self.host, self.port = parsed.hostname, parsed.port or 8080

    def connection(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=300)

    def request(self, path, body=None):
        conn = self.connection()
        try:
            conn.request("GET" if body is None else "POST", path,
                         None if body is None else json.dumps(body).encode(),
                         {"Content-Type": "application/json"})
            with conn.getresponse() as response:
                data = response.read()
                if response.status != 200:
                    raise ValueError(f"llama-server {path}: HTTP {response.status}: {data.decode('utf-8', 'replace')}")
                return json.loads(data)
        finally:
            conn.close()


class Tokenizer:
    def __init__(self, client):
        self.client = client

    def encode(self, text, parse_special=False):
        return self.client.request("/tokenize", {"content": text, "add_special": False,
                                                 "parse_special": parse_special})["tokens"]

    def decode(self, ids, errors="replace"):
        return self.client.request("/detokenize", {"tokens": list(ids)})["content"]


class Template:
    def __init__(self, client):
        self.client = client

    @staticmethod
    def starts_in_reasoning(prompt):
        return prompt.rfind("<think>") > prompt.rfind("</think>")

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        if images_of(messages):
            raise ValueError("this llama-server backend is configured for text only")
        # Strata normalizes arguments to mappings; the native OpenAI endpoint
        # expects JSON strings and function tools wrapped with their type.
        turns = []
        for message in drop_empty_assistant_turns(messages):
            turn = dict(message)
            if turn.get("tool_calls"):
                turn["tool_calls"] = [{**call, "type": "function", "function": {
                    **call["function"], "arguments": json.dumps(call["function"]["arguments"])}}
                    for call in turn["tool_calls"]]
            turns.append(turn)
        body = {"messages": turns, "add_generation_prompt": add_generation_prompt,
                "chat_template_kwargs": kwargs}
        if tools:
            body["tools"] = [{"type": "function", "function": tool} for tool in tools]
        return self.client.request("/apply-template", body)["prompt"]


class LlamaServerBackend:
    def __init__(self, client, props, eos_id):
        self.client, self.eos_id = client, eos_id
        self.max_context = int(props["default_generation_settings"]["n_ctx"])
        self.info = {"backend": "llama-server-metal", "version": props.get("build_info"),
                     "expert_streaming": False, "model_path": props.get("model_path")}
        self.last, self.progress = {}, None

    def generate(self, ids, max_new, sampling, cancel):
        if cancel.is_set():
            return
        self.last, self.progress = {}, None
        body = {"prompt": list(ids), "n_predict": max_new, "stream": True,
                "return_tokens": True, "return_progress": True, "cache_prompt": True}
        for key in ("temperature", "top_p", "top_k", "min_p", "seed",
                    "frequency_penalty", "presence_penalty"):
            if sampling.get(key) is not None:
                body[key] = sampling[key]
        if sampling.get("repetition_penalty") is not None:
            body["repeat_penalty"] = sampling["repetition_penalty"]
        if sampling.get("penalty_last_n") is not None:
            body["repeat_last_n"] = sampling["penalty_last_n"]
        conn, events, stopped = self.client.connection(), queue.Queue(), threading.Event()
        # Connecting before the reader starts keeps the socket available for
        # cancellation even while getresponse() is waiting during prefill.
        conn.connect()
        transport = conn.sock

        def read():
            try:
                conn.request("POST", "/completion", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
                with conn.getresponse() as response:
                    if response.status != 200:
                        raise RuntimeError(f"llama-server completion: HTTP {response.status}: "
                                           + response.read().decode("utf-8", "replace"))
                    for line in response:
                        if stopped.is_set():
                            break
                        if line.startswith(b"data: "):
                            data = line[6:].strip()
                            if data == b"[DONE]":
                                break
                            events.put(json.loads(data))
            except Exception as error:
                if not stopped.is_set():
                    events.put(error)
            finally:
                events.put(None)

        worker = threading.Thread(target=read, daemon=True)
        worker.start()
        started, first, count, finished, seen = time.monotonic(), None, 0, False, 0
        try:
            while not cancel.is_set():
                try:
                    event = events.get(timeout=0.1)
                except queue.Empty:
                    yield None  # Strata heartbeat; also checks client disconnects.
                    continue
                if isinstance(event, Exception):
                    raise event
                if event is None:
                    if not finished:
                        raise RuntimeError("llama-server stream ended before its completion event")
                    break
                if event.get("error"):
                    raise RuntimeError(f"llama-server: {event['error']}")
                progress = event.get("prompt_progress")
                if progress:
                    self.progress = (progress["processed"], progress["total"])
                    # Native progress events can include a placeholder token 0.
                    # It is not model output (token 0 is a real '!' in Qwen).
                    continue
                # llama-server holds back a chunk whose text ends mid-UTF-8 and then lists only the newest
                # token in "tokens", so held-back ids (an emoji's first bytes, or a whole speculative step)
                # never appear. Its cumulative "tokens_predicted" counts them all: when it moves further than
                # "tokens" says, the chunk's text (which is complete) is tokenized instead. Otherwise the
                # server's own ids are used, so the continuation prompt matches its cache exactly.
                text, ids = event.get("content"), event.get("tokens", [])
                predicted = event.get("tokens_predicted")
                missing = predicted is None or predicted - seen > len(ids)
                if predicted is not None:
                    seen = predicted
                if text and missing:
                    ids = self.client.request("/tokenize", {"content": text, "add_special": False,
                                                            "parse_special": True})["tokens"]
                for token in ids:
                    first = first or time.monotonic()
                    count += 1
                    # Drain the final timing event before Strata breaks on EOS.
                    if token != self.eos_id:
                        yield token
                if event.get("stop"):
                    finished = True
                    timings = event.get("timings", {})
                    self.last = {"generated": count, "prompt_tokens": len(ids),
                                 "prompt_ms": timings.get("prompt_ms", 0),
                                 "decode_ms": timings.get("predicted_ms", 0),
                                 "reused": timings.get("cache_n"),
                                 "finish": event.get("stop_type")}
                    if event.get("stop_type") == "eos":
                        yield self.eos_id
                    break
        finally:
            stopped.set()
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()
            worker.join(timeout=1)
            self.progress = None
            if not self.last:
                ended = time.monotonic()
                self.last = {"generated": count, "prompt_tokens": len(ids),
                             "prompt_ms": ((first or ended) - started) * 1000,
                             "decode_ms": (ended - (first or ended)) * 1000}

    def close(self):
        # The launcher owns the native process. Never stop an attached service.
        pass


def create_backend(config):
    client = Client(config.get("llama_server_url", "http://127.0.0.1:8096"))
    client.request("/health")
    props = client.request("/props")
    tok = Tokenizer(client)
    eos = tok.encode(props["eos_token"], parse_special=True)
    if len(eos) != 1:
        raise ValueError("llama-server must expose a singleton EOS token")
    stops = set(eos)
    for special in ("<|im_end|>", "<|endoftext|>"):
        encoded = tok.encode(special, parse_special=True)
        if len(encoded) == 1:
            stops.add(encoded[0])
    return BackendBundle(LlamaServerBackend(client, props, eos[0]), tok, Template(client), stops)
