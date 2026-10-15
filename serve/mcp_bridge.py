"""Expose the configured Strata tool allowlists to another local MCP client.

Run: python -m serve.mcp_bridge config/qwen-iq2-xs.json
This starts tool servers only; it does not load another model.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading

from serve.mcp import McpCancelled, hub_from_config


class Bridge:
    def __init__(self, hub):
        self.hub = hub
        self.pending = {}
        self.lock = threading.Lock()
        self.output_lock = threading.Lock()

    def result(self, method, params, cancel):
        if method == "initialize":
            return {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                    "serverInfo": {"name": "strata-local-tools", "version": "1.0"}}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [{"name": t["name"], "description": t["description"],
                               "inputSchema": t["parameters"]} for t in self.hub.template_tools()]}
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments", {})
            if name not in self.hub.routes() or not isinstance(args, dict):
                raise ValueError("Unknown or disallowed tool, or invalid arguments")
            r = self.hub.call(name, args, cancel)
            return {"content": [{"type": "text", "text": r["text"]}], "isError": not r["ok"]}
        raise LookupError("Method not found")

    def dispatch(self, msg, output):
        if not isinstance(msg, dict):
            return
        method, params = msg.get("method"), msg.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        if "id" not in msg:
            if method == "notifications/cancelled":
                with self.lock:
                    event = self.pending.get(params.get("requestId"))
                    if event:
                        event.set()
            return
        rid, cancel = msg["id"], threading.Event()
        with self.lock:
            self.pending[rid] = cancel

        def answer():
            response = {"jsonrpc": "2.0", "id": rid}
            try:
                response["result"] = self.result(method, params, cancel)
            except McpCancelled:
                return
            except Exception as exc:
                code = -32602 if isinstance(exc, ValueError) else -32601 if isinstance(exc, LookupError) else -32603
                response["error"] = {"code": code, "message": str(exc)}
            finally:
                with self.lock:
                    self.pending.pop(rid, None)
            if not cancel.is_set():
                with self.output_lock:
                    output.write(json.dumps(response, ensure_ascii=False) + "\n")
                    output.flush()

        threading.Thread(target=answer, daemon=True).start()

    def close(self):
        with self.lock:
            for event in self.pending.values():
                event.set()
        self.hub.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    # The existing hub logs to stdout. Reserve the original stream for MCP,
    # and redirect all diagnostics (including child logs) to stderr.
    wire = sys.stdout
    sys.stdout = sys.stderr
    with open(args.config, encoding="utf-8-sig") as f:
        hub = hub_from_config(json.load(f))
    if hub is None:
        raise SystemExit("No MCP servers configured")
    bridge = Bridge(hub)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        hub.start(wait=True)
        failed = [s.name for s in hub.servers.values() if s.status != "ready"]
        if failed:
            raise SystemExit("MCP servers failed to start: " + ", ".join(failed))
        for line in sys.stdin:
            try:
                bridge.dispatch(json.loads(line), wire)
            except json.JSONDecodeError:
                with bridge.output_lock:
                    wire.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                           "error": {"code": -32700, "message": "Parse error"}}) + "\n")
                    wire.flush()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
