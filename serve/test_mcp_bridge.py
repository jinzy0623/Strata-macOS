"""Verify the external MCP bridge's allowlist and JSON-RPC wire protocol."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from serve.mcp import StdioTransport

ROOT = Path(__file__).resolve().parents[1]


class BridgeTest(unittest.TestCase):
    def test_wire_allowlist_results_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"mcp_servers": {"fake": {
                "command": sys.executable, "args": [str(ROOT / "serve/mcp_fake_server.py")],
                "tools": ["echo", "fail"]}}}))
            transport = StdioTransport("bridge", {
                "command": sys.executable, "args": ["-m", "serve.mcp_bridge", str(config)],
                "cwd": str(ROOT)})
            try:
                transport.start()
                init = transport.request("initialize", {"protocolVersion": "2026-07-28"}, 10)
                self.assertEqual(init["protocolVersion"], "2025-06-18")
                transport.notify("notifications/initialized", {})
                tools = transport.request("tools/list", {}, 10)["tools"]
                self.assertEqual({t["name"] for t in tools}, {"fake__echo", "fake__fail"})
                result = transport.request("tools/call", {"name": "fake__echo", "arguments": {"text": "héllo"}}, 10)
                self.assertEqual(result["content"][0]["text"], "héllo")
                self.assertFalse(result["isError"])
                error = transport.request("tools/call", {"name": "fake__fail"}, 10)
                self.assertTrue(error["isError"])
                with self.assertRaisesRegex(Exception, "disallowed"):
                    transport.request("tools/call", {"name": "fake__big"}, 10)
                self.assertEqual(transport.request("ping", {}, 10), {})
            finally:
                transport.close()


if __name__ == "__main__":
    unittest.main()
