# File and browser tools in the macOS Strata chat

The Qwen IQ2_XS launcher now loads a filesystem MCP server. Tools are enabled
by default in the browser chat at http://127.0.0.1:8095/.

The model can read, write, list and search files under:

`~/Strata-workspace`

Put documents there, then ask Qwen to read or work on them. For example:
“List the files in my workspace” or “Create a file called notes.txt containing
my meeting notes.” Tool calls and their results appear in the conversation.

Sampling has a **Use tools from MCP servers** switch. Monitor shows the
connected server and available tool count. Changes are applied after reloading
the page; existing browser chats are preserved.

Configuration is in `config/qwen-iq2-xs.json`, under `mcp_servers` and `mcp`.
The optional per-server `tools` list selects which tool names are exposed;
omitting it exposes everything that server offers. The current six tools are
read_text_file, write_file, list_directory, search_files, get_file_info and
list_allowed_directories. Node.js is already installed and the filesystem
server package is pinned to 2026.8.31.

Additional MCP servers may use `command` / `args` for stdio, or `url` for
Streamable HTTP. Restart the launcher after editing server configuration.

API clients use their own tools unless they send `strata_mcp: true` to
`/v1/chat/completions`. Both streaming and non-streaming requests execute the
configured tools and return tool activity in `strata_mcp`. Foreign browser
origins and requests without application/json cannot enable local tools.

Tool results are capped at 6,000 characters and each answer allows up to eight
tool rounds. Incomplete tool calls are displayed but never executed. Tool
results and call history remain available to follow-up browser messages.

Validation: 51 automated checks passed across the MCP and macOS suites, including errors, timeouts,
cancellation, partial calls, origin checks, streamed and non-streamed tool
rounds. A real Qwen request called the filesystem server, read a test file and
returned its exact contents. Qwen also created a file through its write tool
and read the result back successfully.

## Web browsing

Qwen also has eight browser tools from Microsoft's Playwright MCP server,
pinned to `@playwright/mcp@0.0.83`. Together with the six filesystem tools,
Monitor shows 14 available tools.

The browser runs headlessly in an isolated Chrome session. Qwen can navigate
to URLs, read rendered pages, follow links, type into search forms, press keys,
go back, wait for content and manage tabs. It uses a separate session from
your personal browser. Browser-generated output stays in
`~/Strata-workspace/browser-output`.

Examples: “Open https://example.com and tell me what it says”, or
“Search the web for the official Python documentation, open the result and
include its source URL.” Search happens through a website's search form; no
search API key is needed. The existing MCP switch controls both file and web
tools. Qwen is not given the server's code execution or upload tools.

This uses the official server: https://github.com/microsoft/playwright-mcp

## DeepSeek Harness on this Mac

The desktop profile at `~/.dsh/profiles/desktop/cordis.patch.yml`
uses Strata at `http://127.0.0.1:8095/v1`, model `qwen3.8-flash-next`, with a
120,000-token context and 16,384 maximum output tokens. Qwen is the default
model. The highest reasoning option maps to `high`, which Strata supports.

Harness executes its own tools. Its `mcp-strata-local` plugin runs
`python -m serve.mcp_bridge config/qwen-iq2-xs.json`, using Strata's existing
server configuration and allowlists. This adds the same 14 tools as
`mcp__strata__files__...` and `mcp__strata__web__...`. The bridge starts a
separate isolated browser and filesystem server; it does not load a second
model. Its workspace restriction applies to these file tools; Harness's
built-in tools keep their own permissions and workspace settings.

Keep the Strata launcher running for model requests. Harness starts and stops
the tool bridge automatically. Provider and MCP configuration changes reload
through Harness's profile loader; use a new chat to see newly registered
tools. Existing chats and other model providers are preserved.
