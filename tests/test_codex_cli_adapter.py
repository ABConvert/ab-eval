from pathlib import Path

from eval_harness.adapters.codex_cli import CodexCliAdapter, parse_events, usage_from_events
from eval_harness.config import ModelConfig

EVENTS = [
    '{"type":"thread.started","thread_id":"t1"}',
    '{"type":"item.completed","item":{"id":"i1","type":"agent_message","text":"looking"}}',
    '{"type":"item.completed","item":{"id":"i2","type":"mcp_tool_call","server":"sandbox","tool":"bash","arguments":{"command":"ls"}}}',
    '{"type":"item.completed","item":{"id":"i3","type":"agent_message","text":"done"}}',
    '{"type":"turn.completed","usage":{"input_tokens":35752,"cached_input_tokens":27136,"cache_write_input_tokens":0,"output_tokens":131,"reasoning_output_tokens":9}}',
]


def test_parse_events_folds_usage_tools_and_text() -> None:
    parsed = parse_events(EVENTS)
    assert parsed["tool_calls"] == 1 and parsed["turns"] == 2
    assert parsed["final_text"] == "done" and parsed["failed"] is None
    u = usage_from_events(parsed["usage"])
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens) == (35752 - 27136, 27136, 131)


def test_parse_events_records_failure() -> None:
    parsed = parse_events(['{"type":"turn.failed","error":{"message":"boom"}}'])
    assert parsed["failed"] is not None and "boom" in parsed["failed"]


def test_command_disables_host_execution_and_points_at_mcp_server(tmp_path: Path) -> None:
    adapter = CodexCliAdapter(ModelConfig(key="t", provider="codex-cli", model="gpt-5.6-terra"))
    cmd = adapter.command(
        scratch=tmp_path,
        instructions=tmp_path / "i.md",
        mcp_args=["--cid", "c"],
        log_path=tmp_path / "l",
        tool_timeout=10,
    )
    joined = " ".join(cmd)
    for feature in ("shell_tool", "unified_exec", "apps", "browser_use", "collaboration_modes"):
        assert f"--disable {feature}" in joined
    assert "--disable code_mode_host" not in joined  # required for MCP calls in this build
    assert "--approve-for-me" in joined and "--sandbox" not in joined
    assert "--ignore-user-config" in joined and "--ephemeral" in joined
    assert 'web_search="disabled"' in joined
    assert "mcp_servers.sandbox.args=" in joined and "eval_harness.adapters.mcp_server" in joined
    assert cmd[-1] == "-"


def test_parse_events_flags_host_execution() -> None:
    parsed = parse_events(
        [
            '{"type":"item.completed","item":{"id":"x","type":"command_execution",'
            '"command":"ls /","exit_code":0}}'
        ]
    )
    assert parsed["host_exec"][0]["command"] == "ls /"
