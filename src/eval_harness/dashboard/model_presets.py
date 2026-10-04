"""Curated suggestions, not an allowlist or a promise of account access.

Checked 2026-10-04 against the vendors' model catalogs:
https://platform.claude.com/docs/en/models/overview
https://developers.openai.com/api/docs/models/gpt-6.1-sol
https://developers.openai.com/api/docs/models/gpt-6-astra

GPT-6.1 tool calling requires Responses, so suggest it through Codex rather than
our Chat Completions adapter. Custom IDs remain available for every provider.
"""

CLAUDE = (
    ("claude-opus-5-5", "Claude Opus 5.5"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("claude-fable-5-1", "Claude Fable 5.1"),
)
PRESETS = {
    "anthropic": CLAUDE,
    "claude-code": CLAUDE,
    "codex-cli": (
        ("gpt-6.1-sol", "GPT-6.1 Sol"),
        ("gpt-6-astra", "GPT-6 Astra"),
    ),
    "openai-compat": (),
}
