from __future__ import annotations

import re


class SecretsFound(Exception):
    def __init__(self, pattern_name: str) -> None:
        super().__init__(f"secret pattern hit: {pattern_name}")
        self.pattern_name = pattern_name


SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    "github_token": re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    "github_pat": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    "shopify_token": re.compile(r"\bshp(at|ca|pa|ss)_[A-Fa-f0-9]{32}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "slack_token": re.compile(r"\bxox[abpr]-[A-Za-z0-9\-]{10,}\b"),
    "google_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "linear_key": re.compile(r"\blin_api_[A-Za-z0-9]{20,}\b"),
}

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_INTERNAL_URL = re.compile(
    r"https?://(?:[a-z0-9\-]+\.)?"
    r"(?:linear\.app|sentry\.io|notion\.(?:so|com)|loom\.com|atlassian\.net)"
    r"[^\s)>\]]*"
)


def sanitize_text(text: str) -> str:
    """Raise on secret shapes; redact emails, internal links and images."""
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            raise SecretsFound(name)
    text = _IMAGE.sub("[image]", text)
    text = _INTERNAL_URL.sub("<internal-url>", text)
    text = _EMAIL.sub("<email>", text)
    return text
