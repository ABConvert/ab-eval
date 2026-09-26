import pytest

from eval_harness.collect.sanitize import SecretsFound, sanitize_text


def test_redacts_emails_and_internal_urls() -> None:
    out = sanitize_text(
        "mail bob@example.com see https://linear.app/acme/issue/DEMO-1 "
        "and ![s](https://uploads.linear.app/x.png)"
    )
    assert "bob@example.com" not in out and "<email>" in out
    assert "linear.app" not in out and "<internal-url>" in out
    assert "[image]" in out


def test_raises_on_api_key_shapes() -> None:
    with pytest.raises(SecretsFound):
        sanitize_text("token sk-ant-api03-" + "a" * 40)
    with pytest.raises(SecretsFound):
        sanitize_text("ghp_" + "A" * 36)
    with pytest.raises(SecretsFound):
        sanitize_text("shpat_" + "0" * 32)


def test_keeps_shop_domains_and_code() -> None:
    text = "example-store.myshopify.com `if (idx >= 1)`"
    assert sanitize_text(text) == text
