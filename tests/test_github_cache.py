import json
from pathlib import Path

from eval_harness.collect.github import PullRequest, load_cache, save_cache, windows


def test_windows_are_at_most_14_days_and_cover_the_range() -> None:
    w = windows("2026-03-09", "2026-04-02")
    assert w[0][0] == "2026-03-09" and w[-1][1] == "2026-04-02"
    assert all(b >= a for a, b in w)
    assert len(w) == 4
    assert windows("2026-09-09", "2026-09-09") == [("2026-09-09", "2026-09-09")]


def test_cache_roundtrip(tmp_path: Path) -> None:
    pr = PullRequest(
        number=1,
        title="t",
        body="",
        head_ref="x/DEMO-1-y",
        author="a",
        merged_at="2026-09-02T00:00:00Z",
        created_at="2026-09-01T00:00:00Z",
        merge_commit="c" * 40,
        files=[],
        review_count=0,
        comment_count=0,
    )
    path = save_cache(tmp_path, "prs-2026-03-09-2026-09-09", [pr])
    assert json.loads(path.read_text())[0]["number"] == 1
    assert load_cache(tmp_path, "prs-2026-03-09-2026-09-09", PullRequest) == [pr]
    assert load_cache(tmp_path, "missing", PullRequest) is None


def test_an_empty_fetch_is_not_cached(tmp_path: Path, monkeypatch: object) -> None:
    """`gh` answers [] for a private repo the login cannot see; caching that hid the fix."""
    from eval_harness.collect import github

    monkeypatch.setattr(github, "_gh_window", lambda *a: [])  # type: ignore[attr-defined]
    assert (
        github.fetch_merged_prs("acme/private", "2026-09-01", "2026-09-02", cache_dir=tmp_path)
        == []
    )
    assert load_cache(tmp_path, "prs-2026-09-01-2026-09-02", PullRequest) is None
