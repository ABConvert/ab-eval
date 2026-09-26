from __future__ import annotations

import re
from difflib import SequenceMatcher

_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)
_COMMENT = re.compile(r"^\s*(//|#|\*|/\*|\*/)")


def touched_files(diff: str) -> set[str]:
    return {m.group(2) for m in _FILE.finditer(diff)}


def changed_lines(diff: str) -> list[str]:
    """Added and removed lines with whitespace collapsed; comment-only lines dropped."""
    out: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
            continue
        body = line[1:]
        if not body.strip() or _COMMENT.match(body):
            continue
        out.append(line[0] + " ".join(body.split()))
    return out


def patch_similarity(generated: str, human: str) -> float:
    """0-1: half file-set Jaccard, half sequence similarity of normalized changed lines."""
    gf, hf = touched_files(generated), touched_files(human)
    jaccard = len(gf & hf) / len(gf | hf) if (gf | hf) else 0.0
    gl, hl = changed_lines(generated), changed_lines(human)
    seq = SequenceMatcher(a=gl, b=hl, autojunk=False).ratio() if (gl or hl) else 0.0
    return round(0.5 * jaccard + 0.5 * seq, 4)
