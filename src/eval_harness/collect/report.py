from __future__ import annotations

import json
from pathlib import Path

from eval_harness.collect.filters import RULES
from eval_harness.collect.pipeline import CollectReport
from eval_harness.paths import collect_report_path


def render_text(r: CollectReport) -> str:
    m = r.match
    lines = [
        f"collect {r.repo} {r.since}..{r.until}",
        f"  merged PRs            {m.total}",
        f"  with ticket key       {m.with_key} ({m.rate:.0%}); non-bot "
        f"{m.human_with_key}/{m.human_total} ({m.human_rate:.0%}); by source {m.by_source}",
        f"  distinct tickets      {r.tickets}",
        f"  rejected by rule      {dict(sorted(r.rejections.items()))}",
        f"  passed structure      {r.accepted}",
        f"  curated out (tier)    {r.curated_out}; uncurated {len(r.uncurated)}",
        f"  skipped at build      {len(r.skipped_build)}"
        + (f" {r.skipped_build}" if r.skipped_build else ""),
        f"  secrets hits          {len(r.secrets_hits)}"
        + (f" {r.secrets_hits}" if r.secrets_hits else ""),
        f"  cases written         {len(r.written)}  kind split {r.kind_split}  "
        f"kind sources {r.kind_sources}",
        f"  splits                {r.splits}",
    ]
    if r.rejections:
        lines.append("  rules:")
        for code, n in sorted(r.rejections.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {code:3} {n:5}  {RULES.get(code, '')}")
    if r.pending:
        lines.append("  awaiting curation:")
        for c in r.pending:
            lines.append(
                f"    {c.key:12} #{c.pr:<6} {c.files:2} files {c.lines:4} lines  {c.title[:60]}"
            )
    for w in r.warnings:
        lines.append(f"  WARNING: {w}")
    return "\n".join(lines)


def write_report(r: CollectReport, path: Path | None = None) -> Path:
    path = path or collect_report_path()
    path.write_text(json.dumps(r.model_dump(), indent=2) + "\n")
    return path
