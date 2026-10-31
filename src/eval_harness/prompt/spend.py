"""The spend ledger for prompt-task commands: one file per month unless one is named."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from eval_harness.adapters.oneshot import LEDGER_ENV, LIMIT_ENV
from eval_harness.paths import results_root


def ensure_monthly_ledger(now: datetime | None = None) -> Path | None:
    """With a limit but no ledger, count against results/_spend/YYYY-MM.jsonl.

    A limit with nothing to count against is refused by oneshot, which is right for the
    one-off judge commands that predate this. A scheduled benchmark wants a monthly budget
    without every cron line naming a file, so prompt-task commands supply one. An explicit
    ABEVAL_SPEND_LEDGER always wins.
    """
    if os.environ.get(LEDGER_ENV):
        return Path(os.environ[LEDGER_ENV])
    if not os.environ.get(LIMIT_ENV):
        return None
    month = (now or datetime.now(UTC)).strftime("%Y-%m")
    path = results_root() / "_spend" / f"{month}.jsonl"
    os.environ[LEDGER_ENV] = str(path)
    return path
