from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from eval_harness.collect.cases import Case, save_case
from eval_harness.collect.classify import classify_kind, kind_from_labels
from eval_harness.collect.curation import Candidate, Verdict, load_verdicts, save_pending
from eval_harness.collect.filters import Decision, decide, reverted_numbers
from eval_harness.collect.github import PullRequest, fetch_merged_prs
from eval_harness.collect.join import MatchStats, match_stats
from eval_harness.collect.sanitize import SecretsFound
from eval_harness.collect.single import case_from
from eval_harness.collect.split import assign_splits, write_splits
from eval_harness.collect.tickets import TicketSource, source_for
from eval_harness.config import ModelConfig, RepoConfig
from eval_harness.paths import cases_dir as default_cases_dir
from eval_harness.paths import raw_cache

MIN_PER_KIND = 20


class CollectReport(BaseModel):
    repo: str
    since: str
    until: str
    match: MatchStats
    tickets: int
    rejections: dict[str, int]
    accepted: int
    curated_out: dict[str, int]
    uncurated: list[str]
    written: list[str]
    skipped_build: dict[str, str]
    kind_split: dict[str, int]
    kind_sources: dict[str, int]
    secrets_hits: list[str]
    splits: dict[str, int] = Field(default_factory=dict)
    pending: list[Candidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


FetchPRs = Callable[..., list[PullRequest]]
Classifier = Callable[[str, str], str]
CaseBuilder = Callable[..., Case]


def collect(
    repo: RepoConfig,
    *,
    since: str,
    until: str,
    tiers: tuple[str, ...] = ("A",),
    refresh: bool = False,
    dry_run: bool = False,
    classifier_model: ModelConfig | None = None,
    verdicts: dict[str, Verdict] | None = None,
    fetch_prs: FetchPRs = fetch_merged_prs,
    tickets: TicketSource | None = None,
    classify: Classifier | None = None,
    case_builder: CaseBuilder = case_from,
    cases_dir: Path | None = None,
    splits_path: Path | None = None,
    save_pending_list: bool = True,
) -> CollectReport:
    cases_dir = cases_dir or default_cases_dir()
    cache_dir = raw_cache(repo.key)
    tickets = tickets if tickets is not None else source_for(repo)
    prs = fetch_prs(repo.github, since, until, cache_dir=cache_dir, refresh=refresh)
    stats = match_stats(prs, repo.linear_team)
    keyed: dict[str, list[PullRequest]] = {}
    for pr in prs:
        key, _ = tickets.key_for(pr)
        if key:
            keyed.setdefault(key.upper(), []).append(pr)
    issues = tickets.fetch(sorted(keyed), cache_dir=cache_dir, refresh=refresh)
    reverted = reverted_numbers(prs)
    verdicts = verdicts if verdicts is not None else load_verdicts()

    decisions: list[Decision] = []
    for pr in prs:
        key, _ = tickets.key_for(pr)
        issue = issues.get(key.upper()) if key else None
        n = len(keyed.get(key.upper(), [])) if key else 0
        decisions.append(decide(pr, issue, repo=repo, prs_for_key=n, reverted=reverted))
    rejections: dict[str, int] = {}
    for d in decisions:
        if not d.accept:
            rejections[d.rule or "?"] = rejections.get(d.rule or "?", 0) + 1
    accepted = [d for d in decisions if d.accept]

    by_number = {p.number: p for p in prs}
    curated_out: dict[str, int] = {}
    uncurated: list[str] = []
    pending: list[Candidate] = []
    written: list[str] = []
    skipped_build: dict[str, str] = {}
    kind_split: dict[str, int] = {}
    kind_sources: dict[str, int] = {}
    secrets_hits: list[str] = []
    cases: list[Case] = []
    for d in accepted:
        key = (d.key or "").upper()
        verdict = verdicts.get(key)
        if verdict is None:
            uncurated.append(key)
            pr = by_number[d.pr]
            counted = [
                f for f in pr.files if repo.is_test_file(f.path) or repo.is_code_file(f.path)
            ]
            pending.append(
                Candidate(
                    key=key,
                    pr=pr.number,
                    title=pr.title,
                    author=pr.author,
                    author_kind=repo.agent_authors.get(pr.author, "human"),
                    files=len(counted),
                    lines=sum(f.additions + f.deletions for f in counted),
                )
            )
            continue
        if verdict.tier not in tiers:
            curated_out[verdict.tier] = curated_out.get(verdict.tier, 0) + 1
            continue
        issue = issues[key]
        kind = kind_from_labels(issue.labels)
        source = "label"
        if kind is None and verdict.kind_review in ("bug_fix", "feature"):
            kind, source = verdict.kind_review, "review"
        if kind is None:
            if classify is None:
                if classifier_model is None:
                    skipped_build[key] = "no label, no reviewer kind, no classifier configured"
                    continue
                model = classifier_model
                kind = classify_kind(issue.title, issue.description, model=model)
                source = f"llm:{model.model}"
            else:
                kind, source = classify(issue.title, issue.description), "llm"
        try:
            case = case_builder(
                repo,
                by_number[d.pr],
                issue,
                tier=verdict.tier,
                kind=kind,
                kind_source=source,
                author_kind=verdict.author_kind,
                spec_level_heuristic=verdict.spec_level_heuristic,
            )
        except SecretsFound as e:
            secrets_hits.append(f"{key}: {e.pattern_name}")
            continue
        except ValueError as e:
            skipped_build[key] = str(e)[:200]
            continue
        cases.append(case)
        kind_split[kind] = kind_split.get(kind, 0) + 1
        kind_sources[source] = kind_sources.get(source, 0) + 1
        if not dry_run:
            save_case(case, cases_dir)
        written.append(key)

    warnings: list[str] = []
    if stats.rate < 0.70:
        warnings.append(
            f"only {stats.rate:.0%} of merged PRs name a ticket. Below 70% the case set is a "
            f"biased sample of the work: pass --force to proceed, or fix how PRs "
            f"reference issues"
        )
    for kind_name in ("bug_fix", "feature"):
        if kind_split.get(kind_name, 0) < MIN_PER_KIND:
            warnings.append(
                f"only {kind_split.get(kind_name, 0)} {kind_name} cases (< {MIN_PER_KIND})"
            )
    if secrets_hits:
        warnings.append(f"{len(secrets_hits)} case(s) not written because the secrets scan hit")
    # Written on a dry run too: choosing what to curate is what a dry run is for.
    if save_pending_list:
        save_pending(repo.key, pending)
    if uncurated:
        warnings.append(
            f"{len(uncurated)} PR(s) passed the rules but have no curation verdict, so no case "
            f"was written for them: review with `eval-harness curate --repo {repo.key}`"
        )
    assignments = assign_splits(cases)
    if not dry_run and cases:
        write_splits(assignments, splits_path) if splits_path else write_splits(assignments)
    split_counts = {
        "dev": sum(v == "dev" for v in assignments.values()),
        "holdout": sum(v == "holdout" for v in assignments.values()),
    }
    return CollectReport(
        repo=repo.key,
        since=since,
        until=until,
        match=stats,
        tickets=len(keyed),
        rejections=rejections,
        accepted=len(accepted),
        curated_out=curated_out,
        uncurated=sorted(uncurated),
        written=written,
        skipped_build=skipped_build,
        kind_split=kind_split,
        kind_sources=kind_sources,
        secrets_hits=secrets_hits,
        splits=split_counts,
        warnings=warnings,
        pending=pending,
    )
